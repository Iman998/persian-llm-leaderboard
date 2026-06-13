"""Evaluator that uses an LLM to grade other model outputs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re

import pandas as pd
from openai import APIConnectionError, APITimeoutError, BadRequestError
from tqdm import tqdm

from .base_evaluator import BaseEvaluator


PERS_TO_ASCII = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
SCORE_REGEX = re.compile(r"([\d\u06F0-\u06F9]+(?:\.\d+)?)")
SECTION_KEY = re.compile(r"[^a-z0-9]+")


class JudgeEvaluator(BaseEvaluator):
    """Use an LLM judge to score candidate answers."""

    def __init__(
        self,
        *,
        model_cfg: dict,
        prompt_path: Path,
        meta_path: Path,
        shots: int = 0,
        shot_seed: int = 42,
        max_retries: int = 3,
        use_reference: bool | None = None,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,
            prompt_path=prompt_path,
            meta_path=meta_path,
            shots=shots,
            shot_seed=shot_seed,
            max_retries=max_retries,
        )
        self.cand_col: str = self.meta.get("candidate_col", "candidate")
        self.use_reference: bool = (
            use_reference
            if use_reference is not None
            else self.meta.get("use_reference", True)
        )
        self.source_language_col: str | None = self.meta.get("source_language_col")
        self.target_language_col: str | None = self.meta.get("target_language_col")
        self.score_min = self.meta.get("judge_score_min")
        self.score_max = self.meta.get("judge_score_max")

    def _build_prompt(self, df: pd.DataFrame, row: pd.Series) -> str:
        ref = str(row[self.acol]) if self.acol else ""
        if not self.use_reference:
            ref = ""
        query = {
            "question": row[self.qcol],
            "reference": ref,
            "candidate": str(row[self.cand_col]),
            "source_language": (
                str(row[self.source_language_col])
                if self.source_language_col
                else ""
            ),
            "target_language": (
                str(row[self.target_language_col])
                if self.target_language_col
                else ""
            ),
        }
        return self.template.render(**query)

    def _extract(self, text: str) -> tuple[str | None, str]:
        """Return ``(score, reason)`` parsed from a judge response."""
        details = self._extract_details(text)
        return details["score"], str(details["reason"])

    @staticmethod
    def _json_object(text: str) -> dict | None:
        """Return the first complete JSON object embedded in *text*."""
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escaped = False
        for index, char in enumerate(text[start:], start=start):
            if escaped:
                escaped = False
                continue
            if char == "\\" and in_string:
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        return None
                    return value if isinstance(value, dict) else None
        return None

    def _normalize_score(self, value: object) -> str | None:
        """Normalize a numeric score and enforce configured bounds."""
        score = str(value).translate(PERS_TO_ASCII).strip()
        try:
            numeric = float(score)
        except ValueError:
            return None
        if self.score_min is not None and numeric < float(self.score_min):
            return None
        if self.score_max is not None and numeric > float(self.score_max):
            return None
        return str(int(numeric)) if numeric.is_integer() else str(numeric)

    def _extract_details(self, text: str) -> dict[str, object]:
        """Parse score, reason, and optional rubric section scores."""
        text = (text or "").strip()
        if not text:
            return {"score": None, "reason": "", "section_scores": {}}

        data = self._json_object(text)
        if data is not None and "score" in data:
            score = self._normalize_score(data["score"])
            reason = str(data.get("reason", "")).strip()
            sections: dict[str, str] = {}
            raw_sections = data.get("section_scores", {})
            if isinstance(raw_sections, dict):
                for name, value in raw_sections.items():
                    normalized = self._normalize_score(value)
                    key = SECTION_KEY.sub("_", str(name).lower()).strip("_")
                    if key and normalized is not None:
                        sections[key] = normalized
            return {
                "score": score,
                "reason": reason,
                "section_scores": sections,
            }

        clean = text.strip("*").strip()
        match = SCORE_REGEX.search(clean)
        if not match:
            return {"score": None, "reason": clean, "section_scores": {}}

        score = self._normalize_score(match.group(1))
        reason = clean[match.end() :].lstrip(" :-\u061f\u060c\u061b\u0020")
        return {
            "score": score,
            "reason": reason.strip(),
            "section_scores": {},
        }

    def evaluate_df(self, df: pd.DataFrame, *, max_workers: int = 4) -> pd.DataFrame:
        """Return judge scores, rationales, raw responses, and rubric scores."""
        preds: list[str | None] = [None] * len(df)
        reasons: list[str] = [""] * len(df)
        raws: list[str] = [""] * len(df)
        section_results: list[dict[str, str]] = [{} for _ in range(len(df))]

        def _worker(idx: int, row: pd.Series):
            prompt = self._build_prompt(df, row)
            for attempt in range(1, self.max_retries + 1):
                try:
                    text = self._query_model(prompt)
                    details = self._extract_details(text)
                    if details["score"] is not None:
                        return idx, details, text
                    self.logger.warning(
                        "row %d retry %d/%d: judge returned no valid score",
                        idx,
                        attempt,
                        self.max_retries,
                    )
                except (APIConnectionError, APITimeoutError, BadRequestError) as exc:
                    self.logger.warning(
                        "row %d retry %d/%d: %s",
                        idx,
                        attempt,
                        self.max_retries,
                        exc,
                    )
            return idx, {"score": None, "reason": "", "section_scores": {}}, ""

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(_worker, idx, row): idx
                for idx, (_, row) in enumerate(df.iterrows())
            }
            for future in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc="LLM requests",
                leave=False,
            ):
                index, details, raw = future.result()
                preds[index] = details["score"]
                reasons[index] = str(details["reason"])
                raws[index] = raw
                section_results[index] = details["section_scores"]

        out = df.copy()
        out["pred"] = preds
        out["reason"] = reasons
        out["raw"] = raws
        section_names = sorted(
            {name for sections in section_results for name in sections}
        )
        for name in section_names:
            out[f"judge_{name}_score"] = [
                sections.get(name) for sections in section_results
            ]
        return out
