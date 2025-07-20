"""Evaluator that uses an LLM to grade other model outputs."""

from __future__ import annotations

from pathlib import Path
import json
import re
import pandas as pd
from openai import APIConnectionError, APITimeoutError, BadRequestError
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .base_evaluator import BaseEvaluator

# Persian digits → ASCII
PERS_TO_ASCII = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
# capture first number (Persian or ASCII) possibly with decimal point
SCORE_REGEX = re.compile(r"([\d\u06F0-\u06F9]+(?:\.\d+)?)")


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

    def _build_prompt(self, df: pd.DataFrame, row: pd.Series) -> str:
        ref = str(row[self.acol]) if self.acol else ""
        if not self.use_reference:
            ref = ""
        query = {
            "question": row[self.qcol],
            "reference": ref,
            "candidate": str(row[self.cand_col]),
        }
        return self.template.render(**query)

    def _extract(self, text: str) -> tuple[str | None, str]:
        """Return ``(score, reason)`` parsed from ``text``.

        The preferred format is ``****<score>: <reason>****`` but JSON objects
        like ``{"score": 7, "reason": "..."}`` are also supported.  If neither
        pattern matches, only the numeric part (if any) is returned.
        """

        text = text.strip()
        if not text:
            return None, ""

        # Attempt JSON parsing first -----------------------------------------
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "score" in data:
                score = str(data["score"]).translate(PERS_TO_ASCII).strip()
                reason = str(data.get("reason", "")).strip()
                return score, reason
        except json.JSONDecodeError:
            pass

        clean = text.strip("*").strip()
        m = SCORE_REGEX.search(clean)
        if not m:
            return None, clean

        score = m.group(1).translate(PERS_TO_ASCII).strip()
        reason = clean[m.end():].lstrip(" :-\u061f\u060c\u061b\u0020")
        return score, reason.strip()

    def evaluate_df(self, df: pd.DataFrame, *, max_workers: int = 4) -> pd.DataFrame:
        """Return a copy of *df* with ``pred``, ``reason`` and ``raw`` columns."""
        preds: list[str | None] = [None] * len(df)
        reasons: list[str] = [""] * len(df)
        raws: list[str] = [""] * len(df)

        def _worker(idx: int, row: pd.Series):
            prompt = self._build_prompt(df, row)
            for attempt in range(1, self.max_retries + 1):
                try:
                    text = self._query_model(prompt)
                    score, reason = self._extract(text)
                    return idx, score, reason, text
                except (APIConnectionError, APITimeoutError, BadRequestError) as e:
                    self.logger.warning(
                        "row %d retry %d/%d: %s", idx, attempt, self.max_retries, e
                    )
            return idx, None, "", ""

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(_worker, idx, row): idx
                for idx, (_, row) in enumerate(df.iterrows())
            }
            for fut in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc="LLM requests",
                leave=False,
            ):
                i, pred, reason, raw = fut.result()
                preds[i] = pred
                reasons[i] = reason
                raws[i] = raw

        out = df.copy()
        out["pred"] = preds
        out["reason"] = reasons
        out["raw"] = raws
        return out
