"""Pairwise LLM judge for model battles."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
from openai import APIConnectionError, APITimeoutError, BadRequestError
from tqdm import tqdm

from .base_evaluator import BaseEvaluator


WINNER_PATTERN = re.compile(
    r"\b(candidate[_\s-]*a|candidate[_\s-]*b|model[_\s-]*1|"
    r"model[_\s-]*2|equal|tie|draw)\b",
    re.IGNORECASE,
)


class BattleEvaluator(BaseEvaluator):
    """Compare two anonymous candidate outputs and select a winner or tie."""

    def __init__(
        self,
        *,
        model_cfg: dict,
        prompt_path: Path,
        meta_path: Path,
        shots: int = 0,
        shot_seed: int = 42,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,
            prompt_path=prompt_path,
            meta_path=meta_path,
            shots=shots,
            shot_seed=shot_seed,
            max_retries=max_retries,
        )
        self.model_1_col = self.meta.get("model_1_col", "model_1_output")
        self.model_2_col = self.meta.get("model_2_col", "model_2_output")
        self.model_1_name = str(self.meta.get("model_1_name", "model_1"))
        self.model_2_name = str(self.meta.get("model_2_name", "model_2"))
        self.use_reference = bool(self.meta.get("use_reference", True))
        self.source_language_col = self.meta.get("source_language_col")
        self.target_language_col = self.meta.get("target_language_col")

    def _swap_candidates(self, row: pd.Series) -> bool:
        """Counterbalance candidate order deterministically for each row."""
        try:
            return int(row.name) % 2 == 1
        except (TypeError, ValueError):
            pass
        seed = str(row.get(self.qcol, row.name)).encode("utf-8")
        return hashlib.sha256(seed).digest()[0] % 2 == 1

    def _presentation(self, row: pd.Series) -> tuple[str, str, bool]:
        swap = self._swap_candidates(row)
        model_1_output = str(row[self.model_1_col])
        model_2_output = str(row[self.model_2_col])
        if swap:
            return model_2_output, model_1_output, True
        return model_1_output, model_2_output, False

    def _build_prompt(self, df: pd.DataFrame, row: pd.Series) -> str:
        candidate_a, candidate_b, _ = self._presentation(row)
        reference = str(row[self.acol]) if self.acol and self.use_reference else ""
        return self.template.render(
            question=row[self.qcol],
            reference=reference,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            source_language=(
                str(row[self.source_language_col])
                if self.source_language_col
                else ""
            ),
            target_language=(
                str(row[self.target_language_col])
                if self.target_language_col
                else ""
            ),
        )

    @staticmethod
    def _extract_choice(text: str) -> tuple[str | None, str]:
        """Return ``candidate_a``, ``candidate_b``, or ``equal`` and a reason."""
        clean = (text or "").strip()
        if not clean:
            return None, ""

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, dict):
            raw_winner = data.get("winner")
            reason = str(data.get("reason", "")).strip()
            if raw_winner is not None:
                choice = BattleEvaluator._normalize_choice(str(raw_winner))
                return choice, reason

        match = WINNER_PATTERN.search(clean)
        if not match:
            return None, clean
        choice = BattleEvaluator._normalize_choice(match.group(1))
        reason = clean[match.end() :].lstrip(" :-")
        return choice, reason.strip()

    @staticmethod
    def _normalize_choice(value: str) -> str | None:
        normalized = re.sub(r"[\s-]+", "_", value.strip().lower())
        aliases = {
            "candidate_a": "candidate_a",
            "candidate_b": "candidate_b",
            "model_1": "candidate_a",
            "model_2": "candidate_b",
            "equal": "equal",
            "tie": "equal",
            "draw": "equal",
        }
        return aliases.get(normalized)

    def _canonical_outcome(self, choice: str, swapped: bool) -> tuple[str, str]:
        if choice == "equal":
            return "equal", "equal"
        model_1_won = (choice == "candidate_a" and not swapped) or (
            choice == "candidate_b" and swapped
        )
        if model_1_won:
            return "model_1", self.model_1_name
        return "model_2", self.model_2_name

    def evaluate_df(self, df: pd.DataFrame, *, max_workers: int = 4) -> pd.DataFrame:
        """Return row-level canonical battle outcomes and judge details."""
        outcomes: list[str | None] = [None] * len(df)
        winner_models: list[str | None] = [None] * len(df)
        reasons: list[str] = [""] * len(df)
        raws: list[str] = [""] * len(df)
        presentation_orders: list[str] = [""] * len(df)

        def _worker(index: int, row: pd.Series):
            _, _, swapped = self._presentation(row)
            prompt = self._build_prompt(df, row)
            for attempt in range(1, self.max_retries + 1):
                try:
                    text = self._query_model(prompt)
                    choice, reason = self._extract_choice(text)
                    if choice is not None:
                        outcome, winner_model = self._canonical_outcome(
                            choice, swapped
                        )
                        order = "model_2|model_1" if swapped else "model_1|model_2"
                        return (
                            index,
                            outcome,
                            winner_model,
                            reason,
                            text,
                            order,
                        )
                    self.logger.warning(
                        "row %d retry %d/%d: battle judge returned no valid winner",
                        index,
                        attempt,
                        self.max_retries,
                    )
                except (APIConnectionError, APITimeoutError, BadRequestError) as exc:
                    self.logger.warning(
                        "row %d retry %d/%d: %s",
                        index,
                        attempt,
                        self.max_retries,
                        exc,
                    )
            return index, None, None, "", "", ""

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(_worker, index, row): index
                for index, (_, row) in enumerate(df.iterrows())
            }
            for future in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc="Battle judge requests",
                leave=False,
            ):
                index, outcome, winner_model, reason, raw, order = future.result()
                outcomes[index] = outcome
                winner_models[index] = winner_model
                reasons[index] = reason
                raws[index] = raw
                presentation_orders[index] = order

        out = df.copy()
        out["pred"] = outcomes
        out["winner_model"] = winner_models
        out["reason"] = reasons
        out["raw"] = raws
        out["presentation_order"] = presentation_orders
        return out
