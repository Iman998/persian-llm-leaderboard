"""Evaluator for GSM8K grade-school mathematical reasoning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from leaderboard_lib.gsm8k_utils import extract_gsm8k_answer

from .base_evaluator import BaseEvaluator


class Gsm8kEvaluator(BaseEvaluator):
    """Prompt for step-by-step solutions and extract the final number."""

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
        self.solution_col = self.meta.get("solution_col", "solution")

    def _build_prompt(self, df: pd.DataFrame, row: pd.Series) -> str:
        shot_ids = (
            self.rng.sample(
                [index for index in df.index if index != row.name],
                k=min(self.shots, len(df) - 1),
            )
            if self.shots
            else []
        )
        shots = [
            {
                "question": df.loc[index, self.qcol],
                "solution": df.loc[index, self.solution_col],
            }
            for index in shot_ids
        ]
        return self.template.render(
            shots=shots,
            question=row[self.qcol],
        )

    def _extract(self, text: str | None) -> str | None:
        return extract_gsm8k_answer(text)
