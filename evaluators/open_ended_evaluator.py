"""Evaluator base for short-answer question answering tasks."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from .base_evaluator import BaseEvaluator


class OpenEndedEvaluator(BaseEvaluator):
    """Evaluator for short-answer QA tasks."""

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

    def _build_prompt(self, df: pd.DataFrame, row: pd.Series) -> str:
        shot_ids = (
            self.rng.sample([i for i in df.index if i != row.name], k=min(self.shots, len(df) - 1))
            if self.shots
            else []
        )
        shots = [
            {"question": df.loc[i, self.qcol], "answer": str(df.loc[i, self.acol]) if self.acol else ""}
            for i in shot_ids
        ]
        query = {"question": row[self.qcol]}
        return self.template.render(shots=shots, **query)
