"""Evaluator for the GPQA graduate-level science benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .mcq_evaluator import MCQEvaluator


class GpqaEvaluator(MCQEvaluator):
    """Evaluate GPQA questions with domain-matched few-shot examples."""

    SYSTEM_PROMPT = (
        "You are answering graduate-level multiple-choice questions in biology, "
        "chemistry, and physics. Work through the scientific details carefully, "
        "then select exactly one of the four options."
    )

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
        self.domain_col = self.meta.get("domain_col", "domain")
        self.explanation_col = self.meta.get("explanation_col", "explanation")

    def _build_prompt(self, df: pd.DataFrame, row: pd.Series) -> str:
        candidates = [
            index
            for index in df.index
            if index != row.name
            and df.loc[index, self.domain_col] == row[self.domain_col]
        ]
        shot_ids = (
            self.rng.sample(candidates, k=min(self.shots, len(candidates)))
            if self.shots
            else []
        )
        shots = [
            {
                "question": df.loc[index, self.qcol],
                "choices": [
                    df.loc[index, column]
                    for column in self.choice_cols
                    if pd.notna(df.loc[index, column])
                ],
                "answer": int(df.loc[index, self.acol]),
                "rationale": df.loc[index, self.explanation_col],
            }
            for index in shot_ids
        ]
        return self.template.render(
            question=row[self.qcol],
            choices=[
                row[column]
                for column in self.choice_cols
                if pd.notna(row[column])
            ],
            shots=shots,
        )

    def _extract(self, text: str | None) -> int | None:
        """Return a numeric option ID compatible with the accuracy metric."""
        option = super()._extract(text)
        return int(option) if option is not None else None
