"""Evaluator for DROP reading comprehension and discrete reasoning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from leaderboard_lib.drop_utils import parse_drop_annotations

from .triviaqa_evaluator import TriviaQAEvaluator


class DROPEvaluator(TriviaQAEvaluator):
    """Answer DROP questions using their supporting passages."""

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
        self.passage_col = self.meta.get("passage_col", "passage")

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
                "passage": df.loc[index, self.passage_col],
                "question": df.loc[index, self.qcol],
                "answer": self._canonical_answer(df.loc[index]),
            }
            for index in shot_ids
        ]
        return self.template.render(
            shots=shots,
            passage=row[self.passage_col],
            question=row[self.qcol],
        )

    def _canonical_answer(self, row: pd.Series) -> str:
        annotations = parse_drop_annotations(row[self.acol])
        first = annotations[0] if annotations else []
        return "; ".join(first)
