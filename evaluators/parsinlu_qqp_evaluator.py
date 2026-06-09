"""Evaluator for ParsiNLU question-pair paraphrase classification."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .base_evaluator import BaseEvaluator


_LABEL_RE = re.compile(
    r"(?:برچسب|label|answer)\s*:\s*\*{0,4}\s*([01۰۱])\s*\*{0,4}",
    re.IGNORECASE,
)
_PERSIAN_DIGITS = str.maketrans("۰۱", "01")


class ParsinLUQQPEvaluator(BaseEvaluator):
    """Classify whether two Persian questions have the same meaning."""

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
        self.question_2_col = self.meta.get("question_2_col", "question_2")

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
                "question_1": df.loc[index, self.qcol],
                "question_2": df.loc[index, self.question_2_col],
                "answer": str(df.loc[index, self.acol]),
            }
            for index in shot_ids
        ]
        return self.template.render(
            shots=shots,
            question_1=row[self.qcol],
            question_2=row[self.question_2_col],
        )

    def _extract(self, text: str | None) -> str | None:
        if not text:
            return None
        matches = _LABEL_RE.findall(text)
        if matches:
            return matches[-1].translate(_PERSIAN_DIGITS)

        stripped = text.strip().translate(_PERSIAN_DIGITS)
        return stripped if stripped in {"0", "1"} else None
