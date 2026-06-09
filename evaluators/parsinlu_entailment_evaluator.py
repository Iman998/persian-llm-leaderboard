"""Evaluator for ParsiNLU three-way natural language inference."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .base_evaluator import BaseEvaluator


_LABELS = ("entailment", "contradiction", "neutral")
_LABEL_RE = re.compile(
    r"(?:برچسب|label|answer)\s*:\s*\*{0,4}\s*"
    r"(entailment|contradiction|neutral)\s*\*{0,4}",
    re.IGNORECASE,
)


class ParsinLUEntailmentEvaluator(BaseEvaluator):
    """Classify the relation between a Persian premise and hypothesis."""

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
        self.hypothesis_col = self.meta.get("hypothesis_col", "sentence_2")

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
                "premise": df.loc[index, self.qcol],
                "hypothesis": df.loc[index, self.hypothesis_col],
                "answer": str(df.loc[index, self.acol]),
            }
            for index in shot_ids
        ]
        return self.template.render(
            shots=shots,
            premise=row[self.qcol],
            hypothesis=row[self.hypothesis_col],
        )

    def _extract(self, text: str | None) -> str | None:
        if not text:
            return None
        matches = _LABEL_RE.findall(text)
        if matches:
            return matches[-1].lower()

        stripped = text.strip().lower()
        return stripped if stripped in _LABELS else None
