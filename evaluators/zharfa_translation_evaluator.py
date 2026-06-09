"""Evaluator for the multilingual Zharfa Translation benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base_evaluator import BaseEvaluator


class ZharfaTranslationEvaluator(BaseEvaluator):
    """Translate each row using its explicit source and target languages."""

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
        self.source_language_col = self.meta.get(
            "source_language_col", "src_language"
        )
        self.target_language_col = self.meta.get(
            "target_language_col", "tgt_lang"
        )

    def _build_prompt(self, df: pd.DataFrame, row: pd.Series) -> str:
        source_language = row[self.source_language_col]
        target_language = row[self.target_language_col]
        candidates = [
            index
            for index in df.index
            if index != row.name
            and df.loc[index, self.source_language_col] == source_language
            and df.loc[index, self.target_language_col] == target_language
        ]
        shot_ids = (
            self.rng.sample(candidates, k=min(self.shots, len(candidates)))
            if self.shots
            else []
        )
        shots = [
            {
                "question": df.loc[index, self.qcol],
                "answer": df.loc[index, self.acol],
                "src_language": df.loc[index, self.source_language_col],
                "tgt_language": df.loc[index, self.target_language_col],
            }
            for index in shot_ids
        ]
        return self.template.render(
            shots=shots,
            question=row[self.qcol],
            src_language=source_language,
            tgt_language=target_language,
        )
