"""Evaluator for tasks that require free-form text generation."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from .open_ended_evaluator import OpenEndedEvaluator


class TextGenerationEvaluator(OpenEndedEvaluator):
    """Evaluator for summarization and translation tasks."""

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
