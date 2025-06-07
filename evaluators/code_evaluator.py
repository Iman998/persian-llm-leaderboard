"""Evaluator for code generation benchmarks."""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

from .open_ended_evaluator import OpenEndedEvaluator


class CodeEvaluator(OpenEndedEvaluator):
    """Evaluator for code generation tasks."""

    CODE_RE = re.compile(r"```(?:\w+\n)?(.*?)```", re.DOTALL)

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

    def _extract(self, text: str) -> str | None:
        m = self.CODE_RE.search(text)
        return m.group(1).strip() if m else text.strip()
