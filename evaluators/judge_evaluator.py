from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

from .base_evaluator import BaseEvaluator

# Persian digits → ASCII
PERS_TO_ASCII = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
# capture first number (Persian or ASCII) possibly with decimal point
SCORE_REGEX = re.compile(r"([\d\u06F0-\u06F9]+(?:\.\d+)?)")


class JudgeEvaluator(BaseEvaluator):
    """Use an LLM judge to score candidate answers."""

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
        self.cand_col: str = self.meta.get("candidate_col", "candidate")

    def _build_prompt(self, df: pd.DataFrame, row: pd.Series) -> str:
        query = {
            "question": row[self.qcol],
            "reference": str(row[self.acol]) if self.acol else "",
            "candidate": str(row[self.cand_col]),
        }
        return self.template.render(**query)

    def _extract(self, text: str) -> str | None:
        m = SCORE_REGEX.search(text)
        if not m:
            return None
        return m.group(1).translate(PERS_TO_ASCII).strip()
