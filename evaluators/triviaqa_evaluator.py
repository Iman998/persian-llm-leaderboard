"""Evaluator for open-domain TriviaQA questions."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from leaderboard_lib.triviaqa_utils import parse_answer_aliases

from .open_ended_evaluator import OpenEndedEvaluator


class TriviaQAEvaluator(OpenEndedEvaluator):
    """Produce and extract concise answers for TriviaQA."""

    ANSWER_RE = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$")

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
        self.canonical_answer_col = self.meta.get(
            "canonical_answer_col", "canonical_answer"
        )

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
                "answer": self._canonical_answer(df.loc[index]),
            }
            for index in shot_ids
        ]
        return self.template.render(shots=shots, question=row[self.qcol])

    def _canonical_answer(self, row: pd.Series) -> str:
        answer = row[self.canonical_answer_col]
        if pd.isna(answer):
            aliases = parse_answer_aliases(row[self.acol])
            return aliases[0] if aliases else ""
        return str(answer)

    def _extract(self, text: str | None) -> str | None:
        if not text:
            return None
        matches = self.ANSWER_RE.findall(text)
        answer = matches[-1].strip() if matches else None
        if answer is None:
            nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
            answer = nonempty_lines[-1] if nonempty_lines else None
        if answer and answer.lower() in {"none", "nan", "null", "n/a"}:
            return answer + "."
        return answer
