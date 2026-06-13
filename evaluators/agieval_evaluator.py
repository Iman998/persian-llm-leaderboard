"""Evaluator for the mixed multiple-choice and cloze AGIEval benchmark."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from leaderboard_lib.agieval_utils import (
    make_agieval_scoring_key,
    normalize_agieval_choice,
)
from leaderboard_lib.math_utils import extract_math_answer

from .base_evaluator import BaseEvaluator


_FINAL_CHOICE_RE = re.compile(
    r"(?i:(?:final\s+answer|answer|答案)\s*(?:is|是|:|：)?\s*)"
    r"[\(\[\{]?\s*([A-G](?:\s*[,;/、]?\s*[A-G])*)",
)


class AgievalEvaluator(BaseEvaluator):
    """Evaluate AGIEval rows with task-matched few-shot examples."""

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
        self.choice_cols = self.meta.get(
            "choice_cols",
            ["choice1", "choice2", "choice3", "choice4", "choice5"],
        )
        self.task_col = self.meta.get("task_col", "task")
        self.type_col = self.meta.get("question_type_col", "question_type")
        self.language_col = self.meta.get("language_col", "language")
        self.passage_col = self.meta.get("passage_col", "passage")
        self.target_col = self.meta.get("target_col", "target")

    def _choices(self, row: pd.Series) -> list[str]:
        return [
            str(row[column])
            for column in self.choice_cols
            if column in row and pd.notna(row[column]) and str(row[column]).strip()
        ]

    def _build_prompt(self, df: pd.DataFrame, row: pd.Series) -> str:
        task = row[self.task_col]
        question_type = row[self.type_col]
        candidates = [
            index
            for index in df.index
            if index != row.name
            and df.loc[index, self.task_col] == task
            and df.loc[index, self.type_col] == question_type
        ]
        shot_ids = (
            self.rng.sample(candidates, k=min(self.shots, len(candidates)))
            if self.shots
            else []
        )
        shots = [
            {
                "passage": df.loc[index, self.passage_col],
                "question": df.loc[index, self.qcol],
                "choices": self._choices(df.loc[index]),
                "answer": df.loc[index, self.target_col],
            }
            for index in shot_ids
        ]
        return self.template.render(
            language=row[self.language_col],
            question_type=question_type,
            passage=row[self.passage_col],
            question=row[self.qcol],
            choices=self._choices(row),
            shots=shots,
        )

    def _extract(self, text: str | None) -> str | None:
        if not text:
            return None
        matches = _FINAL_CHOICE_RE.findall(text)
        if matches:
            return normalize_agieval_choice(matches[-1]) or None
        return extract_math_answer(text)

    def evaluate_df(self, df: pd.DataFrame, *, max_workers: int = 4) -> pd.DataFrame:
        out = super().evaluate_df(df, max_workers=max_workers)
        out["answer_prediction"] = out["pred"]
        out["pred"] = [
            make_agieval_scoring_key(task, question_type, prediction)
            for task, question_type, prediction in zip(
                out[self.task_col],
                out[self.type_col],
                out["answer_prediction"],
            )
        ]
        return out
