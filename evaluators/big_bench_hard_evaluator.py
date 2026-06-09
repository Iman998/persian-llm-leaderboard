"""Evaluator for the BIG-Bench Hard reasoning benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from leaderboard_lib.bbh_utils import extract_bbh_answer, make_bbh_scoring_key

from .base_evaluator import BaseEvaluator


class BigBenchHardEvaluator(BaseEvaluator):
    """Evaluate BBH with its official task-specific chain-of-thought prompts."""

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
        self.task_col = self.meta.get("task_col", "task")
        self.family_col = self.meta.get("task_family_col", "task_family")
        prompt_dir = Path(self.meta.get("cot_prompt_dir", "prompts/big_bench_hard"))
        self.task_prompts = {
            path.stem: self._parse_task_prompt(path)
            for path in sorted(prompt_dir.glob("*.txt"))
        }
        if not self.task_prompts:
            raise FileNotFoundError(f"No BIG-Bench Hard prompts found in {prompt_dir}")

    @staticmethod
    def _parse_task_prompt(path: Path) -> tuple[str, list[str]]:
        text = path.read_text(encoding="utf-8")
        body = text.split("-----", 1)[-1].strip()
        parts = body.split("\n\nQ:")
        instruction = parts[0].strip()
        demonstrations = [f"Q:{part}".strip() for part in parts[1:]]
        return instruction, demonstrations

    def _build_prompt(self, df: pd.DataFrame, row: pd.Series) -> str:
        task = str(row[self.task_col])
        if task not in self.task_prompts:
            raise KeyError(f"No BIG-Bench Hard prompt found for task '{task}'")
        instruction, demonstrations = self.task_prompts[task]
        return self.template.render(
            instruction=instruction,
            demonstrations=demonstrations[: min(self.shots, len(demonstrations))],
            question=row[self.qcol],
        )

    def _extract(self, text: str | None) -> str | None:
        return extract_bbh_answer(text)

    def evaluate_df(self, df: pd.DataFrame, *, max_workers: int = 4) -> pd.DataFrame:
        out = super().evaluate_df(df, max_workers=max_workers)
        out["answer_prediction"] = out["pred"]
        out["pred"] = [
            make_bbh_scoring_key(task, family, prediction)
            for task, family, prediction in zip(
                out[self.task_col],
                out[self.family_col],
                out["answer_prediction"],
            )
        ]
        return out
