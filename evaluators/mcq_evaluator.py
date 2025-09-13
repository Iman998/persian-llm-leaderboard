#!/usr/bin/env python3
"""
MCQEvaluator
============

Generic evaluator for Persian multiple-choice benchmarks.

Key features
------------
* Jinja2 templating with optional few-shot examples.
* Thread-pool concurrency (`evaluate_df(..., max_workers=N)`).
* Robust retry loop for transient OpenAI errors.
* Progress bar via `tqdm`.
* Column names are read from each dataset’s meta.yaml (no hard-coding).
"""
from __future__ import annotations

from pathlib import Path
import re


import pandas as pd

from .base_evaluator import BaseEvaluator

# ────────────────────────────────────────────────────────────────────────────
# Regex: **** ۱۲.0 **** → captures "۱۲.0"  (Persian or ASCII digits, dot allowed)
# ────────────────────────────────────────────────────────────────────────────
ANSWER_REGEX = re.compile(r"\*{4}\s*([\d\u06F0-\u06F9.]+)\s*\*{4}")

# translate map Persian → ASCII
PERS_TO_ASCII = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# ────────────────────────────────────────────────────────────────────────────
# Class
# ────────────────────────────────────────────────────────────────────────────
class MCQEvaluator(BaseEvaluator):
    """Evaluate a DataFrame of MCQ rows against an LLM endpoint."""

    # --------------------------------------------------------------------- #
    # Constructor                                                           #
    # --------------------------------------------------------------------- #
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
        self.choice_cols: list[str] = self.meta["choice_cols"]

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #
    def _extract(self, text: str | None) -> str | None:
        """Return the chosen option number from ``text`` if present."""
        if not text:
            return None

        m = ANSWER_REGEX.search(text)
        if not m:
            return None
        token = m.group(1).translate(PERS_TO_ASCII).strip()  # Persian → ASCII

        try:
            return str(int(float(token)))   # "1.0" → "1"
        except ValueError:
            return None

    def _query_model(self, prompt: str) -> str:
        """Single chat completion call → assistant text."""
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system",   "content": "You are an expert evaluator for multiple-choice questions (MCQs). Your task is to carefully read each question and its options, determine the most accurate answer based on logic and knowledge, and explain your reasoning briefly if needed."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.01,
            top_p=0.01,
            max_tokens = 4000,
        )
        return resp.choices[0].message.content

    def _build_prompt(self, df: pd.DataFrame, row: pd.Series) -> str:
        """Render the Jinja2 template for the current row."""
        shot_ids = (
            self.rng.sample(
                [i for i in df.index if i != row.name],
                k=min(self.shots, len(df) - 1),
            )
            if self.shots else []
        )

        shots = [
            {
                "question": df.loc[i, self.qcol],
                "choices": [
                    df.loc[i, c] for c in self.choice_cols if pd.notna(df.loc[i, c])
                ],
                "answer": str(df.loc[i, self.acol]) if self.acol else "",
                "rationale": "",
            }
            for i in shot_ids
        ]

        query = {
            "question": row[self.qcol],
            "choices": [row[c] for c in self.choice_cols if pd.notna(row[c])],
        }

        return self.template.render(shots=shots, **query)
