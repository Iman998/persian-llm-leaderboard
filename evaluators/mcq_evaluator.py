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
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import random
import re
from typing import List

import pandas as pd
import yaml
from jinja2 import Environment
from tqdm import tqdm
from openai import OpenAI, APIConnectionError, APITimeoutError, BadRequestError

# ────────────────────────────────────────────────────────────────────────────
# Regex: **** ۱۲.0 **** → captures "۱۲.0"  (Persian or ASCII digits, dot allowed)
# ────────────────────────────────────────────────────────────────────────────
ANSWER_REGEX = re.compile(r"\*{4}\s*([\d\u06F0-\u06F9.]+)\s*\*{4}")

# translate map Persian → ASCII
PERS_TO_ASCII = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# ────────────────────────────────────────────────────────────────────────────
# Class
# ────────────────────────────────────────────────────────────────────────────
class MCQEvaluator:
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
        # 1) OpenAI client -------------------------------------------------- #
        self.client = OpenAI(
            api_key=model_cfg["api_key"],
            base_url=model_cfg["base_url"],
            timeout=100_000,
        )
        self.model_name: str = model_cfg["model"]

        # 2) Jinja2 template (inject enumerate)
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["enumerate"] = enumerate
        self.template = env.from_string(prompt_path.read_text(encoding="utf-8"))

        # 3) Dataset meta-information -------------------------------------- #
        meta = yaml.safe_load(meta_path.read_text())
        self.qcol: str = meta["question_col"]
        self.choice_cols: list[str] = meta["choice_cols"]
        self.acol: str | None = meta.get("answer_col")

        # 4) Few-shot settings --------------------------------------------- #
        self.shots = shots
        self.rng = random.Random(shot_seed)

        # 5) Retry policy --------------------------------------------------- #
        self.max_retries = max_retries

        # 6) Logger --------------------------------------------------------- #
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(message)s",
            )

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def evaluate_df(
        self,
        df: pd.DataFrame,
        *,
        max_workers: int = 4,
    ) -> pd.DataFrame:
        """Return a copy of *df* with `pred` and `raw` columns added."""
        preds: List[str | None] = [None] * len(df)
        raws:  List[str]        = [""]  * len(df)

        def _worker(idx: int, row: pd.Series):
            prompt = self._build_prompt(df, row)
            for attempt in range(1, self.max_retries + 1):
                try:
                    text = self._query_model(prompt)
                    return idx, self._extract(text), text
                except (APIConnectionError, APITimeoutError, BadRequestError) as e:
                    self.logger.warning(
                        "row %d retry %d/%d: %s", idx, attempt, self.max_retries, e
                    )
            return idx, None, ""

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(_worker, i, r): i for i, r in df.iterrows()
            }
            for fut in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc="LLM requests",
                leave=False,
            ):
                i, pred, raw = fut.result()
                preds[i] = pred
                raws[i]  = raw

        out = df.copy()
        out["pred"] = preds
        out["raw"]  = raws
        return out

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #
    def _extract(self, text: str) -> str | None:
        """
        Find ****n****, tolerate Persian digits & decimals.
        Return normalized index as ASCII string ("1","2",…)
        or None if parsing fails.
        """
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
                {"role": "system",   "content": "You are an expert evaluator for Persian multiple-choice questions (MCQs). Your task is to carefully read each question and its options, determine the most accurate answer based on logic and knowledge, and explain your reasoning briefly in Persian if needed."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.01,
            top_p=0.01,
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
