"""Base evaluator providing OpenAI request logic and prompting support."""

from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import random
from typing import List

import pandas as pd
import yaml
from jinja2 import Environment
from tqdm import tqdm
from openai import OpenAI, APIConnectionError, APITimeoutError, BadRequestError

from leaderboard_lib.utils import load_api_key


class BaseEvaluator:
    """Base class for evaluating datasets with an LLM backend."""

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
        # OpenAI client ------------------------------------------------------
        api_key = model_cfg.get("api_key") or load_api_key(model_cfg.get("model"))
        if not api_key:
            raise RuntimeError(
                "OpenAI API key missing. Set OPENAI_API_KEY or use secrets.toml"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=model_cfg["base_url"],
            timeout=100_000,
        )
        self.model_name: str = model_cfg["model"]

        # Jinja2 template ----------------------------------------------------
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["enumerate"] = enumerate
        self.template = env.from_string(Path(prompt_path).read_text(encoding="utf-8"))

        # Dataset meta -------------------------------------------------------
        self.meta = yaml.safe_load(Path(meta_path).read_text())
        self.qcol: str = self.meta.get("question_col", "question")
        self.acol: str | None = self.meta.get("answer_col")

        # Few-shot settings --------------------------------------------------
        self.shots = shots
        self.rng = random.Random(shot_seed)

        # Retry policy -------------------------------------------------------
        self.max_retries = max_retries

        # Logger -------------------------------------------------------------
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_df(self, df: pd.DataFrame, *, max_workers: int = 4) -> pd.DataFrame:
        """Return a copy of *df* with `pred` and `raw` columns added."""
        preds: List[str | None] = [None] * len(df)
        raws: List[str] = [""] * len(df)

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
                pool.submit(_worker, idx, row): idx
                for idx, (_, row) in enumerate(df.iterrows())
            }
            for fut in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc="LLM requests",
                leave=False,
            ):
                i, pred, raw = fut.result()
                preds[i] = pred
                raws[i] = raw

        out = df.copy()
        out["pred"] = preds
        out["raw"] = raws
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _query_model(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.01,
            top_p=0.01,
            max_tokens = 4000,

        )
        return resp.choices[0].message.content

    def _build_prompt(self, df: pd.DataFrame, row: pd.Series) -> str:  # noqa: D401
        """Return the prompt for *row* (implemented by subclasses)."""
        raise NotImplementedError

    def _extract(self, text: str) -> str | None:  # noqa: D401
        """Parse model output into the final prediction."""
        return text.strip()
