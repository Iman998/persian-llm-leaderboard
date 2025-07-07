"""
cmd_utils.py
~~~~~~~~~~~~
Builds shell command lists that call other project scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from .paths import MODELS_DIR, RUN_EVAL_SCRIPT


def build_run_eval_cmd(
    *,
    model_stub: str,
    dataset_path: Path,
    meta_path: Path,
    prompt_template: str,
    evaluator: str,
    shots: int,
    workers: int,
    n_rows: int | None,
    out_csv: Path,
) -> List[str | Path]:
    """
    Return a fully-populated argv list for invoking ``scripts/run_eval.py``.
    """
    return [
        sys.executable,
        RUN_EVAL_SCRIPT,
        "--dataset",
        dataset_path,
        "--meta",
        meta_path,
        "--model",
        MODELS_DIR / f"{model_stub}.yaml",
        "--prompt",
        prompt_template,
        "--evaluator",
        evaluator,
        "--shots",
        str(shots),
        "--workers",
        str(workers),
        "--out",
        out_csv,
       # Forward sampling so evaluation can replicate the canonical filename
        # for leaderboard consumption.
        *([] if n_rows is None else ["--n_rows", str(n_rows)]),
    ]
