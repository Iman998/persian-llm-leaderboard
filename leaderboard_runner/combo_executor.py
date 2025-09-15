"""
combo_executor.py
~~~~~~~~~~~~~~~~~
Run one (MODEL × DATASET) evaluation combo.  Keeps sampling and file
normalisation logic local so other modules stay clean.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

import pandas as pd
import yaml

from . import paths
from .cmd_utils import build_run_eval_cmd
from .io_utils import sample_csv
from .meta_utils import load_meta_fields

logger = logging.getLogger(__name__)


def _run_judge_evaluation(
    *,
    model: str,
    dataset: str,
    meta_file: Path,
    result_csv: Path,
    shots: int,
    workers: int,
    n_rows: int | None,
    dry_run: bool,
) -> None:
    """Run ``judge_evaluator.py`` on ``result_csv`` predictions."""

    with meta_file.open("r", encoding="utf-8") as fh:
        meta_cfg = yaml.safe_load(fh) or {}
    question_col = meta_cfg.get("question_col", "question")

    df = pd.read_csv(result_csv)
    cand_col = "candidate"
    if cand_col not in df.columns:
        df[cand_col] = df["pred"]
    else:
        df[cand_col] = df["pred"]

    if question_col not in df.columns and "Question Body" in df.columns:
        df.rename(columns={"Question Body": question_col}, inplace=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = Path(tmp.name)

    out_dir = paths.RESULTS_DIR / f"{dataset}_judge" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{n_rows}" if n_rows else ""
    judge_csv = out_dir / f"{model}{suffix}.csv"

    cmd = build_run_eval_cmd(
        model_stub=model,
        dataset_path=tmp_path,
        meta_path=meta_file,
        prompt_template="prompts/judge.jinja2",
        evaluator="evaluators/judge_evaluator.py",
        n_rows=n_rows,
        shots=shots,
        workers=workers,
        out_csv=judge_csv,
    )

    if dry_run:
        print(" ".join(map(str, cmd)))
    else:
        logger.info("RUN %s × %s → judge %s", model, dataset, judge_csv.name)
        subprocess.run([str(c) for c in cmd], check=True)

        if n_rows:
            shutil.copy2(judge_csv, out_dir / f"{model}.csv")
            for f in out_dir.glob(f"{model}{suffix}_*.csv"):
                shutil.copy2(f, out_dir / f"{model}{f.name[len(model + suffix):]}")

    tmp_path.unlink(missing_ok=True)

def run_single_combo(
    *,
    model: str,
    dataset: str,
    n_rows: int | None,
    shots: int,
    workers: int,
    judge: bool = False,
    dry_run: bool = False,
) -> None:
    """Evaluate *model* on *dataset* and write results into ``results/``.

    If ``judge`` is ``True`` and the dataset is a ``text_generation`` or
    ``summarization`` task, a second pass is executed using
    :mod:`evaluators.judge_evaluator` on the predictions from the first run.
    Judge scores are written under ``results/<dataset>_judge/<model>/``.
    """
    csv_file = paths.DATASETS_DIR / dataset / "test.csv"
    meta_file = paths.DATASETS_DIR / dataset / "meta.yaml"
    model_yaml = paths.MODELS_DIR / f"{model}.yaml"

    if not csv_file.exists():
        logger.warning("Dataset not found, skipping: %s", csv_file)
        return
    if not model_yaml.exists():
        logger.warning("Model config missing, skipping: %s", model_yaml)
        return

    suffix = f"_{n_rows}" if n_rows else ""
    tmp_files: List[Path] = []

    # Sample CSV if requested ------------------------------------------------ #
    dataset_path: Path
    if n_rows:
        dataset_path = sample_csv(csv_file, n_rows)
        tmp_files.append(dataset_path)
    else:
        dataset_path = csv_file

    # Output path ------------------------------------------------------------ #
    out_dir = paths.RESULTS_DIR / dataset / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{model}{suffix}.csv"

    # Meta fields ------------------------------------------------------------ #
    prompt_template, evaluator = load_meta_fields(meta_file)

    # Build & run command ---------------------------------------------------- #
    cmd = build_run_eval_cmd(
        model_stub=model,
        dataset_path=dataset_path,
        meta_path=meta_file,
        prompt_template=prompt_template,
        evaluator=evaluator,
        n_rows=n_rows,
        shots=shots,
        workers=workers,
        out_csv=out_csv,
    )

    if dry_run:
        print(" ".join(map(str, cmd)))
    else:
        logger.info("RUN %s × %s → %s", model, dataset, out_csv.name)
        subprocess.run([str(c) for c in cmd], check=True)

        # If sampling, also copy canonical filename
        if n_rows:
            shutil.copy2(out_csv, out_dir / f"{model}.csv")
            for f in out_dir.glob(f"{model}{suffix}_*.csv"):
                shutil.copy2(f, out_dir / f"{model}{f.name[len(model + suffix):]}")

        # Optional LLM-judge evaluation ------------------------------------- #
        with meta_file.open("r", encoding="utf-8") as fh:
            meta_cfg = yaml.safe_load(fh)
            meta_cfg.get("judge", False)
        if (
            judge
            and meta_cfg.get("task") in {"text_generation", "summarization"}
            and meta_cfg.get("judge", False)
        ):
            _run_judge_evaluation(
                model=model,
                dataset=dataset,
                meta_file=meta_file,
                result_csv=out_csv,
                shots=shots,
                workers=workers,
                n_rows=n_rows,
                dry_run=dry_run,
            )

    # Cleanup temporary files ------------------------------------------------ #
    for t in tmp_files:
        t.unlink(missing_ok=True)
