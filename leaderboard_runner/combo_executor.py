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
from pathlib import Path
from typing import List

from . import paths
from .cmd_utils import build_run_eval_cmd
from .io_utils import sample_csv
from .meta_utils import load_meta_fields

logger = logging.getLogger(__name__)


def run_single_combo(
    *,
    model: str,
    dataset: str,
    n_rows: int | None,
    shots: int,
    workers: int,
    dry_run: bool = False,
) -> None:
    """Evaluate *model* on *dataset* and write results into ``results/``."""
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

    # Cleanup temporary files ------------------------------------------------ #
    for t in tmp_files:
        t.unlink(missing_ok=True)
