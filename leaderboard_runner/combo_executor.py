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
from typing import Any, List

import pandas as pd
import yaml

from . import paths
from .cmd_utils import build_run_eval_cmd
from .io_utils import sample_csv
from .meta_utils import load_judge_configs, load_meta_fields

logger = logging.getLogger(__name__)


def _run_judge_evaluation(
    *,
    candidate_model: str,
    judge_config: dict[str, Any],
    dataset: str,
    meta_cfg: dict[str, Any],
    result_csv: Path,
    shots: int,
    workers: int,
    n_rows: int | None,
    dry_run: bool,
) -> None:
    """Run ``judge_evaluator.py`` on ``result_csv`` predictions."""
    judge_model_file = paths.MODELS_DIR / f"{judge_config['model']}.yaml"
    if not judge_model_file.exists():
        raise FileNotFoundError(f"Judge model config missing: {judge_model_file}")

    question_col = meta_cfg.get("question_col", "question")

    df = pd.read_csv(result_csv)
    cand_col = "candidate"
    if cand_col not in df.columns:
        df[cand_col] = df["pred"]
    else:
        df[cand_col] = df["pred"]

    if question_col not in df.columns and "Question Body" in df.columns:
        df.rename(columns={"Question Body": question_col}, inplace=True)

    judge_meta = meta_cfg.copy()
    judge_meta.update(
        {
            "evaluator": judge_config["evaluator"],
            "prompt_template": judge_config["prompt_template"],
            "candidate_col": cand_col,
            "use_reference": judge_config["use_reference"],
            "metrics": judge_config["metrics"],
        }
    )
    if judge_config.get("score_min") is not None:
        judge_meta["judge_score_min"] = judge_config["score_min"]
    if judge_config.get("score_max") is not None:
        judge_meta["judge_score_max"] = judge_config["score_max"]
    judge_meta["judge"] = False

    with tempfile.TemporaryDirectory(prefix="llm-judge-") as tmp_dir:
        tmp_path = Path(tmp_dir) / "candidates.csv"
        tmp_meta = Path(tmp_dir) / "meta.yaml"
        df.to_csv(tmp_path, index=False)
        tmp_meta.write_text(
            yaml.safe_dump(judge_meta, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        mode_slug = judge_config["mode"].replace("-", "_")
        out_dir = (
            paths.RESULTS_DIR
            / f"{dataset}_judge_{mode_slug}"
            / candidate_model
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{n_rows}" if n_rows else ""
        judge_csv = out_dir / f"{candidate_model}{suffix}.csv"

        cmd = build_run_eval_cmd(
            model_stub=judge_config["model"],
            dataset_path=tmp_path,
            meta_path=tmp_meta,
            prompt_template=judge_config["prompt_template"],
            evaluator=judge_config["evaluator"],
            # The candidate result is already sampled by the first pass.
            n_rows=None,
            shots=shots,
            workers=workers,
            out_csv=judge_csv,
        )

        if dry_run:
            print(" ".join(map(str, cmd)))
        else:
            logger.info(
                "RUN judge %s on %s × %s → %s",
                judge_config["model"],
                candidate_model,
                dataset,
                judge_csv.name,
            )
            subprocess.run([str(c) for c in cmd], check=True)

            if n_rows:
                shutil.copy2(judge_csv, out_dir / f"{candidate_model}.csv")
                for f in out_dir.glob(f"{candidate_model}{suffix}_*.csv"):
                    shutil.copy2(
                        f,
                        out_dir
                        / f"{candidate_model}{f.name[len(candidate_model + suffix):]}",
                    )


def run_single_combo(
    *,
    model: str,
    dataset: str,
    n_rows: int | None,
    shots: int,
    workers: int,
    judge: bool = False,
    judge_model: str | None = None,
    judge_mode: str = "reference",
    judge_only: bool = False,
    dry_run: bool = False,
) -> None:
    """Evaluate *model* on *dataset* and write results into ``results/``.

    ``judge_only`` reuses an existing candidate result and never invokes the
    candidate model. Judge modes are stored in separate result directories.
    """
    csv_file = paths.DATASETS_DIR / dataset / "test.csv"
    meta_file = paths.DATASETS_DIR / dataset / "meta.yaml"
    model_yaml = paths.MODELS_DIR / f"{model}.yaml"

    if not meta_file.exists():
        logger.warning(
            "Dataset not found (missing metadata), skipping: %s",
            meta_file,
        )
        return
    if not model_yaml.exists():
        logger.warning("Model config missing, skipping: %s", model_yaml)
        return

    suffix = f"_{n_rows}" if n_rows else ""
    out_dir = paths.RESULTS_DIR / dataset / model
    out_csv = out_dir / f"{model}{suffix}.csv"

    with meta_file.open("r", encoding="utf-8") as fh:
        meta_cfg = yaml.safe_load(fh) or {}
    judge_configs = load_judge_configs(
        meta_cfg,
        candidate_model=model,
        judge_model_override=judge_model,
        judge_mode=judge_mode,
    )
    judge_supported = meta_cfg.get("task") in {
        "text_generation",
        "summarization",
        "translation",
    }

    if judge_only:
        if not judge or not judge_supported or not judge_configs:
            logger.info("SKIP %s × %s: judge is not configured", model, dataset)
            return
        if not out_csv.exists():
            raise FileNotFoundError(
                f"Candidate result missing for judge-only mode: {out_csv}"
            )
        for judge_config in judge_configs:
            _run_judge_evaluation(
                candidate_model=model,
                judge_config=judge_config,
                dataset=dataset,
                meta_cfg=meta_cfg,
                result_csv=out_csv,
                shots=shots,
                workers=workers,
                n_rows=n_rows,
                dry_run=dry_run,
            )
        return

    if not csv_file.exists():
        logger.warning("Dataset not found, skipping: %s", csv_file)
        return

    tmp_files: List[Path] = []

    # Sample CSV if requested ------------------------------------------------ #
    dataset_path: Path
    if n_rows:
        dataset_path = sample_csv(csv_file, n_rows)
        tmp_files.append(dataset_path)
    else:
        dataset_path = csv_file

    # Output path ------------------------------------------------------------ #
    out_dir.mkdir(parents=True, exist_ok=True)

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
        if judge and judge_supported:
            for judge_config in judge_configs:
                _run_judge_evaluation(
                    candidate_model=model,
                    judge_config=judge_config,
                    dataset=dataset,
                    meta_cfg=meta_cfg,
                    result_csv=out_csv,
                    shots=shots,
                    workers=workers,
                    n_rows=n_rows,
                    dry_run=dry_run,
                )

    # Cleanup temporary files ------------------------------------------------ #
    for t in tmp_files:
        t.unlink(missing_ok=True)
