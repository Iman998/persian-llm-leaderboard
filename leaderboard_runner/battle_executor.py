"""Run pairwise battles from existing model result CSVs."""

from __future__ import annotations

import logging
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import pandas as pd
import yaml

from . import paths
from .cmd_utils import build_run_eval_cmd


logger = logging.getLogger(__name__)


def _result_path(dataset: str, model: str, n_rows: int | None) -> Path:
    suffix = f"_{n_rows}" if n_rows else ""
    return paths.RESULTS_DIR / dataset / model / f"{model}{suffix}.csv"


def _restore_question_column(df: pd.DataFrame, question_col: str) -> pd.DataFrame:
    restored = df.copy()
    if question_col not in restored and "Question Body" in restored:
        restored.rename(columns={"Question Body": question_col}, inplace=True)
    return restored


def _align_results(
    model_1_df: pd.DataFrame,
    model_2_df: pd.DataFrame,
    *,
    meta_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two result frames using stable dataset identity columns."""
    question_col = meta_cfg.get("question_col", "question")
    answer_col = meta_cfg.get("answer_col", "Key")
    model_1_df = _restore_question_column(model_1_df, question_col)
    model_2_df = _restore_question_column(model_2_df, question_col)

    identity_cols = [
        column
        for column in (
            question_col,
            answer_col,
            meta_cfg.get("source_language_col"),
            meta_cfg.get("target_language_col"),
        )
        if column
        and column in model_1_df.columns
        and column in model_2_df.columns
    ]
    if question_col not in identity_cols:
        raise ValueError(f"Battle results are missing question column '{question_col}'")
    if len(model_1_df) != len(model_2_df):
        raise ValueError("Battle result files contain different row counts")

    left_keys = model_1_df[identity_cols].astype(str)
    right_keys = model_2_df[identity_cols].astype(str)
    if left_keys.equals(right_keys):
        return model_1_df.reset_index(drop=True), model_2_df.reset_index(drop=True)

    if left_keys.duplicated().any() or right_keys.duplicated().any():
        raise ValueError(
            "Battle result rows differ and cannot be aligned because identity "
            "columns are not unique"
        )

    left_index = pd.MultiIndex.from_frame(left_keys)
    right_index = pd.MultiIndex.from_frame(right_keys)
    if set(left_index) != set(right_index):
        raise ValueError("Battle result files do not contain the same dataset rows")
    order = right_index.get_indexer(left_index)
    aligned_right = model_2_df.iloc[order].reset_index(drop=True)
    return model_1_df.reset_index(drop=True), aligned_right


def _battle_config(
    meta_cfg: dict[str, Any],
    *,
    judge_model: str | None,
) -> dict[str, Any] | None:
    raw = meta_cfg.get("battle", True)
    if isinstance(raw, dict):
        config = raw.copy()
        enabled = bool(config.pop("enabled", True))
    else:
        config = {}
        enabled = bool(raw)
    if not enabled:
        return None
    return {
        "model": judge_model or config.get("model"),
        "prompt_template": config.get(
            "prompt_template", "prompts/battle.jinja2"
        ),
        "evaluator": config.get(
            "evaluator", "evaluators/battle_evaluator.py"
        ),
        "use_reference": bool(config.get("use_reference", True)),
    }


def load_battle_inputs(
    *,
    dataset: str,
    model_1: str,
    model_2: str,
    n_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return aligned existing outputs and dataset metadata for a matchup."""
    meta_path = paths.DATASETS_DIR / dataset / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset metadata missing: {meta_path}")
    meta_cfg = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}

    model_1_path = _result_path(dataset, model_1, n_rows)
    model_2_path = _result_path(dataset, model_2, n_rows)
    for result_path in (model_1_path, model_2_path):
        if not result_path.exists():
            raise FileNotFoundError(f"Battle candidate result missing: {result_path}")

    model_1_df, model_2_df = _align_results(
        pd.read_csv(model_1_path),
        pd.read_csv(model_2_path),
        meta_cfg=meta_cfg,
    )
    battle_df = model_1_df.copy()
    battle_df["model_1_output"] = model_1_df["pred"]
    battle_df["model_2_output"] = model_2_df["pred"]
    battle_df["model_1"] = model_1
    battle_df["model_2"] = model_2
    return battle_df, meta_cfg


def build_battle_meta(
    meta_cfg: dict[str, Any],
    *,
    model_1: str,
    model_2: str,
    use_reference: bool,
) -> dict[str, Any]:
    """Return temporary evaluator metadata for one pairwise match."""
    battle_meta = meta_cfg.copy()
    battle_meta.update(
        {
            "answer_col": meta_cfg.get("answer_col", "Key"),
            "model_1_col": "model_1_output",
            "model_2_col": "model_2_output",
            "model_1_name": model_1,
            "model_2_name": model_2,
            "use_reference": use_reference,
            "category_cols": [],
            "metrics": [],
            "battle": False,
        }
    )
    return battle_meta


def run_battle(
    *,
    dataset: str,
    model_1: str,
    model_2: str,
    judge_model: str | None,
    n_rows: int | None,
    workers: int,
    dry_run: bool = False,
) -> None:
    """Judge two models using their existing result CSVs."""
    if model_1 == model_2:
        raise ValueError("Battle models must be different")

    meta_path = paths.DATASETS_DIR / dataset / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset metadata missing: {meta_path}")
    meta_cfg = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    config = _battle_config(meta_cfg, judge_model=judge_model)
    if config is None:
        logger.info("SKIP battle for %s: disabled in metadata", dataset)
        return
    if not config["model"]:
        raise ValueError(
            "Battle judge model is not configured; set --battle-judge-model "
            "or meta.yaml::battle.model"
        )

    judge_model_path = paths.MODELS_DIR / f"{config['model']}.yaml"
    if not judge_model_path.exists():
        raise FileNotFoundError(f"Battle judge model missing: {judge_model_path}")

    battle_df, _ = load_battle_inputs(
        dataset=dataset,
        model_1=model_1,
        model_2=model_2,
        n_rows=n_rows,
    )
    battle_df["judge_model"] = config["model"]

    battle_meta = build_battle_meta(
        meta_cfg,
        model_1=model_1,
        model_2=model_2,
        use_reference=config["use_reference"],
    )

    safe_pair = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{model_1}__vs__{model_2}")
    out_dir = paths.RESULTS_DIR / "battle" / dataset / safe_pair
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "battle.csv"

    with tempfile.TemporaryDirectory(prefix="llm-battle-") as tmp_dir:
        input_path = Path(tmp_dir) / "battle_input.csv"
        battle_meta_path = Path(tmp_dir) / "meta.yaml"
        battle_df.to_csv(input_path, index=False)
        battle_meta_path.write_text(
            yaml.safe_dump(battle_meta, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        cmd = build_run_eval_cmd(
            model_stub=config["model"],
            dataset_path=input_path,
            meta_path=battle_meta_path,
            prompt_template=config["prompt_template"],
            evaluator=config["evaluator"],
            n_rows=None,
            shots=0,
            workers=workers,
            out_csv=out_csv,
        )
        if dry_run:
            print(" ".join(map(str, cmd)))
            return
        logger.info(
            "RUN battle %s vs %s on %s with %s",
            model_1,
            model_2,
            dataset,
            config["model"],
        )
        subprocess.run([str(part) for part in cmd], check=True)
