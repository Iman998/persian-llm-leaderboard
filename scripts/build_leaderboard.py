#!/usr/bin/env python3
"""build_leaderboard.py
────────────────────────────────────────────────────────
Scan ``results/*.csv`` produced by *run_eval.py*, look up each dataset’s
``meta.yaml`` to find its *answer column*, compute the appropriate metric, and
emit a single leaderboard CSV.

Key features
~~~~~~~~~~~~
1. **Dynamic gold‑answer column** – the script reads ``answer_col`` from every
   dataset’s ``meta.yaml`` (fallback ``"Key"``).
2. **Metric plug‑ins** – every metric function receives ``answer_col`` so no
   hard‑coding.
3. **Excludes per‑category breakdown files** – any CSV whose *model stub*
   still contains an underscore (e.g. ``gemma-3-27b-it_Category.csv``) is
   skipped. The main CSV is always ``<dataset>_<model>.csv``.
4. **English‑only comments**.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
import yaml

# Desired column order in the final leaderboard (extend as needed)
COL_ORDER: List[str] = [
    "Model Type", "Model", "Average",
    "MMLU (Accuracy)", "bbh (Accuracy)",
    "khayyam_challenge (Accuracy)", "parsinlu_mc (Accuracy)",
    "parsinlu_nli (Accuracy)", "parsinlu_qqp (Accuracy)",
    "persian_ARC (Accuracy)", "pquad (f1)",
]

_PERS_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

def _norm(x) -> str:  # noqa: ANN001
    """Normalise answer strings for robust comparison."""
    if pd.isna(x):
        return ""
    s = str(x).strip().translate(_PERS_DIGITS)
    try:
        return str(int(float(s)))
    except ValueError:
        return s

# ────────────────────────────── metric helpers ────────────────────────────── #

def accuracy(df: pd.DataFrame, answer_col: str) -> float:
    """Exact‑match accuracy (0‑100)."""
    return (
        df["pred"].map(_norm) == df[answer_col].map(_norm)
    ).mean() * 100.0


def dummy(_: pd.DataFrame, __: str | None = None) -> float:  # noqa: D401,E501
    """Placeholder metric for generation / short‑answer datasets."""
    return 0.0

_METRIC_SUFFIX_MAP: Dict[str, Tuple[Callable, str]] = {
    "generation": (dummy, "Dummy"),
    "shortanswer": (dummy, "Basic_containment"),
}

def pick_metric(ds_name: str) -> Tuple[Callable, str]:
    for suffix, (fn, label) in _METRIC_SUFFIX_MAP.items():
        if ds_name.endswith(suffix):
            return fn, label
    return accuracy, "Accuracy"

# ─────────────────────────── leaderboard builder ──────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results",
                        help="Directory containing <dataset>_<model>.csv files.")
    parser.add_argument("--out", required=True,
                        help="Output path for the leaderboard CSV.")
    parser.add_argument("--datasets_dir", default="datasets",
                        help="Root folder where each dataset has a meta.yaml.")
    parser.add_argument("--models_dir", default="models",
                        help="Directory containing <model>.yaml files.")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []

    for csv_path in Path(args.results_dir).glob("*.csv"):
        # Main results follow <dataset>_<model>.csv.  Any extra underscore in
        # the *model stub* means it is a per‑category breakdown → skip.
        stem_parts = csv_path.stem.split("_", 1)
        if len(stem_parts) != 2:
            continue
        dataset, model_stub = stem_parts
        if "_" in model_stub or dataset.endswith(("raw", "Level")):
            continue

        # Load model metadata --------------------------------------------------
        model_yaml = Path(args.models_dir) / f"{model_stub}.yaml"
        if not model_yaml.exists():
            continue  # Skip unknown model
        model_cfg = yaml.safe_load(model_yaml.read_text())
        model_type = model_cfg.get("type", "Instruct")
        model_name = model_cfg.get("display_name", model_stub)

        # Load evaluation CSV --------------------------------------------------
        df = pd.read_csv(csv_path)

        # Locate meta.yaml to get answer_col -----------------------------------
        meta_file = Path(args.datasets_dir) / dataset / "meta.yaml"
        answer_col = "Key"  # fallback
        if meta_file.exists():
            meta_cfg = yaml.safe_load(meta_file.read_text())
            answer_col = meta_cfg.get("answer_col", "Key")
        if answer_col not in df.columns:
            continue  # skip if answer column absent

        # Metric selection -----------------------------------------------------
        metric_fn, metric_label = pick_metric(dataset)
        value = metric_fn(df, answer_col) if metric_fn is not dummy else metric_fn(df)

        rows.append({
            "dataset": dataset,
            "model": model_name,
            "model_type": model_type,
            "metric_label": metric_label,
            "value": value,
        })

    if not rows:
        print("No result CSVs found; nothing to build.")
        return

    # Long → wide pivot --------------------------------------------------------
    long = pd.DataFrame(rows)
    wide = (
        long.pivot(index=["model_type", "model"],
                    columns="dataset", values="value")
            .reset_index()
    )
    wide.columns.name = None

    # Rename dataset columns with metric label --------------------------------
    rename_map = {
        ds: f"{ds} ({lbl})"
        for ds, lbl in long.drop_duplicates("dataset")[["dataset", "metric_label"]]
        .itertuples(index=False)
    }
    rename_map.update({"model_type": "Model Type", "model": "Model"})
    wide = wide.rename(columns=rename_map)

    # Ensure expected columns exist -------------------------------------------
    for col in COL_ORDER:
        if col not in wide.columns:
            wide[col] = ""

    # Compute Average over accuracy columns -----------------------------------
    acc_cols = [c for c in wide.columns if c.endswith("(Accuracy)")]
    acc_vals = wide[acc_cols].apply(pd.to_numeric, errors="coerce")
    counts = acc_vals.notna().sum(axis=1)
    wide["Average"] = (
        (acc_vals.sum(axis=1) / counts).round(5).where(counts > 0, "")
    )

    # Re‑order columns ---------------------------------------------------------
    wide = wide[[c for c in COL_ORDER if c in wide.columns]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path, index=False)
    print(f"🏆  Leaderboard written → {out_path}")


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        sys.exit("Python ≥ 3.9 required")
    main()