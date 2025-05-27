#!/usr/bin/env python3
"""
build_leaderboard.py
────────────────────
Scan ./results/*.csv and generate ./dashboard/leaderboard.csv.

Rules
-----
• Missing accuracy columns are ignored; they do NOT count as zero.
• If a model has no Accuracy values at all, its Average is left blank.
• Datasets ending with `_generation` or `_shortanswer` use Dummy metrics.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import re
import pandas as pd
import yaml

# Desired column order
COL_ORDER = [
    "Model Type", "Model", "Average",
    "MMLU (Accuracy)","bbh (Accuracy)",
    "khayyam_challenge (Accuracy)", "parsinlu_mc (Accuracy)",
    "parsinlu_nli (Accuracy)", "parsinlu_qqp (Accuracy)",
    "persian_ARC (Accuracy)", 
    "pquad (f1)",
]

# ───────────────────────────── metric helpers ─────────────────────────── #
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

def _norm(x) -> str:
    """'۱.0 ' → '1'; NaN → ''."""
    if pd.isna(x):
        return ""
    s = str(x).strip().translate(PERSIAN_DIGITS)
    try:
        return str(int(float(s)))
    except ValueError:
        return s

def accuracy(df: pd.DataFrame) -> float:
    return (df["pred"].map(_norm) == df["Key"].map(_norm)).mean() * 100.0

def dummy(_: pd.DataFrame) -> float:
    return 0.0

METRIC_MAP = {"generation": dummy, "shortanswer": dummy}

def pick_metric(ds_name: str):
    for suff, fn in METRIC_MAP.items():
        if ds_name.endswith(suff):
            label = "Dummy" if suff == "generation" else "Basic_containment"
            return fn, label
    return accuracy, "Accuracy"

# ───────────────────────────── dashboard build ───────────────────────── #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for csv_path in Path(args.results_dir).glob("*.csv"):
        m = re.match(r"(.*)_(.*)\.csv", csv_path.name)
        if not m:
            continue
        dataset, model_stub = m.group(1), m.group(2)
        # Skip auxiliary *_Level.csv, *_raw.csv, etc.
        if dataset.endswith(("raw", "Level")) or "_" in model_stub:
            continue

        model_yaml = Path(f"models/{model_stub}.yaml")
        if not model_yaml.exists():
            continue

        model_cfg  = yaml.safe_load(model_yaml.read_text())
        model_type = model_cfg.get("type", "Instruct")
        model_name = model_cfg.get("display_name", model_stub)

        df = pd.read_csv(csv_path)
        metric_fn, _ = pick_metric(dataset)
        rows.append(
            dict(
                dataset=dataset,
                model=model_name,
                model_type=model_type,
                value=metric_fn(df),
            )
        )

    if not rows:
        print("No result CSVs found; nothing to build.")
        return

    # Long → wide pivot
    long = pd.DataFrame(rows)
    wide = (
        long.pivot(index=["model_type", "model"], columns="dataset", values="value")
            .reset_index()
    )
    wide.columns.name = None

    # Rename dataset columns with suffix
    rename_map = {
        ds: f"{ds} ({'Dummy' if ds.endswith('generation') else 'Accuracy'})"
        for ds in long["dataset"].unique()
    }
    rename_map.update({"model_type": "Model Type", "model": "Model"})
    wide = wide.rename(columns=rename_map)

    # Ensure all expected columns exist
    for col in COL_ORDER:
        if col not in wide:
            wide[col] = ""

    # Average over available Accuracy cells
    acc_cols = [c for c in COL_ORDER if c.endswith("(Accuracy)")]
    acc_vals = wide[acc_cols].apply(pd.to_numeric, errors="coerce")
    counts   = acc_vals.notna().sum(axis=1)
    wide["Average"] = (
        (acc_vals.sum(axis=1) / counts).round(5).where(counts > 0, "")
    )

    wide = wide[COL_ORDER]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path, index=False)
    print(f"🏆  Leaderboard written → {out_path}")

if __name__ == "__main__":
    main()
