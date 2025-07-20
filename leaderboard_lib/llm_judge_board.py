#!/usr/bin/env python3
"""Build leaderboard CSV focusing on summarization and translation tasks."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd
import yaml

from .data_utils import _norm
from .utils import _load_module

# Desired column order for LLM-as-Judge leaderboard
COL_ORDER: List[str] = [
    "Model Type",
    "Model",
    "Average",
    "summarization_quality (Score)",
    "translation_quality (Score)",
]


# ────────────────────────────── metric helpers ────────────────────────────── #

def accuracy(df: pd.DataFrame, answer_col: str) -> float:
    """Exact-match accuracy (0-100)."""
    return (df["pred"].map(_norm) == df[answer_col].map(_norm)).mean() * 100.0


def load_metric(name: str) -> Callable[[pd.Series, pd.Series], float]:
    """Load ``compute`` from ``metrics/<name>.py``."""

    path = Path("metrics") / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Metric '{name}' not found: expected {path}")

    mod = _load_module(str(path))
    if not hasattr(mod, "compute"):
        raise AttributeError(f"Metric module {path} is missing a 'compute' function")
    return getattr(mod, "compute")


# ─────────────────────────── leaderboard builder ──────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Directory containing results/<dataset>/<model>/<model>.csv files.",
    )
    parser.add_argument(
        "--out", required=True, help="Output path for the leaderboard CSV."
    )
    parser.add_argument(
        "--datasets_dir", default="data", help="Root folder where each dataset has a meta.yaml."
    )
    parser.add_argument(
        "--models_dir", default="models", help="Directory containing <model>.yaml files."
    )
    args = parser.parse_args()

    model_names = sorted(
        [p.stem for p in Path(args.models_dir).glob("*.yaml")], key=len, reverse=True
    )
    models_alt = "|".join(map(re.escape, model_names))
    file_re = re.compile(rf"^(?P<model>{models_alt})(?:_(?P<suffix>.+?))?\.csv$")

    rows: List[Dict[str, Any]] = []

    def _pretty_metric(name: str) -> str:
        mapping = {
            "accuracy": "Accuracy",
            "exact_match": "Exact Match",
            "f1": "F1",
            "bleu": "BLEU",
            "rouge": "ROUGE-L",
            "llm_judge_score": "Score",
        }
        return mapping.get(name, name)

    for csv_path in Path(args.results_dir).rglob("*.csv"):
        m = file_re.match(csv_path.name)
        if not m:
            continue
        try:
            dataset = csv_path.relative_to(args.results_dir).parts[0]
        except ValueError:
            dataset = csv_path.parent.parent.name
        model_stub, suffix = m.group("model", "suffix")
        if suffix or dataset.endswith(("raw", "Level")):
            continue

        model_yaml = Path(args.models_dir) / f"{model_stub}.yaml"
        if not model_yaml.exists():
            continue
        model_cfg = yaml.safe_load(model_yaml.read_text())
        model_type = model_cfg.get("type", "Instruct")
        model_name = model_cfg.get("display_name", model_stub)

        df = pd.read_csv(csv_path)

        meta_dataset = dataset[:-6] if dataset.endswith("_judge") else dataset
        meta_file = Path(args.datasets_dir) / meta_dataset / "meta.yaml"
        meta_cfg = {}
        if meta_file.exists():
            meta_cfg = yaml.safe_load(meta_file.read_text())
        else:
            print(f"Warning: {meta_file} not found; using defaults.")

        answer_col = meta_cfg.get("answer_col", "Key")
        if answer_col not in df.columns:
            continue

        metrics = meta_cfg.get("metrics", ["accuracy"])

        for metric_name in metrics:
            metric_fn = load_metric(metric_name)
            value = metric_fn(df["pred"], df[answer_col])
            rows.append(
                {
                    "dataset": meta_dataset,
                    "metric_label": _pretty_metric(metric_name),
                    "model": model_name,
                    "model_type": model_type,
                    "value": value,
                }
            )

    if not rows:
        print("No result CSVs found; nothing to build.")
        return

    long = pd.DataFrame(rows)
    wide = (
        long.pivot_table(
            index=["model_type", "model"],
            columns=["dataset", "metric_label"],
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    wide.columns.name = None

    rename_map = {
        (ds, lbl): f"{ds} ({lbl})"
        for ds, lbl in long[["dataset", "metric_label"]].drop_duplicates().itertuples(index=False)
    }
    rename_map.update({("model_type", ""): "Model Type", ("model", ""): "Model"})

    new_cols = []
    for c in wide.columns:
        if isinstance(c, tuple):
            new_cols.append(
                rename_map.get(c, c[0] if c[1] == "" else f"{c[0]} ({c[1]})")
            )
        else:
            new_cols.append(rename_map.get((c, ""), c))
    wide.columns = new_cols

    for col in COL_ORDER:
        if col not in wide.columns:
            wide[col] = ""

    num_cols = [c for c in wide.columns if c not in {"Model Type", "Model"}]
    num_vals = wide[num_cols].apply(pd.to_numeric, errors="coerce")
    counts = num_vals.notna().sum(axis=1)
    wide["Average"] = (num_vals.sum(axis=1) / counts).round(5).where(counts > 0, "")

    extra_cols = [c for c in wide.columns if c not in COL_ORDER]
    wide = wide[[c for c in COL_ORDER if c in wide.columns] + extra_cols]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path, index=False)
    print(f"🏆  Leaderboard written → {out_path}")

if __name__ == "__main__":
    main()
