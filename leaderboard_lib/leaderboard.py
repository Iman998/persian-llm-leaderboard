"""Build the leaderboard CSV from individual evaluation results.

The functions here are used by :mod:`scripts.build_leaderboard` to aggregate
per-dataset evaluation CSV files into a single leaderboard."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd
import yaml

from .data_utils import _norm
from .utils import _load_module

# Desired column order in the final leaderboard (extend as needed)
COL_ORDER: List[str] = [
    "Model Type",
    "Train Type",
    "Model",
    "Parameters",
    "Active Parameters",
    "Organization",
    "License",
    "Average",
    "Language Average",
    "mmlu (Accuracy)",
    "mmlu-pro (Accuracy)"
]


# ────────────────────────────── metric helpers ────────────────────────────── #

def accuracy(df: pd.DataFrame, answer_col: str) -> float:
    """Exact-match accuracy (0-100)."""
    return (df["pred"].map(_norm) == df[answer_col].map(_norm)).mean() * 100.0


def load_metric(name: str) -> Callable[[pd.Series, pd.Series], float]:
    """Load ``compute`` from ``metrics/<name>.py``.

    Raises a ``FileNotFoundError`` if the metric module does not exist to help
    troubleshoot missing metrics.
    """

    path = Path("metrics") / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Metric '{name}' not found: expected {path}")

    mod = _load_module(str(path))
    if not hasattr(mod, "compute"):
        raise AttributeError(f"Metric module {path} is missing a 'compute' function")
    return getattr(mod, "compute")


# ─────────────────────────── leaderboard builder ─────────────────────────── #

def main(board: str | None = None) -> None:
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
    parser.add_argument(
        "--board",
        default=board or "leaderboard",
        choices=["leaderboard", "translation", "summarization"],
        help="Dataset board type to build.",
    )
    parser.add_argument(
        "--lang",
        default="all",
        choices=["all", "fa", "en"],
        help="Filter datasets by language (meta.yaml::language)",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Only include datasets whose name contains any of these substrings",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Exclude datasets whose name contains any of these substrings",
    )
    args = parser.parse_args()

    # Language-specific boards should not include the cross-lingual average
    col_order = COL_ORDER.copy()
    if args.lang != "all" and "Language Average" in col_order:
        col_order.remove("Language Average")

    model_names = sorted(
        [p.stem for p in Path(args.models_dir).glob("*.yaml")], key=len, reverse=True
    )
    models_alt = "|".join(map(re.escape, model_names))
    file_re = re.compile(rf"^(?P<model>{models_alt})(?:_(?P<suffix>.+?))?\.csv$")

    rows: List[Dict[str, Any]] = []
    dataset_lang: Dict[str, str | None] = {}

    def _pretty_metric(name: str) -> str:
        """Return a display-friendly metric label."""
        mapping = {
            "accuracy": "Accuracy",
            "exact_match": "Exact Match",
            "f1": "F1",
            "bleu": "BLEU",
            "chrf": "chrF",
            "meteor": "METEOR",
            "ter": "1 - TER",
            "rouge1": "ROUGE-1",
            "rouge2": "ROUGE-2",
            "rougel": "ROUGE-L",
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

        ds_lower = dataset.lower()
        if args.include and not any(key.lower() in ds_lower for key in args.include):
            continue
        if args.exclude and any(key.lower() in ds_lower for key in args.exclude):
            continue

        # Load model metadata --------------------------------------------------
        model_yaml = Path(args.models_dir) / f"{model_stub}.yaml"
        if not model_yaml.exists():
            logging.warning(
                "Skipping result '%s' because model YAML not found: %s",
                csv_path,
                model_yaml,
            )
            continue  # Skip unknown model
        model_cfg = yaml.safe_load(model_yaml.read_text())
        model_type = model_cfg.get("model_type", model_cfg.get("type", "Instruct"))
        train_type = model_cfg.get("train_type", "")
        model_name = model_cfg.get("display_name", model_stub)
        parameters = model_cfg.get("parameters", "")
        active_parameters = model_cfg.get("active_parameters", "")
        organization = model_cfg.get("organization", "")
        license_name = model_cfg.get("license", "")

        # Load evaluation CSV --------------------------------------------------
        df = pd.read_csv(csv_path)

        # Locate meta.yaml to get answer_col -----------------------------------
        meta_file = Path(args.datasets_dir) / dataset / "meta.yaml"
        meta_cfg = yaml.safe_load(meta_file.read_text()) if meta_file.exists() else {}

        board = meta_cfg.get("board", "leaderboard")
        if board != args.board:
            continue

        lang = meta_cfg.get("language")
        if args.lang != "all" and lang != args.lang:
            continue
        dataset_lang[dataset] = lang

        answer_col = meta_cfg.get("answer_col", "Key")
        if answer_col not in df.columns:
            continue  # skip if answer column absent

        # Metric selection -----------------------------------------------------
        metrics = meta_cfg.get("metrics", ["accuracy"])

        for metric_name in metrics:
            metric_fn = load_metric(metric_name)
            value = metric_fn(df["pred"], df[answer_col])

            rows.append(
                {
                    "dataset": dataset,
                    "metric_label": _pretty_metric(metric_name),
                    "model": model_name,
                    "model_type": model_type,
                    "train_type": train_type,
                    "parameters": parameters,
                    "active_parameters": active_parameters,
                    "organization": organization,
                    "license": license_name,
                    "value": value,
                }
            )

    if not rows:
        print("No result CSVs found; nothing to build.")
        sys.exit(1)

    has_language = any(lang in ("en", "fa") for lang in dataset_lang.values())
    if not has_language and "Language Average" in col_order:
        col_order.remove("Language Average")

    # Long → wide pivot (average duplicates) ----------------------------------
    long = pd.DataFrame(rows)
    wide = (
        long.pivot_table(
            index=[
                "model_type",
                "train_type",
                "model",
                "parameters",
                "active_parameters",
                "organization",
                "license",
            ],
            columns=["dataset", "metric_label"],
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    wide.columns.name = None

    # Rename dataset columns with metric label --------------------------------
    rename_map = {
        (ds, lbl): f"{ds} ({lbl})"
        for ds, lbl in long[["dataset", "metric_label"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    rename_map.update({
        ("model_type", ""): "Model Type",
        ("train_type", ""): "Train Type",
        ("model", ""): "Model",
        ("parameters", ""): "Parameters",
        ("active_parameters", ""): "Active Parameters",
        ("organization", ""): "Organization",
        ("license", ""): "License",
    })

    new_cols = []
    for c in wide.columns:
        if isinstance(c, tuple):
            new_cols.append(
                rename_map.get(c, c[0] if c[1] == "" else f"{c[0]} ({c[1]})")
            )
        else:
            new_cols.append(rename_map.get((c, ""), c))
    wide.columns = new_cols

    # Ensure expected columns exist only for the main (unfiltered) leaderboard.
    # Filtered boards skip placeholder columns so that unrelated datasets remain hidden.
    if args.lang == "all" and not args.include and not args.exclude:
        for col in col_order:
            if col not in wide.columns:
                wide[col] = ""

    # Compute Average over accuracy columns -----------------------------------
    acc_cols = [c for c in wide.columns if c.endswith("(Accuracy)")]
    acc_vals = wide[acc_cols].apply(pd.to_numeric, errors="coerce")
    counts = acc_vals.notna().sum(axis=1)
    wide["Average"] = (acc_vals.sum(axis=1) / counts).round(5).where(counts > 0, "")

    if args.lang == "all" and has_language:
        en_cols = [
            rename_map[(ds, "Accuracy")]
            for ds, lang in dataset_lang.items()
            if lang == "en" and (ds, "Accuracy") in rename_map
        ]
        fa_cols = [
            rename_map[(ds, "Accuracy")]
            for ds, lang in dataset_lang.items()
            if lang == "fa" and (ds, "Accuracy") in rename_map
        ]
        en_vals = wide[en_cols].apply(pd.to_numeric, errors="coerce") if en_cols else pd.DataFrame(index=wide.index)
        fa_vals = wide[fa_cols].apply(pd.to_numeric, errors="coerce") if fa_cols else pd.DataFrame(index=wide.index)
        en_counts = en_vals.notna().sum(axis=1)
        fa_counts = fa_vals.notna().sum(axis=1)
        en_avg = en_vals.sum(axis=1) / en_counts.replace(0, pd.NA)
        fa_avg = fa_vals.sum(axis=1) / fa_counts.replace(0, pd.NA)
        en_avg_weighted = en_avg.fillna(0) * 2 / 3
        fa_avg_weighted = fa_avg.fillna(0) / 3
        lang_avg = en_avg_weighted + fa_avg_weighted
        lang_avg[(en_counts == 0) & (fa_counts == 0)] = pd.NA
        wide["Language Average"] = lang_avg.round(5).where(lang_avg.notna(), "")

    # Re-order columns ---------------------------------------------------------
    extra_cols = [c for c in wide.columns if c not in col_order]
    wide = wide[[c for c in col_order if c in wide.columns] + extra_cols]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path, index=False)
    print(f"\U0001F3C6  Leaderboard written → {out_path}")


if __name__ == "__main__":
    main()
