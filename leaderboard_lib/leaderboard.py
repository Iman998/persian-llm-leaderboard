"""Build the leaderboard CSV from individual evaluation results.

The functions here are used by :mod:`scripts.build_leaderboard` to aggregate
per-dataset evaluation CSV files into a single leaderboard.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd
import yaml

from .data_utils import _norm
from .utils import _load_module

# ────────────────────────────── metric helpers ────────────────────────────── #


def accuracy(df: pd.DataFrame, answer_col: str) -> float:
    """Exact‑match accuracy (0‑100)."""

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


# ──────────────────────────── dynamic columns ─────────────────────────────── #


def _build_column_order(wide: pd.DataFrame) -> List[str]:
    """Return the final column order derived from *wide*.

    The order rules are:
    1. Always start with the standard trio → ``Model Type``, ``Model``, ``Average``.
    2. List every *Accuracy* metric column next (alphabetically) so users can
       easily relate them to the ``Average`` column.
    3. Append any remaining metric columns alphabetically.
    """

    base_cols = ["Model Type", "Model", "Average"]

    metric_cols = [c for c in wide.columns if c not in base_cols]

    acc_cols = sorted([c for c in metric_cols if c.endswith("(Accuracy)")])
    other_cols = sorted([c for c in metric_cols if c not in acc_cols])

    return base_cols + acc_cols + other_cols


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

    # Pre‑compile regex to match any model stub found under --models_dir
    model_names = sorted(
        [p.stem for p in Path(args.models_dir).glob("*.yaml")], key=len, reverse=True
    )
    models_alt = "|".join(map(re.escape, model_names))
    file_re = re.compile(rf"^(?P<model>{models_alt})(?:_(?P<suffix>.+?))?\\.csv$")

    rows: List[Dict[str, Any]] = []

    def _pretty_metric(name: str) -> str:
        """Return a display‑friendly metric label."""

        mapping = {
            "accuracy": "Accuracy",
            "exact_match": "Exact Match",
            "f1": "F1",
            "bleu": "BLEU",
            "rouge": "ROUGE-L",
        }
        return mapping.get(name, name)

    # ───────────────────── iterate result CSVs and build "long" format
    for csv_path in Path(args.results_dir).rglob("*.csv"):
        m = file_re.match(csv_path.name)
        if not m:
            continue

        model_stub, suffix = m.group("model", "suffix")

        try:
            dataset = csv_path.relative_to(args.results_dir).parts[0]
        except ValueError:
            dataset = csv_path.parent.parent.name

        # ────────── build list of helper suffixes per‑dataset ──────────── #
        meta_file = Path(args.datasets_dir) / dataset / "meta.yaml"
        dynamic_helpers = []
        if meta_file.exists():
            meta_cfg = yaml.safe_load(meta_file.read_text())
            dynamic_helpers = [c.lower() for c in meta_cfg.get("category_cols", [])]

        helper_suffixes = ["raw", *dynamic_helpers]

        if suffix and any(suffix.lower().endswith(s) for s in helper_suffixes):
            # Skip CSVs that correspond to raw outputs or per‑category breakdowns
            continue

        # ──────────────── load model metadata
        model_yaml = Path(args.models_dir) / f"{model_stub}.yaml"
        if not model_yaml.exists():
            continue  # unknown model → ignore
        model_cfg = yaml.safe_load(model_yaml.read_text())
        model_type = model_cfg.get("type", "Instruct")
        model_name = model_cfg.get("display_name", model_stub)

        # ──────────────── load evaluation CSV
        df = pd.read_csv(csv_path)

        # ──────────────── locate dataset meta.yaml
        meta_file = Path(args.datasets_dir) / dataset / "meta.yaml"
        answer_col = "Key"  # sensible default
        if meta_file.exists():
            meta_cfg = yaml.safe_load(meta_file.read_text())
            answer_col = meta_cfg.get("answer_col", "Key")
        if answer_col not in df.columns:
            continue  # malformed CSV

        # ──────────────── metric selection
        metrics = ["accuracy"]
        if meta_file.exists():
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
                    "value": value,
                }
            )

    if not rows:
        print("No result CSVs found; nothing to build.")
        return

    # ────────────────────────── long → wide pivot
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

    # ────────────────────────── rename multi‑index columns
    rename_map = {
        (ds, lbl): f"{ds} ({lbl})"
        for ds, lbl in long[["dataset", "metric_label"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    rename_map.update({("model_type", ""): "Model Type", ("model", ""): "Model"})

    new_cols: List[str] = []
    for c in wide.columns:
        if isinstance(c, tuple):
            new_cols.append(
                rename_map.get(c, c[0] if c[1] == "" else f"{c[0]} ({c[1]})")
            )
        else:
            new_cols.append(rename_map.get((c, ""), c))
    wide.columns = new_cols

    # ────────────────────────── compute Average over Accuracy metrics
    acc_cols = [c for c in wide.columns if c.endswith("(Accuracy)")]
    acc_vals = wide[acc_cols].apply(pd.to_numeric, errors="coerce")
    counts = acc_vals.notna().sum(axis=1)
    wide["Average"] = (acc_vals.sum(axis=1) / counts).round(5).where(counts > 0, "")

    # ────────────────────────── final column order
    wide = wide[_build_column_order(wide)]

    # ────────────────────────── write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path, index=False)
    print(f"🏆  Leaderboard written → {out_path}")


if __name__ == "__main__":
    main()
