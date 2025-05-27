#!/usr/bin/env python3
"""
Run one (MODEL × DATASET) evaluation and materialise all artifacts.

Outputs
-------
1. <out>.csv            : main results (gold, pred, raw …)
2. <out>_<cat>.csv      : accuracy by each category in meta.yaml::category_cols
3. <out>_raw.csv        : same as main (dashboard download convenience)
"""
from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import sys
from typing import Any

import pandas as pd
import yaml

# ────────────────────────────── helpers ──────────────────────────────── #
def _load_module(path: str):
    """Import an arbitrary .py file and return the module object."""
    spec = importlib.util.spec_from_file_location("dyn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

_PERS_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

def _norm(x) -> str:
    """Normalise keys/preds: Persian digits→ASCII, 1.0→1, strip spaces."""
    if pd.isna(x):
        return ""
    s = str(x).strip().translate(_PERS_DIGITS)
    try:
        return str(int(float(s)))
    except ValueError:
        return s

# ───────────────────────────────── main ──────────────────────────────── #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  required=True)
    p.add_argument("--meta",     required=True)
    p.add_argument("--model",    required=True)
    p.add_argument("--prompt",   required=True)
    p.add_argument("--shots",    type=int, default=0)
    p.add_argument("--workers",  type=int, default=4)
    p.add_argument("--evaluator", default="evaluators/mcq_evaluator.py")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    Evaluator = getattr(_load_module(args.evaluator), "MCQEvaluator")

    model_cfg: dict[str, Any] = yaml.safe_load(Path(args.model).read_text())
    meta_cfg : dict[str, Any] = yaml.safe_load(Path(args.meta ).read_text())
    df = pd.read_csv(args.dataset)

    evaluator = Evaluator(
        model_cfg=model_cfg,
        prompt_path=Path(args.prompt),
        meta_path=Path(args.meta),
        shots=args.shots,
    )
    result_df = evaluator.evaluate_df(df, max_workers=args.workers)

    # ── save main results ─────────────────────────────────────────────── #
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"✅  main results → {out_path}")

    # ── category breakdowns (robust accuracy) ─────────────────────────── #
    for cat in meta_cfg.get("category_cols", []):
        if cat not in result_df.columns:
            print(f"⚠️  column '{cat}' missing – skipped")
            continue

        cat_df = (
            result_df
            .groupby(cat, dropna=False)
            .apply(
                lambda g: (g["pred"].map(_norm) == g["Key"].map(_norm)).mean() * 100,
                include_groups=False,          # avoid future pandas warning
            )
            .reset_index(name="Accuracy")
            .sort_values(cat)
        )
        cat_file = out_path.with_name(f"{out_path.stem}_{cat}.csv")
        cat_df.to_csv(cat_file, index=False)
        print(f"📊  {cat} breakdown → {cat_file}")

    # ── raw copy for dashboard download ───────────────────────────────── #
    raw_file = out_path.with_name(f"{out_path.stem}_raw.csv")
    result_df.to_csv(raw_file, index=False)
    print(f"🗃️  raw outputs → {raw_file}")

if __name__ == "__main__":
    if sys.version_info < (3, 9):
        sys.exit("Python ≥ 3.9 required")
    main()
