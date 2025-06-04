#!/usr/bin/env python3
"""run_eval.py

Evaluate **one model on one dataset** and materialise all artefacts.

Outputs
-------
1. <out>.csv          – main results (gold answer, prediction, raw response …)
2. <out>_<cat>.csv    – per-category accuracy for every column in
                       ``meta.yaml::category_cols``.
3. <out>_raw.csv      – identical to the main file (dashboard download).

Extra CLI options
-----------------
--n_rows N   Sample *N* random rows **after** loading the CSV.
--verbose    Show INFO-level logs (default is WARNING).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ───────────────────────── global logging ─────────────────────────────── #
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ───────────────────────── helper utilities ───────────────────────────── #

def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("dyn", path)
    mod = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
    spec.loader.exec_module(mod)                         # type: ignore[attr-defined]
    return mod

_PERS_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def _norm(x) -> str:  # noqa: ANN001
    """Normalise Persian/Latin digits to plain integers (for answer matching)."""
    if pd.isna(x):
        return ""
    s = str(x).strip().translate(_PERS_DIGITS)
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def _read_dataset(path: str, verbose: bool = False) -> pd.DataFrame:
    """Read a CSV; fall back to the python engine if the fast C parser fails."""
    try:
        return pd.read_csv(path)  # fast C engine
    except pd.errors.ParserError as err:
        msg = (
            f"ParserError while reading {path} with the C engine:\n{err}\n"
            "→ Falling back to engine='python', on_bad_lines='skip'."
        )
        (logging.warning if verbose else logging.debug)(msg)
        return pd.read_csv(
            path,
            engine="python",
            on_bad_lines="skip",
            quoting=csv.QUOTE_MINIMAL,
            quotechar='"',
            escapechar="\\",
        )

# ───────────────────────────────── main ───────────────────────────────── #

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--evaluator", default="evaluators/mcq_evaluator.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_rows", type=int, default=None,
                        help="Sample N random rows after loading the dataset")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable INFO-level logs")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    Evaluator = getattr(_load_module(args.evaluator), "MCQEvaluator")

    model_cfg: dict[str, Any] = yaml.safe_load(Path(args.model).read_text())
    meta_cfg: dict[str, Any] = yaml.safe_load(Path(args.meta).read_text())
    df = _read_dataset(args.dataset, verbose=args.verbose)

    if args.n_rows:
        df = df.sample(n=min(args.n_rows, len(df)), random_state=42)
        logging.info("Sampling %d rows → dataframe now has %d rows",
                     args.n_rows, len(df))

    answer_col: str = meta_cfg.get("answer_col", "Key")
    question_col: str = meta_cfg.get("question_col", "question")

    if answer_col not in df.columns:
        sys.exit(
            f"❌  Column '{answer_col}' (answer_col) not found in dataset. "
            "Check meta.yaml::answer_col and CSV headers."
        )

    # ── evaluation ────────────────────────────────────────────────────── #
    evaluator = Evaluator(
        model_cfg=model_cfg,
        prompt_path=Path(args.prompt),
        meta_path=Path(args.meta),
        shots=args.shots,
        max_retries=5,
    )
    result_df = evaluator.evaluate_df(df, max_workers=args.workers)

    # ── dashboard-friendly columns ────────────────────────────────────── #
    if "Key" not in result_df.columns:
        result_df["Key"] = result_df[answer_col]

    if "Question Body" not in result_df.columns and question_col in result_df.columns:
        result_df = result_df.rename(columns={question_col: "Question Body"})

    # ── save main results ─────────────────────────────────────────────── #
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"✅  main results → {out_path}")

    # ── per-category breakdowns ───────────────────────────────────────── #
    for cat in meta_cfg.get("category_cols", []):
        if cat not in result_df.columns:
            print(f"⚠️  column '{cat}' missing – skipped")
            continue

        def _cat_acc(g: pd.DataFrame) -> float:
            return (g["pred"].map(_norm) == g["Key"].map(_norm)).mean() * 100

        try:
            cat_df = (
                result_df.groupby(cat, dropna=False)
                .apply(_cat_acc, include_groups=False)   # pandas ≥ 2.1
                .reset_index(name="Accuracy")
                .sort_values(cat)
            )
        except TypeError:
            cat_df = (
                result_df.groupby(cat, dropna=False)
                .apply(_cat_acc)                         # pandas < 2.1
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
