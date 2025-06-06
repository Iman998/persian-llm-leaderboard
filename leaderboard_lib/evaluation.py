"""CLI logic for evaluating a single model on a dataset.

This module exposes :func:`main` used by :mod:`scripts.run_eval` to
materialise evaluation results.  Helper functions handle argument
parsing, configuration loading, sampling, evaluation and result saving."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any
import shutil

import pandas as pd
import yaml

from .data_utils import _read_dataset, _norm
from .utils import _load_module

# ───────────────────────── global logging ─────────────────────────────── #
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ───────────────────────── helper utilities ───────────────────────────── #

def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--evaluator")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--n_rows",
        type=int,
        default=None,
        help="Sample N random rows after loading the dataset",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logs",
    )
    return parser.parse_args()


def load_configs(args: argparse.Namespace) -> tuple[type, dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Load model/meta YAML and dataset, returning the Evaluator class."""
    meta_cfg: dict[str, Any] = yaml.safe_load(Path(args.meta).read_text())
    if not args.evaluator:
        args.evaluator = meta_cfg.get("evaluator", "evaluators/mcq_evaluator.py")
    if not args.prompt:
        args.prompt = meta_cfg.get("prompt_template", "prompts/mcq_fewshot.jinja2")

    class_name = "".join(part.title() for part in Path(args.evaluator).stem.split("_"))
    module = _load_module(args.evaluator)
    Evaluator = getattr(module, class_name, None)
    if Evaluator is None:
        for attr in dir(module):
            if attr.lower() == class_name.lower():
                Evaluator = getattr(module, attr)
                break
    if Evaluator is None:
        raise AttributeError(
            f"Class '{class_name}' not found in {args.evaluator}"
        )

    model_cfg: dict[str, Any] = yaml.safe_load(Path(args.model).read_text())
    df = _read_dataset(args.dataset, verbose=args.verbose)
    return Evaluator, model_cfg, meta_cfg, df


def sample_dataset(df: pd.DataFrame, n_rows: int | None, verbose: bool) -> pd.DataFrame:
    """Optionally sample ``n_rows`` from ``df``."""
    if n_rows:
        df = df.sample(n=min(n_rows, len(df)), random_state=42)
        logging.info("Sampling %d rows → dataframe now has %d rows", n_rows, len(df))
    return df


def run_evaluation(
    Evaluator: type,
    model_cfg: dict[str, Any],
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Instantiate ``Evaluator`` and evaluate the dataframe."""
    evaluator = Evaluator(
        model_cfg=model_cfg,
        prompt_path=Path(args.prompt),
        meta_path=Path(args.meta),
        shots=args.shots,
        max_retries=5,
    )
    return evaluator.evaluate_df(df, max_workers=args.workers)


def save_results(
    result_df: pd.DataFrame,
    meta_cfg: dict[str, Any],
    out_path: Path,
    answer_col: str,
    question_col: str,
) -> None:
    """Write evaluation outputs and per-category breakdowns."""
    if "Key" not in result_df.columns:
        result_df["Key"] = result_df[answer_col]

    if "Question Body" not in result_df.columns and question_col in result_df.columns:
        result_df = result_df.rename(columns={question_col: "Question Body"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"✅  main results → {out_path}")

    for cat in meta_cfg.get("category_cols", []):
        if cat not in result_df.columns:
            print(f"⚠️  column '{cat}' missing – skipped")
            continue

        def _cat_acc(g: pd.DataFrame) -> float:
            return (g["pred"].map(_norm) == g["Key"].map(_norm)).mean() * 100

        try:
            cat_df = (
                result_df.groupby(cat, dropna=False)
                .apply(_cat_acc, include_groups=False)  # pandas ≥ 2.1
                .reset_index(name="Accuracy")
                .sort_values(cat)
            )
        except TypeError:
            cat_df = (
                result_df.groupby(cat, dropna=False)
                .apply(_cat_acc)  # pandas < 2.1
                .reset_index(name="Accuracy")
                .sort_values(cat)
            )

        cat_file = out_path.with_name(f"{out_path.stem}_{cat}.csv")
        cat_df.to_csv(cat_file, index=False)
        print(f"📊  {cat} breakdown → {cat_file}")

    raw_file = out_path.with_name(f"{out_path.stem}_raw.csv")
    result_df.to_csv(raw_file, index=False)
    print(f"🗃️  raw outputs → {raw_file}")


# ───────────────────────────────── main ───────────────────────────────── #

def main() -> None:  # noqa: D401
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    Evaluator, model_cfg, meta_cfg, df = load_configs(args)
    df = sample_dataset(df, args.n_rows, args.verbose)

    answer_col: str = meta_cfg.get("answer_col", "Key")
    question_col: str = meta_cfg.get("question_col", "question")

    if answer_col not in df.columns:
        sys.exit(
            f"❌  Column '{answer_col}' (answer_col) not found in dataset. "
            "Check meta.yaml::answer_col and CSV headers."
        )

    result_df = run_evaluation(Evaluator, model_cfg, df, args)

    out_path = Path(args.out)
    save_results(result_df, meta_cfg, out_path, answer_col, question_col)

    # ------------------------------------------------------------------
    # Ensure sampled runs appear on the leaderboard
    # ------------------------------------------------------------------
    if args.n_rows:
        base_stem = out_path.stem.rsplit(f"_{args.n_rows}", 1)[0]
        dst = out_path.with_name(base_stem + out_path.suffix)
        if dst != out_path:
            shutil.copy2(out_path, dst)
            for f in out_path.parent.glob(f"{out_path.stem}_*.csv"):
                new_name = f.name.replace(out_path.stem, base_stem, 1)
                shutil.copy2(f, f.with_name(new_name))

