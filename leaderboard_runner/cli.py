"""Command-line interface for running all model/dataset combinations."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List

from .board_builder import rebuild_leaderboard
from .combo_executor import run_single_combo
from .io_utils import parse_csv_or_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_all.py",
        description="Evaluate all model × dataset pairs and rebuild leaderboard",
    )
    p.add_argument(
        "-m",
        "--models",
        default="gpt-4.1-nano-2025-04-14",
        help="CSV list or file path with model names",
    )
    p.add_argument(
        "-d",
        "--datasets",
        default="mmlu",
        help="CSV list or file path with dataset names",
    )
    p.add_argument(
        "-n",
        "--n_rows",
        type=int,
        default=None,
        help="sample N rows (omit for full evaluation)",
    )
    p.add_argument(
        "-s",
        "--shots",
        type=int,
        default=3,
        help="few-shot examples per prompt",
    )
    p.add_argument(
        "-w",
        "--workers",
        type=int,
        default=100,
        help="Python worker threads",
    )
    p.add_argument("--dry", action="store_true", help="print commands only")
    p.add_argument("--debug", action="store_true", help="verbose logging")
    return p


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level, format="%(levelname)s: %(message)s")

    models = parse_csv_or_file(args.models)
    datasets = parse_csv_or_file(args.datasets)

    logging.info(
        "workers=%d  shots=%d  sample=%s",
        args.workers,
        args.shots,
        "full" if args.n_rows is None else args.n_rows,
    )

    for model in models:
        for ds in datasets:
            run_single_combo(
                model=model,
                dataset=ds,
                n_rows=args.n_rows,
                shots=args.shots,
                workers=args.workers,
                dry_run=args.dry,
            )

    rebuild_leaderboard(dry_run=args.dry)
