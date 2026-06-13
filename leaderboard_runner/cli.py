"""Command-line interface for running all model/dataset combinations."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List

from .board_builder import rebuild_leaderboard
from .combo_executor import run_single_combo
from .io_utils import parse_csv_or_file
from .meta_utils import JUDGE_MODES


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
    p.add_argument(
        "--judge",
        action="store_true",
        help="run LLM-judge evaluation for text-generation or translation tasks",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help="override the judge model configured in dataset meta.yaml",
    )
    p.add_argument(
        "--judge-mode",
        choices=JUDGE_MODES,
        default="reference",
        help="judge with the gold reference, without it, or run both passes",
    )
    p.add_argument(
        "--judge-only",
        action="store_true",
        help="reuse an existing candidate result CSV without generating it again",
    )
    p.add_argument("--dry", action="store_true", help="print commands only")
    p.add_argument("--debug", action="store_true", help="verbose logging")
    return p


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject invalid numeric options before spawning subprocesses."""
    if args.n_rows is not None and args.n_rows <= 0:
        parser.error("--n_rows must be a positive integer")
    if args.shots < 0:
        parser.error("--shots must be zero or a positive integer")
    if args.workers <= 0:
        parser.error("--workers must be a positive integer")
    if args.judge_only and not args.judge:
        parser.error("--judge-only requires --judge")


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

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
                judge=args.judge,
                judge_model=args.judge_model,
                judge_mode=args.judge_mode,
                judge_only=args.judge_only,
                dry_run=args.dry,
            )

    rebuild_leaderboard(dry_run=args.dry)
