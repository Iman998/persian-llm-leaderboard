"""Command-line interface for running all model/dataset combinations."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List

from .battle_executor import run_battle
from .board_builder import rebuild_leaderboard
from .combo_executor import run_single_combo
from .io_utils import parse_csv_or_file
from .league_executor import run_league
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
    p.add_argument(
        "--battle",
        action="store_true",
        help="compare two models using their existing result CSVs",
    )
    p.add_argument(
        "--battle-only",
        action="store_true",
        help="run battles without generating or judging candidate outputs",
    )
    p.add_argument("--battle-model-1", help="first model stub in the battle")
    p.add_argument("--battle-model-2", help="second model stub in the battle")
    p.add_argument(
        "--battle-judge-model",
        help="judge model stub for pairwise battles",
    )
    p.add_argument(
        "--league",
        action="store_true",
        help="run or continue a named sampled Elo league",
    )
    p.add_argument(
        "--league-only",
        action="store_true",
        help="run the league without generating candidate outputs",
    )
    p.add_argument("--league-name", help="persistent name of the league")
    p.add_argument(
        "--league-models",
        default=None,
        help="CSV list or file path with league models; defaults to --models",
    )
    p.add_argument(
        "--league-datasets",
        default=None,
        help="CSV list or file path with generation datasets; defaults to --datasets",
    )
    p.add_argument(
        "--league-judge-model",
        default=None,
        help="judge model for a new league",
    )
    p.add_argument(
        "--league-matches",
        type=int,
        default=10,
        help="number of new matches to schedule",
    )
    p.add_argument(
        "--league-rows-per-match",
        type=int,
        default=20,
        help="sampled rows judged in each match",
    )
    p.add_argument(
        "--league-k-factor",
        type=float,
        default=32.0,
        help="Elo K-factor",
    )
    p.add_argument(
        "--league-initial-elo",
        type=float,
        default=1000.0,
        help="starting Elo for a new league",
    )
    p.add_argument(
        "--league-calibration-games",
        type=int,
        default=2,
        help="games per model before nearby-Elo matchmaking",
    )
    p.add_argument(
        "--league-repeat-penalty",
        type=float,
        default=64.0,
        help="rating-distance penalty for repeated pairs",
    )
    p.add_argument(
        "--league-seed",
        type=int,
        default=42,
        help="seed for deterministic row rotation",
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
    if args.battle_only and not args.battle:
        parser.error("--battle-only requires --battle")
    if args.league_only and not args.league:
        parser.error("--league-only requires --league")
    if args.league:
        if not args.league_name or not args.league_name.strip():
            parser.error("--league requires --league-name")
        if args.league_matches <= 0:
            parser.error("--league-matches must be a positive integer")
        if args.league_rows_per_match <= 0:
            parser.error("--league-rows-per-match must be a positive integer")
        if args.league_k_factor <= 0:
            parser.error("--league-k-factor must be positive")
        if args.league_calibration_games < 0:
            parser.error("--league-calibration-games cannot be negative")
        if args.league_repeat_penalty < 0:
            parser.error("--league-repeat-penalty cannot be negative")


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level, format="%(levelname)s: %(message)s")

    models = parse_csv_or_file(args.models)
    datasets = parse_csv_or_file(args.datasets)
    league_models = parse_csv_or_file(args.league_models or args.models)
    league_datasets = parse_csv_or_file(args.league_datasets or args.datasets)
    battle_model_1 = args.battle_model_1
    battle_model_2 = args.battle_model_2
    if args.battle:
        if not battle_model_1 and models:
            battle_model_1 = models[0]
        if not battle_model_2 and len(models) > 1:
            battle_model_2 = models[1]
        if not battle_model_1 or not battle_model_2:
            parser.error(
                "--battle requires two models via --battle-model-1/2 "
                "or the first two --models entries"
            )

    logging.info(
        "workers=%d  shots=%d  sample=%s",
        args.workers,
        args.shots,
        "full" if args.n_rows is None else args.n_rows,
    )

    if not args.battle_only and not args.league_only:
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

    if args.battle:
        for dataset in datasets:
            run_battle(
                dataset=dataset,
                model_1=battle_model_1,
                model_2=battle_model_2,
                judge_model=args.battle_judge_model,
                n_rows=args.n_rows,
                workers=args.workers,
                dry_run=args.dry,
            )

    if args.league:
        run_league(
            name=args.league_name,
            models=league_models,
            datasets=league_datasets,
            judge_model=args.league_judge_model,
            matches=args.league_matches,
            rows_per_match=args.league_rows_per_match,
            workers=args.workers,
            k_factor=args.league_k_factor,
            initial_elo=args.league_initial_elo,
            calibration_games=args.league_calibration_games,
            repeat_penalty=args.league_repeat_penalty,
            seed=args.league_seed,
            dry_run=args.dry,
        )

    rebuild_leaderboard(dry_run=args.dry)
