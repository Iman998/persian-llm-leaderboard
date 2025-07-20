"""
board_builder.py
~~~~~~~~~~~~~~~~
Thin wrapper around ``scripts/build_leaderboard.py`` so CLI remains
decoupled from subprocess details.
"""

import logging
import subprocess
import sys
from pathlib import Path

from . import paths

logger = logging.getLogger(__name__)


def rebuild_leaderboard(*, dry_run: bool = False) -> None:
    """Aggregate results into dashboard/leaderboard.csv."""
    base_cmd = [
        sys.executable,
        paths.BUILD_BOARD_SCRIPT,
        "--results_dir",
        paths.RESULTS_DIR,
        "--datasets_dir",
        paths.DATASETS_DIR,
    ]

    boards = [
        (paths.LEADERBOARD_OUT, "all"),
        (paths.LEADERBOARD_FA_OUT, "fa"),
        (paths.LEADERBOARD_EN_OUT, "en"),
    ]

    for out_path, lang in boards:
        cmd = base_cmd + ["--out", out_path]
        if lang != "all":
            cmd += ["--lang", lang]
        if dry_run:
            print(" ".join(map(str, cmd)))
        else:
            subprocess.run([str(c) for c in cmd], check=True)
            logger.info("DONE Leaderboard updated → %s", out_path)

    # Build LLM-judge board ----------------------------------------------- #
    judge_cmd = [
        sys.executable,
        paths.BUILD_JUDGE_BOARD_SCRIPT,
        "--results_dir",
        paths.RESULTS_DIR,
        "--datasets_dir",
        paths.DATASETS_DIR,
        "--out",
        paths.LEADERBOARD_JUDGE_OUT,
    ]
    if dry_run:
        print(" ".join(map(str, judge_cmd)))
    else:
        subprocess.run([str(c) for c in judge_cmd], check=True)
        logger.info("DONE LLM-judge board updated → %s", paths.LEADERBOARD_JUDGE_OUT)
