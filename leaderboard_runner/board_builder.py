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
    cmd = [
        sys.executable,
        paths.BUILD_BOARD_SCRIPT,
        "--results_dir",
        paths.RESULTS_DIR,
        "--datasets_dir",
        paths.DATASETS_DIR,
        "--out",
        paths.LEADERBOARD_OUT,
    ]
    if dry_run:
        print(" ".join(map(str, cmd)))
    else:
        subprocess.run([str(c) for c in cmd], check=True)
        logger.info("DONE Leaderboard updated → %s", paths.LEADERBOARD_OUT)
