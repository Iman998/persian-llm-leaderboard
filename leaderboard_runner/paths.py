"""
paths.py
~~~~~~~~
Centralised, single-source definition of project paths.  Importing this
module avoids accidental divergence across files.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent         # repository root
DATASETS_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
DASHBOARD_DIR = ROOT / "dashboard"
LEADERBOARD_OUT = DASHBOARD_DIR / "leaderboard.csv"

RUN_EVAL_SCRIPT = ROOT / "scripts" / "run_eval.py"
BUILD_BOARD_SCRIPT = ROOT / "scripts" / "build_leaderboard.py"

__all__ = [
    "ROOT",
    "DATASETS_DIR",
    "RESULTS_DIR",
    "MODELS_DIR",
    "LEADERBOARD_OUT",
    "RUN_EVAL_SCRIPT",
    "BUILD_BOARD_SCRIPT",
]
