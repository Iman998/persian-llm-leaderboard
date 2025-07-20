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
LEADERBOARD_FA_OUT = DASHBOARD_DIR / "leaderboard_fa.csv"
LEADERBOARD_EN_OUT = DASHBOARD_DIR / "leaderboard_en.csv"
LEADERBOARD_JUDGE_OUT = DASHBOARD_DIR / "llm_judge_board.csv"

RUN_EVAL_SCRIPT = ROOT / "scripts" / "run_eval.py"
BUILD_BOARD_SCRIPT = ROOT / "scripts" / "build_leaderboard.py"
BUILD_JUDGE_BOARD_SCRIPT = ROOT / "scripts" / "build_llm_judge_board.py"

__all__ = [
    "ROOT",
    "DATASETS_DIR",
    "RESULTS_DIR",
    "MODELS_DIR",
    "LEADERBOARD_OUT",
    "LEADERBOARD_FA_OUT",
    "LEADERBOARD_EN_OUT",
    "LEADERBOARD_JUDGE_OUT",
    "RUN_EVAL_SCRIPT",
    "BUILD_BOARD_SCRIPT",
    "BUILD_JUDGE_BOARD_SCRIPT",
]
