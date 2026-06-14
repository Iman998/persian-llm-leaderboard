"""
Centralised definitions of important project paths.

Placing them in a dedicated module prevents circular imports
and allows other helpers to import the same constants without
re‑calculating directories independently.
"""
from pathlib import Path

ROOT_DIR      = Path(__file__).resolve().parents[2]    # project root
RESULTS_DIR   = ROOT_DIR / "results"
DATASETS_DIR  = ROOT_DIR / "data"
MODELS_DIR    = ROOT_DIR / "models"
DASHBOARD_CSV = ROOT_DIR / "dashboard" / "leaderboard.csv"
DASHBOARD_FA_CSV = ROOT_DIR / "dashboard" / "leaderboard_fa.csv"
DASHBOARD_EN_CSV = ROOT_DIR / "dashboard" / "leaderboard_en.csv"
TRANSLATION_CSV = ROOT_DIR / "dashboard" / "translation_leaderboard.csv"
SUMMARIZATION_CSV = ROOT_DIR / "dashboard" / "summarization_leaderboard.csv"
SUMMARIZATION_FA_CSV = ROOT_DIR / "dashboard" / "summarization_leaderboard_fa.csv"
SUMMARIZATION_EN_CSV = ROOT_DIR / "dashboard" / "summarization_leaderboard_en.csv"
BATTLE_CSV = ROOT_DIR / "dashboard" / "battle_board.csv"
