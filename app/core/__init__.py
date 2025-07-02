"""
Ensure `persian-llm-leaderboard` root is on sys.path when any `app.core`
module is imported standalone (e.g. by Streamlit multipage runner).
"""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
