#!/usr/bin/env python3
"""
Entry-point kept for shell wrapper compatibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Ensure repository root is importable so `leaderboard_runner` can be found.
# This handles cases where the script is executed via an absolute/relative
# path instead of `python -m`.
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]   # repo root (one level above /scripts)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leaderboard_runner.cli import main     # noqa: E402  (import after sys.path fix)

if __name__ == "__main__":
    if sys.version_info < (3, 10):
        sys.exit("Python ≥ 3.10 required")
    main()
