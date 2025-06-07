#!/usr/bin/env python3
"""CLI entry point for building the LLM-as-Judge leaderboard."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from leaderboard_lib.llm_judge_board import main

if __name__ == "__main__":
    if sys.version_info < (3, 10):
        sys.exit("Python ≥ 3.10 required")
    main()
