#!/usr/bin/env python3

"""CLI entry point for building the aggregated leaderboard.

All heavy lifting is performed by :func:`leaderboard_lib.leaderboard.main`.
This wrapper simply forwards command line arguments such as ``--language``.
"""

import sys
from pathlib import Path

# Ensure the repository root is on ``sys.path`` before importing :mod:`leaderboard_lib`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from leaderboard_lib.leaderboard import main

if __name__ == "__main__":
    if sys.version_info < (3, 10):
        sys.exit("Python ≥ 3.10 required")
    main(sys.argv[1:])
