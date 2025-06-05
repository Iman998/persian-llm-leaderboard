#!/usr/bin/env python3
"""CLI entry point for building the aggregated leaderboard.

All heavy lifting is performed by :func:`persian_leaderboard.leaderboard.main`."""

from persian_leaderboard.leaderboard import main

if __name__ == "__main__":
    import sys
    if sys.version_info < (3, 9):
        sys.exit("Python ≥ 3.9 required")
    main()
