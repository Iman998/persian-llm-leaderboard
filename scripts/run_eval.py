#!/usr/bin/env python3
"""CLI entry point for running a single evaluation.

This wrapper delegates to :func:`persian_leaderboard.evaluation.main` which
implements the full evaluation logic."""

from persian_leaderboard.evaluation import main

if __name__ == "__main__":
    import sys
    if sys.version_info < (3, 9):
        sys.exit("Python ≥ 3.9 required")
    main()
