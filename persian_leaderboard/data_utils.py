"""Data loading and normalisation helpers for evaluation.

This module provides utilities to read datasets robustly and normalise
Persian or Latin digits for consistent comparisons."""

from __future__ import annotations

import csv
import logging
from typing import Any

import pandas as pd

_PERS_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def _norm(x: Any) -> str:
    """Normalise Persian/Latin digits to plain integers (for answer matching)."""
    if pd.isna(x):
        return ""
    s = str(x).strip().translate(_PERS_DIGITS)
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def _read_dataset(path: str, verbose: bool = False) -> pd.DataFrame:
    """Read a CSV; fall back to the python engine if the fast C parser fails."""
    try:
        return pd.read_csv(path)  # fast C engine
    except pd.errors.ParserError as err:
        msg = (
            f"ParserError while reading {path} with the C engine:\n{err}\n"
            "→ Falling back to engine='python', on_bad_lines='skip'."
        )
        (logging.warning if verbose else logging.debug)(msg)
        return pd.read_csv(
            path,
            engine="python",
            on_bad_lines="skip",
            quoting=csv.QUOTE_MINIMAL,
            quotechar='"',
            escapechar="\\",
        )
