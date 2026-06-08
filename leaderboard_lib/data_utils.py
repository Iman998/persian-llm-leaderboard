"""Data loading and normalisation helpers for evaluation.

This module provides utilities to read datasets robustly and normalise
Persian or Latin digits for consistent comparisons."""

from __future__ import annotations

import csv
import logging
import re
from typing import Any

import pandas as pd

_PERS_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
_TAG_SEPARATOR = re.compile(r"\s*[|,]\s*")


def _norm(x: Any) -> str:
    """Normalise Persian/Latin digits to plain integers (for answer matching)."""
    if pd.isna(x):
        return ""
    s = str(x).strip().translate(_PERS_DIGITS)
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def split_tag_value(value: Any) -> list[str]:
    """Return distinct tags from a comma- or pipe-separated cell."""
    if value is None or (not isinstance(value, (list, tuple, set)) and pd.isna(value)):
        return []

    values = value if isinstance(value, (list, tuple, set)) else [value]
    tags: list[str] = []
    seen: set[str] = set()
    for item in values:
        if item is None or pd.isna(item):
            continue
        for tag in _TAG_SEPARATOR.split(str(item).strip()):
            if tag and tag not in seen:
                tags.append(tag)
                seen.add(tag)
    return tags


def explode_tag_column(
    df: pd.DataFrame, column: str, *, keep_empty: bool = False
) -> pd.DataFrame:
    """Expand a multi-tag column so each tag has its own row."""
    if column not in df.columns:
        return df.copy()

    expanded = df.copy()
    expanded[column] = expanded[column].map(split_tag_value)
    if keep_empty:
        expanded[column] = expanded[column].map(
            lambda tags: tags if tags else [pd.NA]
        )
    return expanded.explode(column)


def filter_rows_by_tags(
    df: pd.DataFrame, filters: dict[str, set[str]]
) -> pd.DataFrame:
    """Apply AND-across-columns and OR-within-column tag filters."""
    filtered = df
    for column, allowed in filters.items():
        if column not in filtered.columns or not allowed:
            continue
        mask = filtered[column].map(
            lambda value: bool(set(split_tag_value(value)) & allowed)
        )
        filtered = filtered[mask]
    return filtered


def tag_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Count the number of source rows assigned to each distinct tag."""
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "Count", "Percent"])

    counts = (
        explode_tag_column(df[[column]], column)
        .dropna(subset=[column])
        .groupby(column, sort=False)
        .size()
        .rename("Count")
        .reset_index()
        .sort_values(["Count", column], ascending=[False, True])
        .reset_index(drop=True)
    )
    counts["Percent"] = (
        counts["Count"].div(len(df)).mul(100) if len(df) else 0.0
    )
    return counts


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
