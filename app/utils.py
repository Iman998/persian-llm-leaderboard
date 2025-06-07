"""Utility helpers for the Streamlit dashboard.

This module contains reusable functions for reading CSV files, parsing
metadata, and scanning result directories. They are separated from the
``streamlit_app`` module so that the UI code remains concise.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
import streamlit as st

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
DATASETS_DIR = ROOT_DIR / "data"        # where <dataset>/meta.yaml lives
DASHBOARD_CSV = ROOT_DIR / "dashboard" / "leaderboard.csv"
MODELS_DIR = ROOT_DIR / "models"


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with caching enabled."""
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_meta(ds: str) -> Dict[str, object]:
    """Return metadata about the dataset.

    The returned dictionary always contains ``question_col``, ``answer_col`` and
    ``category_cols`` keys. Values fall back to sensible defaults when
    ``meta.yaml`` is missing.
    """
    meta_file = DATASETS_DIR / ds / "meta.yaml"
    if meta_file.exists():
        cfg = yaml.safe_load(meta_file.read_text())
    else:
        cfg = {}
    return {
        "question_col": cfg.get("question_col", "question"),
        "answer_col": cfg.get("answer_col", "Key"),
        "category_cols": cfg.get("category_cols", []),
    }


def numeric_cols(df: pd.DataFrame) -> List[str]:
    """List the numeric columns in ``df``."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def medal_colors(avgs: pd.Series) -> List[str]:
    """Map ``Average`` values to medal colors.

    Returns a list of hex color strings (or ``""`` when no medal applies)
    corresponding to each value in ``avgs`` sorted by rank. The top three
    values receive gold, silver and bronze respectively.
    """

    ranks = avgs.rank(method="first", ascending=False).astype(int)
    medal_map = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}
    return [medal_map.get(r, "") for r in ranks]


def gradient(df: pd.DataFrame):
    """Return a styled DataFrame with numeric gradients and medal highlights."""

    try:
        import matplotlib  # noqa: F401

        medal_cols = medal_colors(df.get("Average", pd.Series(dtype=float)))

        def _avg_style(_: pd.Series) -> List[str]:
            return [f"background-color: {c}" if c else "" for c in medal_cols]

        def _model_style(_: pd.Series) -> List[str]:
            return [f"color: {c}" if c else "" for c in medal_cols]

        styler = (
            df.style.background_gradient(
                axis=0, cmap="RdYlGn", subset=numeric_cols(df)
            )
            .apply(_avg_style, subset=["Average"], axis=0)
            .apply(_model_style, subset=["Model"], axis=0)
        )
        return styler
    except ImportError:  # pragma: no cover - optional dependency
        return df


# ---------------------------------------------------------------------------
# File-name parsing helpers
# ---------------------------------------------------------------------------
model_names = sorted([p.stem for p in MODELS_DIR.glob("*.yaml")], key=len, reverse=True)
models_alt = "|".join(map(re.escape, model_names))
FILE_RE_NEW = re.compile(rf"^(?P<model>{models_alt})(?:_(?P<suffix>.+?))?\.csv$")
FILE_RE_LEGACY = re.compile(rf"^(?P<dataset>.+?)_(?P<model>{models_alt})(?:_(?P<suffix>.+?))?\.csv$")


def parse_file(p: Path) -> Tuple[str, str, str] | None:
    """Extract dataset, model and suffix from ``p``.

    Returns ``None`` when the filename does not match the expected patterns or
    represents a temporary sample file.
    """
    m = FILE_RE_NEW.match(p.name)
    if m:
        try:
            ds = p.relative_to(RESULTS_DIR).parts[0]
        except ValueError:
            ds = p.parent.parent.name
        mdl, suf = m.group("model", "suffix")
        if suf and suf.isdigit():
            return None
        return ds, mdl, suf or ""

    m = FILE_RE_LEGACY.match(p.name)
    if m:
        ds, mdl, suf = m.group("dataset", "model", "suffix")
        if suf and suf.isdigit():
            return None
        return ds, mdl, suf or ""

    return None


def scan_result_maps() -> Tuple[List[str], Dict[Tuple[str, str], Path], Dict[Tuple[str, str], Path], Dict[Tuple[str, str, str], Path]]:
    """Scan :data:`RESULTS_DIR` and organise discovered CSV files.

    Returns
    -------
    datasets : list[str]
        Sorted list of dataset names that have results.
    main_map : dict[(str, str), Path]
        Mapping ``(dataset, model)`` → main result CSV.
    raw_map : dict[(str, str), Path]
        Mapping ``(dataset, model)`` → raw output CSV.
    cat_map : dict[(str, str, str), Path]
        Mapping ``(dataset, model, category)`` → per-category CSV.
    """
    main_map: Dict[Tuple[str, str], Path] = {}
    raw_map: Dict[Tuple[str, str], Path] = {}
    cat_map: Dict[Tuple[str, str, str], Path] = {}

    for p in RESULTS_DIR.rglob("*.csv"):
        parsed = parse_file(p)
        if not parsed:
            continue
        ds, mdl, suf = parsed
        if suf == "raw" or suf.endswith("_raw"):
            raw_map[(ds, mdl)] = p
        elif suf.isdigit() or suf.split("_", 1)[0].isdigit():
            continue  # ignore n_rows sample outputs
        elif suf:
            cat_map[(ds, mdl, suf)] = p
        else:
            main_map[(ds, mdl)] = p

    datasets = sorted({k[0] for k in main_map})
    return datasets, main_map, raw_map, cat_map
