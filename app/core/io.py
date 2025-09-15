"""
I/O utilities – loading CSV files and dataset metadata.

All costly operations are wrapped in `st.cache_data` so that
the same file is only read once per session.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
import yaml

from .paths import DATASETS_DIR

__all__ = ["load_csv", "load_meta", "numeric_cols"]


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file with Streamlit cache."""
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_meta(ds_name: str) -> Dict[str, object]:
    """
    Load the `meta.yaml` file that accompanies each evaluation
    dataset.

    Fallbacks are provided when the YAML file or individual
    keys are absent to keep the UI resilient.
    """
    meta_file = DATASETS_DIR / ds_name / "meta.yaml"
    cfg = yaml.safe_load(meta_file.read_text()) if meta_file.exists() else {}
    return {
        "question_col":  cfg.get("question_col", "question"),
        "answer_col":    cfg.get("answer_col", "Key"),
        "category_cols": cfg.get("category_cols", []),
        "choice_cols":   cfg.get("choice_cols", []),
        "language":     cfg.get("language", ""),
        "description":  cfg.get("description", ""),
        "board":        cfg.get("board", "leaderboard"),
    }


def numeric_cols(df: pd.DataFrame) -> List[str]:
    """Return a list of numeric columns inside `df`."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
