"""
Styling helpers for Streamlit DataFrames.

`apply_gradient` shades numeric columns with a red‑→yellow‑→green
gradient and highlights the top‑3 `Average` rows with medal colours.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from matplotlib import cm, colors as mcolors

from .io import numeric_cols

__all__ = ["apply_gradient"]


def _medal_colors(avgs: pd.Series) -> List[str]:
    """Return a list of gold/silver/bronze hex colours by rank."""
    ranks = avgs.rank(method="first", ascending=False).astype(int)
    return [{1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}.get(r, "") for r in ranks]


def apply_gradient(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Colour all numeric columns with RdYlGn gradient.
    If an “Average” column exists the top‑3 rows get medals.
    """
    styler = df.style.background_gradient(
        axis=0, cmap="RdYlGn", subset=numeric_cols(df)
    )

    if "Average" in df.columns:
        avgs = df["Average"].astype(float)
        medals = _medal_colors(avgs)

        def _rowstyles(_: pd.Series) -> List[str]:
            return [
                f"background-color:{m};color:#000" if m else "" for m in medals
            ]

        styler = styler.apply(_rowstyles, subset=["Average"], axis=0)
        if "Model" in df.columns:
            styler = styler.apply(_rowstyles, subset=["Model"], axis=0)
        if "Rank" in df.columns:
            styler = styler.apply(_rowstyles, subset=["Rank"], axis=0)

    return styler

