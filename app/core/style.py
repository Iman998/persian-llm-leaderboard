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

import streamlit as st

__all__ = ["apply_gradient", "render_styler"]


def _medal_colors(avgs: pd.Series) -> List[str]:
    """Return a list of gold/silver/bronze hex colours by rank."""
    ranks = avgs.rank(method="first", ascending=False).astype(int)
    return [{1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}.get(r, "") for r in ranks]


def apply_gradient(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Colour numeric columns with a RdYlGn gradient and apply
    custom colouring rules for other columns.
    The top‑3 ``Average`` rows receive medal backgrounds.
    """
    numeric = numeric_cols(df)
    styler = df.style
    if numeric:
        styler = styler.background_gradient(axis=0, cmap="RdYlGn", subset=numeric)

    def _param_column_style(series: pd.Series) -> mcolors.Normalize:
        vals = pd.to_numeric(series.astype(str).str.replace("B", "", regex=False), errors="coerce")
        vmin, vmax = vals.min(), vals.max()
        if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
            return mcolors.Normalize(0, 1)
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    blues = cm.Blues

    def _param_style(val: str, norm: mcolors.Normalize) -> str:
        num = pd.to_numeric(str(val).replace("B", ""), errors="coerce")
        ratio = 1.0 if np.isnan(num) else norm(num)
        colour = mcolors.to_hex(blues(ratio))
        return f"background-color:{colour}"

    for col in [c for c in ["Parameters", "Active Parameters"] if c in df.columns]:
        norm = _param_column_style(df[col])
        df[col] = df[col].astype(str).str.replace("B", "", regex=False)
        styler = styler.applymap(lambda v: _param_style(v, norm), subset=[col])

    # License column colours
    if "License" in df.columns:
        def _license_style(val: str) -> str:
            v = str(val).strip().lower()
            if v == "proprietary":
                colour = "#ff4d4d"  # red
            elif v == "mit":
                colour = "#008000"  # dark green
            else:
                colour = "#90ee90"  # light green
            return f"background-color:{colour}"

        styler = styler.applymap(_license_style, subset=["License"])

    # Highlight specific organization
    if "Organization" in df.columns:
        def _org_style(val: str) -> str:
            return "background-color:#00ff00" if str(val).strip() == "ZharfaTech" else ""

        styler = styler.applymap(_org_style, subset=["Organization"])

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


def render_styler(styler: pd.io.formats.style.Styler) -> None:
    """Display a styled DataFrame in Streamlit with interactive features."""
    st.dataframe(styler, use_container_width=True, hide_index=True)

