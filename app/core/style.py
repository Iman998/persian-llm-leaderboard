"""
Styling helpers for Streamlit DataFrames.

`apply_gradient` shades numeric columns with a red‑→yellow‑→green
gradient and highlights the top‑3 `Average` rows with medal colours.
"""
from __future__ import annotations

from typing import List, Callable

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
    Colour numeric columns with a RdYlGn gradient and apply
    custom colouring rules for other columns.
    The top‑3 ``Average`` rows receive medal backgrounds.
    """
    numeric = numeric_cols(df)
    rdylgn = cm.get_cmap("RdYlGn")
    styler = df.style

    def _score_style(val: str) -> str:
        num = pd.to_numeric(val, errors="coerce")
        if np.isnan(num):
            return ""
        num = max(0.0, min(100.0, float(num)))
        colour = mcolors.to_hex(rdylgn(num / 100.0))
        return (
            f"background:linear-gradient(90deg,{colour} {num}%,transparent {num}%);"
        )

    if numeric:
        styler = styler.applymap(_score_style, subset=numeric)

    def _param_column_style(series: pd.Series) -> mcolors.Normalize:
        vals = pd.to_numeric(series.astype(str).str.replace("B", "", regex=False), errors="coerce")
        vmin, vmax = vals.min(), vals.max()
        if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
            return mcolors.Normalize(0, 1)
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    blues = cm.Blues

    def _param_style(norm: mcolors.Normalize) -> Callable[[str], str]:
        def _style(val: str) -> str:
            num = pd.to_numeric(str(val).replace("B", ""), errors="coerce")
            ratio = 1.0 if np.isnan(num) else norm(num)
            ratio = 0.0 if np.isnan(ratio) else ratio
            # lighten max colour by scaling ratio
            colour = mcolors.to_hex(blues(0.2 + 0.6 * ratio))
            return f"background-color:{colour}"
        return _style

    for col in [c for c in ["Parameters", "Active Parameters"] if c in df.columns]:
        norm = _param_column_style(df[col])
        df[col] = df[col].astype(str).str.replace("B", "", regex=False)
        styler = styler.applymap(_param_style(norm), subset=[col])

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

