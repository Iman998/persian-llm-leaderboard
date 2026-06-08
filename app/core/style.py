"""
Shared visual and DataFrame styling helpers for the Streamlit app.

`apply_gradient` shades numeric columns with a soft green scale and
highlights the top‑3 `Average` rows with restrained medal colours.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from matplotlib import cm, colors as mcolors

from .io import numeric_cols

import streamlit as st

__all__ = [
    "SCORE_CMAP",
    "apply_gradient",
    "inject_global_styles",
    "page_header",
    "render_styler",
]

SCORE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "soft_score",
    [
        "#e8a4a4",
        "#f3ce9d",
        "#f4e7b2",
        "#cde3bd",
        "#73ad87",
    ],
)


def inject_global_styles() -> None:
    """Apply the dashboard's visual system once per Streamlit rerun."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #202521;
            --muted: #69716b;
            --line: #dce1dc;
            --surface: #ffffff;
            --canvas: #f2f5f2;
            --accent: #23885a;
            --accent-dark: #196b45;
            --accent-soft: #eaf5ef;
        }

        .stApp {
            background: var(--canvas);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: color-mix(in srgb, var(--canvas) 92%, transparent);
            border-bottom: 1px solid var(--line);
            backdrop-filter: blur(10px);
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1480px;
            padding: 2.25rem 2.5rem 4rem;
        }

        [data-testid="stSidebar"] {
            background: #26302a;
            border-right: 3px solid #4d8b66;
        }

        [data-testid="stSidebar"] * {
            color: #edf1ee;
        }

        [data-testid="stSidebar"] [data-baseweb="radio"] > div,
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="base-input"] {
            background: #303b34;
            border-color: #4a594f;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding: .58rem .7rem;
            margin-bottom: .28rem;
            border: 1px solid transparent;
            border-radius: 6px;
            transition: background .16s ease, border-color .16s ease;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: #303b34;
            border-color: #4a594f;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: #35443a;
            border-color: #587060;
            box-shadow: inset 3px 0 0 #42a875;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
            color: #fff;
            font-weight: 750;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
            display: none;
        }

        [data-testid="stSidebar"] hr {
            border-color: #414648;
        }

        .app-brand {
            padding: .25rem 0 1.2rem;
            border-bottom: 1px solid #414648;
            margin-bottom: 1.1rem;
        }

        .app-brand__mark {
            display: inline-grid;
            width: 2.2rem;
            height: 2.2rem;
            place-items: center;
            margin-bottom: .7rem;
            border-radius: 6px;
            background: #23885a;
            border: 1px solid #196b45;
            font-size: .85rem;
            font-weight: 800;
            color: #fff;
        }

        .app-brand__name {
            color: #fff;
            font-size: 1.06rem;
            font-weight: 720;
            line-height: 1.25;
        }

        .app-brand__meta {
            margin-top: .3rem;
            color: #a8b1ab;
            font-size: .76rem;
        }

        .page-heading {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1.5rem;
            padding: .2rem 0 .75rem;
            margin-bottom: 1rem;
        }

        .page-heading h1 {
            margin: 0;
            color: var(--ink);
            font-size: clamp(1.75rem, 3vw, 2.5rem);
            font-weight: 760;
            line-height: 1.1;
            letter-spacing: 0;
        }

        .page-heading p {
            max-width: 720px;
            margin: .5rem 0 0;
            color: var(--muted);
            font-size: .96rem;
            line-height: 1.55;
        }

        h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        [data-testid="stMetric"] {
            min-height: 98px;
            padding: 1rem 1.1rem;
            background: var(--surface);
            border: 1px solid var(--line);
            border-top: 2px solid #77a88c;
            border-radius: 6px;
            box-shadow: 0 1px 2px rgb(17 19 21 / 4%);
            transition: transform .16s ease, border-color .16s ease,
                box-shadow .16s ease;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: #aebcaf;
            box-shadow: 0 7px 20px rgb(38 67 48 / 8%);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-size: 1.55rem;
        }

        [data-testid="stDataFrame"] {
            overflow: hidden;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 6px;
            box-shadow: 0 1px 3px rgb(31 50 37 / 5%);
        }

        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 1.2rem;
            padding: 0;
            background: transparent;
            border-bottom: 1px solid var(--line);
            border-radius: 0;
        }

        [data-testid="stTabs"] [data-baseweb="tab"] {
            height: 2.9rem;
            padding: 0 .05rem;
            border-radius: 0;
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            background: transparent;
            color: var(--accent);
            font-weight: 700;
        }

        .stButton > button,
        .stDownloadButton > button {
            min-height: 2.55rem;
            border-color: #cdd1d6;
            border-radius: 5px;
            background: #fff;
            color: #25282c;
            font-weight: 650;
            box-shadow: 0 1px 2px rgb(17 19 21 / 5%);
            transition: border-color .15s ease, background .15s ease,
                transform .15s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            border-color: #9daf9f;
            background: var(--accent-soft);
            color: #254332;
        }

        [data-baseweb="select"] > div,
        [data-baseweb="base-input"] {
            border-color: #cfd3d8;
            border-radius: 5px;
        }

        [data-testid="stExpander"] {
            background: #f4f8f5;
            border: 1px solid #cbdacf;
            border-left: 3px solid #5d9a72;
            border-radius: 6px;
        }

        [data-testid="stMain"] [data-testid="stExpander"] summary p {
            color: #315d43;
            font-weight: 650;
        }

        [data-testid="stMain"] [data-testid="stExpander"] summary svg {
            color: #3f7655;
        }

        [data-testid="stMain"] [data-testid="stExpander"] [data-baseweb="tag"] {
            background: #dfece3;
            border: 1px solid #bdd2c3;
        }

        [data-testid="stMain"] [data-testid="stExpander"] [data-baseweb="tag"] span {
            color: #28563b;
            font-weight: 600;
        }

        [data-testid="stMain"] [data-testid="stExpander"] [data-baseweb="tag"] svg {
            color: #527660;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: #303b34;
            border: 1px solid #56655a;
            box-shadow: inset 3px 0 0 #5d9a72;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
            color: #f2f5f3;
            font-weight: 650;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
            color: #9bc5a9;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="base-input"] {
            background: #fff;
            border-color: #cbd5cd;
            color: #202521;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-testid="stExpander"] input {
            color: #202521;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"]
        [data-baseweb="select"] div {
            color: #4e5750;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"]
        input::placeholder {
            color: #4e5750;
            opacity: 1;
        }

        [data-testid="stSlider"] [role="slider"] {
            background: var(--accent);
            border-color: #fff;
        }

        [data-testid="stCheckbox"] svg,
        [data-baseweb="select"] svg {
            color: var(--accent);
        }

        [data-testid="stSidebar"] [data-baseweb="tag"] {
            background: #315340;
            border: 1px solid #47715a;
        }

        [data-testid="stSidebar"] [data-baseweb="tag"] span {
            color: #f3f7f4;
        }

        [data-testid="stSidebar"] [data-baseweb="tag"] svg {
            color: #dce7df;
        }

        [data-testid="stAlert"] {
            border-radius: 5px;
        }

        @media (max-width: 800px) {
            [data-testid="stMainBlockContainer"] {
                padding: 1.4rem 1rem 3rem;
            }

            .page-heading {
                display: block;
                margin-bottom: 1rem;
            }

            .page-heading h1 {
                font-size: 1.75rem;
            }

            [data-testid="stHorizontalBlock"] {
                gap: .65rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, description: str) -> None:
    """Render a consistent page title and concise supporting context."""
    st.markdown(
        (
            '<div class="page-heading"><div>'
            f"<h1>{title}</h1><p>{description}</p>"
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def _map_styles(
    styler: pd.io.formats.style.Styler,
    func,
    *,
    subset: list[str] | str,
) -> pd.io.formats.style.Styler:
    """Apply elementwise styles across pandas versions."""
    if hasattr(styler, "map"):
        return styler.map(func, subset=subset)
    return styler.applymap(func, subset=subset)


def _medal_colors(avgs: pd.Series) -> List[str]:
    """Return a list of gold/silver/bronze hex colours by rank."""
    ranks = avgs.rank(method="first", ascending=False, na_option="bottom")
    ranks = ranks.fillna(len(avgs) + 1).astype(int)
    medal_map = {1: "#F6E7A6", 2: "#E6E9E7", 3: "#E8C9AD"}
    return [medal_map.get(r, "") for r in ranks]


def _contrast_color(hex_color: str) -> str:
    """Return black or white depending on the brightness of ``hex_color``."""
    r, g, b = mcolors.to_rgb(hex_color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000" if luminance > 0.5 else "#fff"


def apply_gradient(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Colour numeric columns with a soft green gradient and apply
    custom colouring rules for other columns.
    The top‑3 ``Average`` rows receive medal backgrounds.
    """
    numeric = numeric_cols(df)
    styler = df.style
    if numeric:
        norms = {}
        for col in numeric:
            vals = pd.to_numeric(df[col], errors="coerce")
            vmin, vmax = vals.min(), vals.max()
            if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
                norms[col] = mcolors.Normalize(0, 1)
            else:
                norms[col] = mcolors.Normalize(vmin=vmin, vmax=vmax)

        def _num_style(val: str, norm: mcolors.Normalize) -> str:
            num = pd.to_numeric(val, errors="coerce")
            if np.isnan(num):
                return ""
            ratio = norm(num)
            colour = mcolors.to_hex(SCORE_CMAP(ratio))
            text = _contrast_color(colour)
            return f"background-color:{colour};color:{text}"

        for col in numeric:
            norm = norms[col]
            styler = _map_styles(
                styler,
                lambda v, n=norm: _num_style(v, n),
                subset=[col],
            )

    def _param_column_style(series: pd.Series) -> mcolors.Normalize:
        vals = pd.to_numeric(series.astype(str).str.replace("B", "", regex=False), errors="coerce")
        vmin, vmax = vals.min(), vals.max()
        if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
            return mcolors.Normalize(0, 1)
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    parameter_cmap = mcolors.LinearSegmentedColormap.from_list(
        "soft_parameters",
        ["#f2f4f3", "#dce5df", "#a9bcae"],
    )

    def _param_style(val: str, norm: mcolors.Normalize) -> str:
        num = pd.to_numeric(str(val).replace("B", ""), errors="coerce")
        ratio = 1.0 if np.isnan(num) else norm(num)
        colour = mcolors.to_hex(parameter_cmap(ratio))
        text = _contrast_color(colour)
        return f"background-color:{colour};color:{text}"

    for col in [c for c in ["Parameters", "Active Parameters"] if c in df.columns]:
        norm = _param_column_style(df[col])
        df[col] = df[col].astype(str).str.replace("B", "", regex=False)
        styler = _map_styles(
            styler,
            lambda v: _param_style(v, norm),
            subset=[col],
        )

    # License column colours
    if "License" in df.columns:
        def _license_style(val: str) -> str:
            v = str(val).strip().lower()
            if v == "proprietary":
                colour = "#f2dddd"
            elif v == "mit":
                colour = "#d9eadf"
            else:
                colour = "#e6eee8"
            text = _contrast_color(colour)
            return f"background-color:{colour};color:{text}"

        styler = _map_styles(styler, _license_style, subset=["License"])

    # Highlight specific organization
    if "Organization" in df.columns:
        def _org_style(val: str) -> str:
            if str(val).strip() == "ZharfaTech":
                colour = "#dcece1"
                text = _contrast_color(colour)
                return f"background-color:{colour};color:{text}"
            return ""

        styler = _map_styles(styler, _org_style, subset=["Organization"])

    if "Average" in df.columns:
        avgs = df["Average"].astype(float)
        medals = _medal_colors(avgs)

        def _rowstyles(_: pd.Series) -> List[str]:
            return [
                f"background-color:{m};color:{_contrast_color(m)}" if m else ""
                for m in medals
            ]

        styler = styler.apply(_rowstyles, subset=["Average"], axis=0)
        if "Model" in df.columns:
            styler = styler.apply(_rowstyles, subset=["Model"], axis=0)
        if "Rank" in df.columns:
            styler = styler.apply(_rowstyles, subset=["Rank"], axis=0)

    return styler


def render_styler(
    styler: pd.io.formats.style.Styler,
    column_config: dict | None = None,
) -> None:
    """Display a styled DataFrame in Streamlit with interactive features."""
    st.dataframe(
        styler,
        width="stretch",
        hide_index=True,
        column_config=column_config or {},
    )
