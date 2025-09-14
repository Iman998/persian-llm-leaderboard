"""
🏆 Leaderboard page – overall model ranking.

This module fetches (or builds) `dashboard/leaderboard.csv`,
then displays it with gradient styling and an interactive
bar‑chart “quick view”.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from core.paths import (
    DASHBOARD_CSV,
    DASHBOARD_FA_CSV,
    DASHBOARD_EN_CSV,
    DATASETS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
)
from core.io import load_csv, load_meta, numeric_cols
from core.style import apply_gradient, render_styler


def _build_leaderboard_if_missing(
    board_path: Path,
    lang: str,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> None:
    """
    (Re)generate `dashboard/leaderboard.csv` if it is absent.

    Calls `scripts/build_leaderboard.py` via `subprocess` so
    that any heavy imports inside the script will not slow
    down the Streamlit runtime.
    """
    if board_path.exists():
        return

    with st.spinner("Building leaderboard…"):
        script = Path(__file__).resolve().parents[2] / "scripts" / "build_leaderboard.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--results_dir",
                str(RESULTS_DIR),
                "--datasets_dir",
                str(DATASETS_DIR),
                "--models_dir",
                str(MODELS_DIR),
                "--out",
                str(board_path),
                *( ["--lang", lang] if lang != "all" else [] ),
                *( ["--include", *include] if include else [] ),
                *( ["--exclude", *exclude] if exclude else [] ),
            ],
            capture_output=True,
            text=True,
        )

    if proc.returncode:
        st.error("Failed to build leaderboard CSV")
        st.text(proc.stderr)
        st.stop()


def _render_quick_chart(df: pd.DataFrame) -> None:
    with st.expander("📊 Quick chart"):
        metric = st.selectbox("Metric", numeric_cols(df), 0)
        max_n  = len(df)

        # ⬇︎ guard: only show slider when >1 model
        if max_n > 1:
            page_size = st.slider("Models per view", 1, max_n, min(10, max_n))
        else:
            page_size = 1

        if max_n > page_size:
            start = st.slider("Start index", 0, max_n - page_size, 0)
        else:
            start = 0

        chart_df = df.iloc[start : start + page_size]
        st.altair_chart(
            alt.Chart(chart_df)
            .mark_bar()
            .encode(x=alt.X(metric, type="quantitative"),
                    y=alt.Y("Model", sort="-x")),
            use_container_width=True,
        )


def show() -> None:
    """Streamlit entry‑point for the page (called by bootstrap)."""
    st.title("🏆 Persian‑LLM Leaderboard")

    board_choice = st.sidebar.radio(
        "Language", ["All", "Persian", "English"], key="board_lang"
    )
    board_map = {
        "All": (DASHBOARD_CSV, "all"),
        "Persian": (DASHBOARD_FA_CSV, "fa"),
        "English": (DASHBOARD_EN_CSV, "en"),
    }
    board_path, lang = board_map[board_choice]

    _build_leaderboard_if_missing(
        board_path,
        lang,
        exclude=["translat", "summar", "summary"],
    )
    board_df = load_csv(board_path).sort_values("Average", ascending=False)

    ranks = list(range(1, len(board_df) + 1))
    medals = {1: "\U0001F947", 2: "\U0001F948", 3: "\U0001F949"}
    rank_col = [medals.get(r, str(r)) for r in ranks]
    board_df.insert(0, "Rank", rank_col)

    col_cfg: dict[str, st.column_config.Column] = {}
    if "Language Average" in board_df.columns:
        col_cfg["Language Average"] = st.column_config.NumberColumn(
            "Language Average",
            help="(English Average × 2/3) + (Persian Average × 1/3)",
        )

    desc_map = {}
    for p in DATASETS_DIR.iterdir():
        if p.is_dir():
            meta = load_meta(p.name)
            desc = meta.get("description")
            if desc:
                desc_map[p.name] = desc

    for col in board_df.columns:
        if " (" in col and col.endswith(")"):
            ds = col.split(" (", 1)[0]
            desc = desc_map.get(ds)
            if desc:
                col_cfg[col] = st.column_config.NumberColumn(col, help=desc)

    render_styler(apply_gradient(board_df), column_config=col_cfg or None)
    _render_quick_chart(board_df)

    st.download_button(
        "Download CSV", board_path.read_bytes(), file_name=board_path.name
    )
