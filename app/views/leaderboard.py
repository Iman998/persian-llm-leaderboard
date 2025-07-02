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
import streamlit as st

from core.paths import DASHBOARD_CSV, DATASETS_DIR, MODELS_DIR, RESULTS_DIR
from core.io import load_csv, numeric_cols
from core.style import apply_gradient


def _build_leaderboard_if_missing() -> None:
    """
    (Re)generate `dashboard/leaderboard.csv` if it is absent.

    Calls `scripts/build_leaderboard.py` via `subprocess` so
    that any heavy imports inside the script will not slow
    down the Streamlit runtime.
    """
    if DASHBOARD_CSV.exists():
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
                str(DASHBOARD_CSV),
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

    _build_leaderboard_if_missing()
    board_df = load_csv(DASHBOARD_CSV).sort_values("Average", ascending=False)

    st.dataframe(apply_gradient(board_df), use_container_width=True, height=600)
    _render_quick_chart(board_df)

    st.download_button(
        "Download CSV", DASHBOARD_CSV.read_bytes(), file_name="leaderboard.csv"
    )
