"""Pairwise model battle dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from core.io import load_csv
from core.paths import BATTLE_CSV, MODELS_DIR, RESULTS_DIR
from core.style import apply_gradient, page_header, render_styler


def _build_battle_board_if_missing() -> None:
    if BATTLE_CSV.exists():
        return
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_battle_board.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--results_dir",
            str(RESULTS_DIR),
            "--models_dir",
            str(MODELS_DIR),
            "--out",
            str(BATTLE_CSV),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        st.error("Failed to build Battle board")
        st.text(proc.stderr)
        st.stop()


def _model_rate_table(row: pd.Series) -> pd.DataFrame:
    """Return chart-ready win/loss/tie rates for both battle participants."""
    return pd.DataFrame(
        [
            {
                "Model": row["Model 1"],
                "Win Rate": row["Model 1 Win Rate"],
                "Loss Rate": row["Model 1 Loss Rate"],
                "Equal Rate": row["Equal Rate"],
            },
            {
                "Model": row["Model 2"],
                "Win Rate": row["Model 2 Win Rate"],
                "Loss Rate": row["Model 2 Loss Rate"],
                "Equal Rate": row["Equal Rate"],
            },
        ]
    )


def _battle_chart(rate_table: pd.DataFrame) -> alt.Chart:
    chart_df = rate_table.melt(
        id_vars="Model",
        var_name="Outcome",
        value_name="Rate",
    )
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "Rate:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            y=alt.Y("Model:N", title=None),
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(
                    domain=["Win Rate", "Loss Rate", "Equal Rate"],
                    range=["#23885a", "#c75d5d", "#9a8b45"],
                ),
            ),
            yOffset="Outcome:N",
            tooltip=[
                "Model:N",
                "Outcome:N",
                alt.Tooltip("Rate:Q", format=".1%"),
            ],
        )
        .properties(height=max(220, 70 * len(rate_table)))
    )


def show() -> None:
    """Render the pairwise model Battle page."""
    _build_battle_board_if_missing()
    if not BATTLE_CSV.exists():
        st.info("No battle results are available yet.")
        return

    board = load_csv(BATTLE_CSV)
    if board.empty:
        st.info("No battle results are available yet.")
        return

    datasets = sorted(board["Dataset"].dropna().astype(str).unique())
    dataset = st.sidebar.selectbox("Dataset", datasets, key="battle_dataset")
    dataset_board = board[board["Dataset"].astype(str) == dataset].copy()
    dataset_board["Matchup"] = (
        dataset_board["Model 1"].astype(str)
        + " vs "
        + dataset_board["Model 2"].astype(str)
    )
    matchup = st.sidebar.selectbox(
        "Matchup",
        dataset_board["Matchup"].tolist(),
        key="battle_matchup",
    )
    selected = dataset_board[dataset_board["Matchup"] == matchup].iloc[0]
    rate_table = _model_rate_table(selected)

    page_header(
        "Battle",
        "Head-to-head model comparisons scored by an independent LLM judge. "
        "Candidate order is counterbalanced to reduce position bias.",
    )
    summary = st.columns(4)
    summary[0].metric("Dataset", dataset)
    summary[1].metric("Evaluated rows", f"{int(selected['Evaluated Rows']):,}")
    summary[2].metric("Judge", str(selected["Judge"]))
    summary[3].metric("Equal rate", f"{selected['Equal Rate']:.1%}")
    st.write("")

    render_styler(apply_gradient(rate_table).format(precision=4))
    st.altair_chart(_battle_chart(rate_table), width="stretch")

    st.subheader("All matchups")
    visible = board.copy()
    rate_columns = [column for column in visible if column.endswith("Rate")]
    render_styler(
        apply_gradient(visible).format(
            {column: "{:.2%}" for column in rate_columns}
        )
    )
    st.download_button(
        "Download Battle CSV",
        board.to_csv(index=False).encode(),
        file_name=BATTLE_CSV.name,
    )
