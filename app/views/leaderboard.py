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
from core.style import apply_gradient, page_header, render_styler


def _filter_leaderboard(
    df: pd.DataFrame,
    *,
    model_query: str = "",
    selected_models: list[str] | None = None,
    selected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return the visible leaderboard rows and columns."""
    visible = df.copy()
    query = model_query.strip()
    if query:
        visible = visible[
            visible["Model"].astype(str).str.contains(query, case=False, na=False)
        ]

    if selected_models:
        visible = visible[visible["Model"].isin(selected_models)]

    if selected_columns:
        fixed_columns = [column for column in ("Rank", "Model") if column in visible]
        columns = fixed_columns + [
            column
            for column in selected_columns
            if column in visible and column not in fixed_columns
        ]
        visible = visible[columns]

    return visible


def _leaderboard_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Render compact controls for choosing visible models and columns."""
    with st.expander("Customize leaderboard"):
        search_col, model_col, column_col = st.columns([1, 1.4, 1.4])
        with search_col:
            model_query = st.text_input(
                "Find model",
                placeholder="Search by model name",
                key="leaderboard_model_query",
            )
        with model_col:
            selected_models = st.multiselect(
                "Models",
                options=df["Model"].astype(str).tolist(),
                placeholder="All models",
                key="leaderboard_models",
            )
        with column_col:
            selectable_columns = [
                column for column in df.columns if column not in {"Rank", "Model"}
            ]
            selected_columns = st.multiselect(
                "Columns",
                options=selectable_columns,
                placeholder="All columns",
                key="leaderboard_columns",
            )

    return _filter_leaderboard(
        df,
        model_query=model_query,
        selected_models=selected_models,
        selected_columns=selected_columns,
    )


def _build_leaderboard_if_missing(
    board_path: Path,
    lang: str,
    board: str = "leaderboard",
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
        include_args = ["--include", *include] if include else []
        exclude_args = ["--exclude", *exclude] if exclude else []
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
                "--board",
                board,
                *include_args,
                *exclude_args,
                *( ["--lang", lang] if lang != "all" else [] ),
            ],
            capture_output=True,
            text=True,
        )

    if proc.returncode:
        st.error("Failed to build leaderboard CSV")
        st.text(proc.stderr)
        st.stop()

    if not board_path.exists():
        st.info("No results found for this leaderboard yet.")
        st.stop()


def _render_quick_chart(df: pd.DataFrame) -> None:
    metrics = numeric_cols(df)
    if not metrics:
        return

    with st.expander("Compare model scores", expanded=True):
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

        chart_df = df.iloc[start : start + page_size].copy()
        chart_df["Display score"] = pd.to_numeric(chart_df[metric], errors="coerce")
        st.altair_chart(
            alt.Chart(chart_df)
            .mark_bar(color="#23885a", cornerRadiusEnd=3, size=18)
            .encode(
                x=alt.X(
                    "Display score:Q",
                    title=metric,
                    axis=alt.Axis(gridColor="#e6e8eb", labelColor="#6b7077"),
                ),
                y=alt.Y(
                    "Model:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(labelLimit=260, labelColor="#202521"),
                ),
                tooltip=[
                    alt.Tooltip("Model:N"),
                    alt.Tooltip("Display score:Q", title=metric, format=".4f"),
                ],
            )
            .properties(height=max(220, 34 * len(chart_df))),
            width="stretch",
        )


def show() -> None:
    """Streamlit entry‑point for the page (called by bootstrap)."""
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
    )
    board_df = load_csv(board_path).sort_values("Average", ascending=False)
    page_header(
        "Persian LLM Leaderboard",
        "Compare model quality across Persian and English benchmarks, with "
        "transparent dataset-level scores and model metadata.",
    )

    summary_cols = st.columns([1, 1.6, 1, 1])
    summary_cols[0].metric("Models ranked", f"{len(board_df):,}")
    summary_cols[1].metric(
        "Top model",
        str(board_df.iloc[0]["Model"]) if not board_df.empty else "—",
    )
    summary_cols[2].metric(
        "Best average",
        f"{board_df['Average'].max():.3f}" if not board_df.empty else "—",
    )
    summary_cols[3].metric("View", board_choice)
    st.write("")

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

    visible_df = _leaderboard_controls(board_df)
    st.caption(
        f"Showing {len(visible_df):,} of {len(board_df):,} models and "
        f"{len(visible_df.columns):,} columns."
    )

    if visible_df.empty:
        st.info("No models match the current selection.")
    else:
        visible_config = {
            column: config
            for column, config in col_cfg.items()
            if column in visible_df.columns
        }
        render_styler(
            apply_gradient(visible_df),
            column_config=visible_config or None,
        )
        _render_quick_chart(visible_df)

    st.download_button(
        "↓  Download visible CSV",
        visible_df.to_csv(index=False).encode(),
        file_name=f"visible_{board_path.name}",
    )
