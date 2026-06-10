"""🔤 Translation leaderboard page."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from core.paths import (
    TRANSLATION_CSV,
)
from core.io import load_csv
from core.style import SCORE_CMAP, page_header, render_styler
from .leaderboard import _build_leaderboard_if_missing, _render_quick_chart

_METRIC_COLUMN = re.compile(r"^(?P<dataset>.+) \((?P<metric>[^()]+)\)$")


def _split_metric_column(column: object) -> tuple[str, str] | None:
    """Split a flat ``dataset (metric)`` leaderboard column."""
    match = _METRIC_COLUMN.fullmatch(str(column))
    if not match:
        return None
    return match.group("dataset"), match.group("metric")


def _group_translation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with datasets as headers and metrics as subheaders."""
    grouped = df.copy()
    grouped.columns = pd.MultiIndex.from_tuples(
        [
            _split_metric_column(column) or ("", str(column))
            for column in grouped.columns
        ]
    )
    return grouped


def _add_dataset_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add an average metric column for every translation dataset."""
    result = df.copy()
    dataset_metrics: dict[str, list[str]] = {}
    for column in df.columns:
        metric_column = _split_metric_column(column)
        if metric_column:
            dataset, metric = metric_column
            if metric != "Average":
                dataset_metrics.setdefault(dataset, []).append(str(column))

    for dataset, columns in dataset_metrics.items():
        average_column = f"{dataset} (Average)"
        values = result[columns].apply(pd.to_numeric, errors="coerce")
        average = values.mean(axis=1).round(5)
        insert_at = max(result.columns.get_loc(column) for column in columns) + 1
        result.insert(insert_at, average_column, average)

    return result


def _style_translation_table(
    df: pd.DataFrame,
) -> pd.io.formats.style.Styler:
    """Apply score gradients while preserving grouped column headers."""
    metric_columns = [
        column
        for column in df.columns
        if isinstance(column, tuple) and bool(column[0])
    ]
    styler = df.style
    if metric_columns:
        styler = styler.background_gradient(
            cmap=SCORE_CMAP,
            axis=0,
            subset=metric_columns,
        )
    return styler.format(precision=5, na_rep="")


def show() -> None:
    """Render the translation leaderboard."""
    board_path = TRANSLATION_CSV
    _build_leaderboard_if_missing(
        board_path,
        "all",
        board="translation",
        include=["translat"],
        exclude=["translation_quality"],
    )
    board_df = load_csv(board_path).sort_values("Average", ascending=False)
    page_header(
        "Translation Leaderboard",
        "Inspect translation quality across automatic metrics including BLEU, "
        "METEOR, chrF, and translation error rate.",
    )

    metric_cols = [
        column for column in board_df.columns if _split_metric_column(column)
    ]
    summary_cols = st.columns(3)
    summary_cols[0].metric("Models ranked", f"{len(board_df):,}")
    summary_cols[1].metric("Metrics", f"{len(metric_cols):,}")
    summary_cols[2].metric(
        "Top model",
        str(board_df.iloc[0]["Model"]) if not board_df.empty else "—",
    )
    st.write("")

    ranks = list(range(1, len(board_df) + 1))
    medals = {1: "\U0001F947", 2: "\U0001F948", 3: "\U0001F949"}
    rank_col = [medals.get(r, str(r)) for r in ranks]
    board_df.insert(0, "Rank", rank_col)

    board_df = _add_dataset_averages(board_df)
    grouped_df = _group_translation_columns(board_df)
    render_styler(_style_translation_table(grouped_df))
    _render_quick_chart(board_df)

    st.download_button(
        "↓  Download CSV",
        grouped_df.to_csv(index=False).encode(),
        file_name=board_path.name,
    )
