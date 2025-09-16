"""🔤 Translation leaderboard page."""

from __future__ import annotations

import streamlit as st

from core.paths import (
    TRANSLATION_CSV,
    DATASETS_DIR,
)
from core.io import load_csv, load_meta
from core.style import apply_gradient, render_styler
from .leaderboard import _build_leaderboard_if_missing, _render_quick_chart


def show() -> None:
    """Render the translation leaderboard."""
    st.title("🔤 Translation Leaderboard")

    board_path = TRANSLATION_CSV
    _build_leaderboard_if_missing(
        board_path,
        "all",
        board="translation",
        include=["translat"],
        exclude=["translation_quality"],
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

