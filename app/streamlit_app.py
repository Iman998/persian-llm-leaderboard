#!/usr/bin/env python3
"""
Bootstrapper for the Persian‑LLM Streamlit dashboard.

This file **only** registers pages and sets global Streamlit
configuration.  All page‑specific code lives in `app/pages/`,
and shared non‑UI helpers live in `app/core/`.

Keeping this file tiny guarantees that the Streamlit runtime
does not need to re‑run heavy imports when the user switches
between pages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ──────────────────────────────────────────────────────────
# Ensure project and app packages are import-searchable
# ──────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
for search_path in (ROOT_DIR, APP_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

# Now regular absolute imports work even when run as a script
from app.views import (
    leaderboard,
    battle,
    league,
    dataset_view,
    llm_judge,
    translation_board,
    summarization_board,
)  # noqa: E402
from app.core.style import inject_global_styles  # noqa: E402

st.set_page_config(
    page_title="Persian LLM Leaderboard",
    page_icon=":material/leaderboard:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_styles()

st.sidebar.markdown(
    """
    <div class="app-brand">
        <div class="app-brand__mark">PL</div>
        <div class="app-brand__name">Persian LLM<br>Leaderboard</div>
        <div class="app-brand__meta">Open model evaluation workspace</div>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGE_MAP = {
    "Leaderboard": leaderboard.show,
    "Battle": battle.show,
    "League": league.show,
    "Translation": translation_board.show,
    "Summarization": summarization_board.show,
    "Dataset explorer": dataset_view.show,
    "LLM judge": llm_judge.show,
}

choice = st.sidebar.radio("Workspace", list(PAGE_MAP), label_visibility="collapsed")
st.sidebar.divider()
PAGE_MAP[choice]()
