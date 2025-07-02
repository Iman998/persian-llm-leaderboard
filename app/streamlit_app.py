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
# Ensure `ROOT_DIR` is import-searchable
# ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]          # …/app/..
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Now regular absolute imports work even when run as a script
from app.views import leaderboard, dataset_view, llm_judge  # noqa: E402

st.set_page_config(page_title="Persian-LLM Leaderboard", layout="wide")

PAGE_MAP = {
    "🏆 Leaderboard":  leaderboard.show,
    "📂 Dataset view": dataset_view.show,
    "🤖 LLM Judge":    llm_judge.show,
}

choice = st.sidebar.radio("📑 Page", list(PAGE_MAP))
PAGE_MAP[choice]()
