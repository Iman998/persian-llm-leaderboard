#!/usr/bin/env python3
"""
Streamlit dashboard for Persian-LLM leaderboard.

• Leaderboard page – global CSV.
• Dataset view     – pick a dataset, then any models to compare:
    • Row outputs   – question, gold, predictions side-by-side
    • Category      – accuracy by any category column, side-by-side
"""

from pathlib import Path
import re
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Persian-LLM Leaderboard", layout="wide")

RESULTS_DIR   = Path("results")
DASHBOARD_CSV = Path("dashboard/leaderboard.csv")
MODELS_DIR    = Path("models")          # to detect valid model names

# ───────────── helpers ─────────────────────────────────────────────── #

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def num_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def gradient(df: pd.DataFrame):
    try:
        import matplotlib  # noqa
        return (
            df.style.background_gradient(axis=0, cmap="Greens", subset=num_cols(df))
              .highlight_max(axis=0, subset=["Average"], color="#ffd60a")
        )
    except ImportError:
        return df

# ---------- build regex for parsing filenames ------------------------ #
model_names = sorted([p.stem for p in MODELS_DIR.glob("*.yaml")], key=len, reverse=True)
models_alt  = "|".join(map(re.escape, model_names))
FILE_RE = re.compile(
    rf"^(?P<dataset>.+?)_(?P<model>{models_alt})(?:_(?P<suffix>.+?))?\.csv$"
)

def parse_file(p: Path) -> Tuple[str, str, str]:
    """
    results/<dataset>_<model>[ _<suffix> ].csv
    suffix == 'raw' or category name or ''
    """
    m = FILE_RE.match(p.name)
    if not m:
        raise ValueError(f"Unexpected filename {p.name}")
    ds, mdl, suff = m.group("dataset", "model", "suffix")
    return ds, mdl, suff or ""

# ---------- scan results directory ----------------------------------- #
main_map: Dict[Tuple[str, str], Path] = {}          # (ds, model) → main.csv
raw_map:  Dict[Tuple[str, str], Path] = {}          # (ds, model) → raw.csv
cat_map:  Dict[Tuple[str, str, str], Path] = {}     # (ds, model, cat)

for p in RESULTS_DIR.glob("*.csv"):
    ds, mdl, suf = parse_file(p)
    if suf == "raw":
        raw_map[(ds, mdl)] = p
    elif suf:  # category
        cat_map[(ds, mdl, suf)] = p
    else:
        main_map[(ds, mdl)] = p

datasets = sorted({k[0] for k in main_map})
if not datasets:
    st.error("No result CSVs in ./results – run evaluations first.")
    st.stop()

# ───────────── sidebar navigation ──────────────────────────────────── #
page = st.sidebar.radio("📑 Page", ["Leaderboard", "Dataset view"])

# ───────────── Leaderboard page ─────────────────────────────────────── #
if page == "Leaderboard":
    st.title("🏆 Persian-LLM Leaderboard")
    if not DASHBOARD_CSV.exists():
        st.warning("Leaderboard CSV not found – build it first.")
        st.stop()

    board_df = load_csv(DASHBOARD_CSV).sort_values("Average", ascending=False)
    st.dataframe(gradient(board_df), use_container_width=True, height=600)

    with st.expander("📊 Quick chart"):
        metric = st.selectbox("Metric", num_cols(board_df), 0)
        st.bar_chart(board_df[["Model", metric]].set_index("Model"))

    st.download_button("Download CSV", DASHBOARD_CSV.read_bytes(), "leaderboard.csv")
    st.stop()

# ───────────── Dataset view page ─────────────────────────────────────── #
ds_sel = st.sidebar.selectbox("📂 Dataset", datasets)
models_in_ds = sorted({k[1] for k in main_map if k[0] == ds_sel})
models_sel   = st.sidebar.multiselect("🧠 Models", models_in_ds, default=models_in_ds[:1])

st.title(f"{ds_sel} – Detailed view")

if not models_sel:
    st.info("Select at least one model.")
    st.stop()

tabs = st.tabs(["📄 Row outputs", "📊 Category scores"])

# ---------- Row outputs ------------------------------------------------ #
with tabs[0]:
    merged = None
    for m in models_sel:
        raw_file = raw_map.get((ds_sel, m))
        if not raw_file:
            st.warning(f"No raw file for {m}")
            continue
        df = load_csv(raw_file)[["Question Body", "Key", "pred"]]
        df = df.rename(columns={"Key": "Gold", "pred": m})
        merged = df if merged is None else merged.merge(df[[m]], left_index=True, right_index=True)

    if merged is None:
        st.warning("No raw files for selected models.")
    else:
        st.dataframe(merged.head(500), use_container_width=True, height=400)
        st.download_button(
            "Download comparison CSV",
            merged.to_csv(index=False).encode(),
            file_name=f"{ds_sel}_rows_compare.csv",
        )

# ---------- Category scores ------------------------------------------- #
with tabs[1]:
    cats = sorted({key[2] for key in cat_map if key[0] == ds_sel and key[1] in models_sel})
    if not cats:
        st.info("No category CSVs available.")
        st.stop()

    cat_sel = st.selectbox("Category column", cats)
    frames = []
    for m in models_sel:
        p = cat_map.get((ds_sel, m, cat_sel))
        if p:
            df_c = load_csv(p).rename(columns={"Accuracy": m}).set_index(cat_sel)
            frames.append(df_c)

    if frames:
        comp = pd.concat(frames, axis=1)
        st.dataframe(comp, use_container_width=True)
        st.bar_chart(comp)
        st.download_button(
            "Download category comparison",
            comp.reset_index().to_csv(index=False).encode(),
            file_name=f"{ds_sel}_{cat_sel}_compare.csv",
        )
    else:
        st.warning("Category CSVs not found for selected models.")
