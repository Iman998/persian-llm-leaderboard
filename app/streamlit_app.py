#!/usr/bin/env python3
"""Streamlit dashboard for Persian‑LLM leaderboard (meta‑driven + raw toggle).

Add‑ons (2025‑06‑03)
--------------------
* **Raw answers toggle** – checkbox lets you include/exclude the full model
  output column (``raw``) in the row‑comparison table.
* **Category filters** – when a dataset defines ``category_cols`` in its
  ``meta.yaml`` you can interactively filter the row table by any combination
  of category values.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import (
    DASHBOARD_CSV,
    DATASETS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    gradient,
    load_csv,
    load_meta,
    numeric_cols,
    scan_result_maps,
)

st.set_page_config(page_title="Persian‑LLM Leaderboard", layout="wide")

datasets, main_map, raw_map, cat_map = scan_result_maps()
if not datasets:
    st.error("No result CSVs in ./results – run evaluations first.")
    st.stop()

if not DASHBOARD_CSV.exists():
    with st.spinner("Building leaderboard…"):
        script = Path(__file__).resolve().parents[1] / "scripts" / "build_leaderboard.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--results_dir", str(RESULTS_DIR),
             "--datasets_dir", str(DATASETS_DIR), "--models_dir", str(MODELS_DIR),
             "--out", str(DASHBOARD_CSV)],
            capture_output=True, text=True,
        )
    if proc.returncode != 0:
        st.error("Failed to build leaderboard CSV")
        st.text(proc.stderr)
        st.stop()

# ───────────── sidebar navigation ───────────────────────────────────── #
page = st.sidebar.radio("📑 Page", ["Leaderboard", "Dataset view", "LLM Judge"])

# ───────────── Leaderboard page ─────────────────────────────────────── #
if page == "Leaderboard":
    st.title("🏆 Persian‑LLM Leaderboard")
    if not DASHBOARD_CSV.exists():
        st.warning("Leaderboard CSV not found – build it first.")
        st.stop()

    board_df = load_csv(DASHBOARD_CSV).sort_values("Average", ascending=False)
    st.dataframe(gradient(board_df), use_container_width=True, height=600)

    with st.expander("📊 Quick chart"):
        metric = st.selectbox("Metric", numeric_cols(board_df), 0)
        st.bar_chart(board_df[["Model", metric]].set_index("Model"))

    st.download_button("Download CSV", DASHBOARD_CSV.read_bytes(), "leaderboard.csv")
    st.stop()

# ───────────── Dataset view page ────────────────────────────────────── #
elif page == "Dataset view":
    ds_sel = st.sidebar.selectbox("📂 Dataset", datasets)
    meta = load_meta(ds_sel)
    models_in_ds = sorted({k[1] for k in main_map if k[0] == ds_sel})
    models_sel   = st.sidebar.multiselect("🧠 Models", models_in_ds, default=models_in_ds[:1])

    st.title(f"{ds_sel} – Detailed view")

    if not models_sel:
        st.info("Select at least one model.")
        st.stop()

# Category filters ------------------------------------------------------ #
    cat_filters = {}
    if meta["category_cols"]:
        with st.sidebar.expander("🔍 Category filters", expanded=False):
            for col in meta["category_cols"]:
                raw_file_any = raw_map.get((ds_sel, models_sel[0]))  # sample file
                if raw_file_any is not None:
                    df_sample = load_csv(raw_file_any)
                    if col not in df_sample.columns:
                        st.warning(f"Column {col} not found in {raw_file_any.name}; skipping filter.")
                        continue
                    col_vals = df_sample[col].dropna().unique()
                    sel = st.multiselect(col, sorted(col_vals))
                    if sel:
                        cat_filters[col] = set(sel)

# Raw toggle ------------------------------------------------------------ #
    show_raw = st.sidebar.checkbox("Show raw model output", value=False)

# Build tabs ------------------------------------------------------------ #
    row_tab, cat_tab = st.tabs(["📄 Row outputs", "📊 Category scores"])

    question_aliases = {meta["question_col"], "Question Body", "question"}
    answer_aliases   = {meta["answer_col"], "Gold", "Key", "answer"}

# ---------- Row outputs ------------------------------------------------ #
    with row_tab:
        merged = None
        for m in models_sel:
            raw_file = raw_map.get((ds_sel, m))
            if not raw_file:
                st.warning(f"No raw file for {m}")
                continue
            df = load_csv(raw_file)

            # Apply category filters first
            for col, allowed in cat_filters.items():
                if col in df.columns:
                    df = df[df[col].isin(allowed)]

            q_present = next((c for c in question_aliases if c in df.columns), None)
            g_present = next((c for c in answer_aliases if c in df.columns), None)
            if q_present is None or g_present is None or "pred" not in df.columns:
                st.warning(f"Columns missing in {raw_file.name}; skipped.")
                continue

            keep_cols = [q_present, g_present, "pred"]
            if show_raw and "raw" in df.columns:
                keep_cols.append("raw")
            df = df[keep_cols].rename(columns={
                q_present: "Question",
                g_present: "Gold",
                "pred": m,
                "raw": f"{m}‑raw" if show_raw else "raw",
            })
            merged = df if merged is None else merged.merge(
                df.drop(columns=["Question", "Gold" if "Gold" in df.columns else g_present]),
                left_index=True,
                right_index=True,
            )

        if merged is None:
            st.warning("No compatible raw files for selected models.")
        else:
            st.dataframe(merged.head(500), use_container_width=True, height=400)
            st.download_button(
                "Download comparison CSV",
                merged.to_csv(index=False).encode(),
                file_name=f"{ds_sel}_rows_compare.csv",
            )

# ---------- Category scores ------------------------------------------- #
    with cat_tab:
        cats = sorted({k[2] for k in cat_map if k[0] == ds_sel and k[1] in models_sel})
        if not cats:
            st.info("No category CSVs available.")
            st.stop()

        cat_sel = st.selectbox("Category column", cats)
        frames = []
        for m in models_sel:
            p = cat_map.get((ds_sel, m, cat_sel))
            if p:
                df_c = load_csv(p)
                if cat_sel not in df_c.columns:
                    st.warning(f"{cat_sel} column missing in {p.name}; skipped.")
                    continue
                df_c = df_c.rename(columns={"Accuracy": m}).set_index(cat_sel)
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

elif page == "LLM Judge":
    st.title("🤖 LLM Judge")
    st.info("This section is under construction.")
