#!/usr/bin/env python3
"""Streamlit dashboard for Persian‑LLM leaderboard (meta‑driven + raw toggle).

Add‑ons (2025‑06‑03)
--------------------
* **Raw answers toggle** – checkbox lets you include/exclude the full model
  output column (``raw``) in the row‑comparison table.
* **Category filters** – when a dataset defines ``category_cols`` in its
  ``meta.yaml`` you can interactively filter the row table by any combination
  of category values. These filters now also apply to the **Category scores**
  tab so breakdowns reflect the selected subset.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt
from leaderboard_lib.data_utils import _norm

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
        max_models = len(board_df)
        if max_models > 1:
            page_size = st.slider(
                "Models per view",
                1,
                max_models,
                min(10, max_models),
            )
        else:
            page_size = 1

        if max_models > page_size:
            start = st.slider("Start index", 0, max_models - page_size, 0)
        else:
            start = 0
        chart_df = board_df.iloc[start:start + page_size]
        mark_opts = {"color": "green"} if max_models == 1 else {}
        chart = (
            alt.Chart(chart_df)
            .mark_bar(**mark_opts)
            .encode(
                x=alt.X(metric, type="quantitative"),
                y=alt.Y("Model", sort="-x"),
            )
        )
        st.altair_chart(chart, use_container_width=True)

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
            page_size = st.selectbox(
                "Rows per page", [50, 100, 200], key="rows_page_size"
            )
            total_rows = len(merged)
            total_pages = max(1, (total_rows + page_size - 1) // page_size)
            page = st.number_input(
                "Page", 1, total_pages, 1, key="rows_page", format="%d"
            )
            start = (page - 1) * page_size
            end = start + page_size
            st.dataframe(
                gradient(merged.iloc[start:end]),
                use_container_width=True,
                height=400,
            )
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
            df_c = None
            if cat_filters:
                raw_file = raw_map.get((ds_sel, m))
                if not raw_file:
                    st.warning(f"No raw file for {m}; cannot apply filters.")
                    continue
                df = load_csv(raw_file)
                for col, allowed in cat_filters.items():
                    if col in df.columns:
                        df = df[df[col].isin(allowed)]
                if cat_sel not in df.columns or meta["answer_col"] not in df.columns:
                    st.warning(f"Columns missing in {raw_file.name}; skipped.")
                    continue
                def _acc(g: pd.DataFrame) -> float:
                    return (
                        g["pred"].map(_norm) == g[meta["answer_col"]].map(_norm)
                    ).mean() * 100
                df_c = (
                    df.groupby(cat_sel, dropna=False)
                    .apply(_acc)
                    .reset_index(name=m)
                    .set_index(cat_sel)
                )
            else:
                p = cat_map.get((ds_sel, m, cat_sel))
                if p:
                    df_c = load_csv(p)
                    if cat_sel not in df_c.columns:
                        st.warning(f"{cat_sel} column missing in {p.name}; skipped.")
                        continue
                    df_c = df_c.rename(columns={"Accuracy": m}).set_index(cat_sel)
            if df_c is not None:
                frames.append(df_c)

        if frames:
            comp = pd.concat(frames, axis=1)
            page_size = st.selectbox(
                "Rows per page", [50, 100, 200], key="cat_page_size"
            )
            total_rows = len(comp)
            total_pages = max(1, (total_rows + page_size - 1) // page_size)
            page = st.number_input(
                "Page", 1, total_pages, 1, key="cat_page", format="%d"
            )
            start = (page - 1) * page_size
            end = start + page_size
            st.dataframe(
                gradient(comp.iloc[start:end]), use_container_width=True, height=400
            )
            st.bar_chart(comp, use_container_width=True)
            st.download_button(
                "Download category comparison",
                comp.reset_index().to_csv(index=False).encode(),
                file_name=f"{ds_sel}_{cat_sel}_compare.csv",
            )
        else:
            st.warning("Category CSVs not found for selected models.")

elif page == "LLM Judge":
    st.title("🤖 LLM Judge")

    judge_datasets = [
        d
        for d in datasets
        if "summarization" in d.lower() or "translation" in d.lower()
    ]

    if not judge_datasets:
        st.info("No LLM-judge datasets available.")
        st.stop()

    ds_sel = st.sidebar.selectbox("📂 Dataset", judge_datasets)

    models = sorted({m for (ds, m) in main_map if ds == ds_sel})
    if not models:
        st.info("No model results for this dataset.")
        st.stop()

    frames = []
    metric_names: set[str] = set()
    for m in models:
        p = main_map.get((ds_sel, m))
        if not p:
            continue
        df = load_csv(p)
        df_num = df.apply(pd.to_numeric, errors="coerce")
        cols = [
            c
            for c in df_num.columns
            if (
                c.lower().startswith("score")
                or c.lower().startswith("pred")
                or "judge" in c.lower()
            )
            and not df_num[c].isna().all()
        ]
        if not cols:
            cols = [c for c in df_num.columns if not df_num[c].isna().all()]
        if not cols:
            continue

        scores = {c: df_num[c].mean() for c in cols}
        if len(scores) > 1:
            scores["Average"] = sum(scores.values()) / len(scores)
        frames.append({"Model": m, **scores})
        metric_names.update(scores.keys())

    if not frames:
        st.info("No judge scores found.")
        st.stop()

    metric_order = ["Model"] + [c for c in sorted(metric_names) if c != "Model"]
    df_table = pd.DataFrame(frames)[metric_order]
    st.dataframe(gradient(df_table), use_container_width=True, height=500)
    st.download_button(
        "Download CSV",
        df_table.to_csv(index=False).encode(),
        file_name=f"{ds_sel}_judge_scores.csv",
    )

    # ------------------- Optional per-category breakdown -------------------- #
    cat_names = sorted({k[2] for k in cat_map if k[0] == ds_sel and k[1] in models})
    if cat_names:
        st.subheader("Category breakdown")
        cat_sel = st.selectbox("Category column", cat_names)

        frames = []
        for m in models:
            p = cat_map.get((ds_sel, m, cat_sel))
            if not p:
                continue
            df_c = load_csv(p)
            if cat_sel not in df_c.columns:
                st.warning(f"{cat_sel} column missing in {p.name}; skipped.")
                continue
            metric_cols = [c for c in df_c.columns if c != cat_sel]
            metric_col = metric_cols[0] if metric_cols else None
            if metric_col is None:
                continue
            df_c = df_c.rename(columns={metric_col: m}).set_index(cat_sel)
            frames.append(df_c)

        if frames:
            comp_df = pd.concat(frames, axis=1)
            page_size = st.selectbox(
                "Rows per page", [50, 100, 200], key="judge_cat_page_size"
            )
            total_rows = len(comp_df)
            total_pages = max(1, (total_rows + page_size - 1) // page_size)
            page = st.number_input(
                "Page", 1, total_pages, 1, key="judge_cat_page", format="%d"
            )
            start = (page - 1) * page_size
            end = start + page_size
            st.dataframe(
                gradient(comp_df.iloc[start:end]), use_container_width=True, height=400
            )
            st.bar_chart(comp_df, use_container_width=True)
            st.download_button(
                "Download category comparison",
                comp_df.reset_index().to_csv(index=False).encode(),
                file_name=f"{ds_sel}_{cat_sel}_judge_compare.csv",
            )
        else:
            st.warning("Category CSVs not found for selected models.")
