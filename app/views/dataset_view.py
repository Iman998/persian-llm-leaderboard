"""
📂 Dataset‑view page – row‑level inspection and per‑category scores.

Allows users to:
* pick one dataset
* select one or more models
* optionally filter rows by category columns
* search rows by keywords in the question text
* toggle raw model outputs
* view accuracy breakdowns per category
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from leaderboard_lib.data_utils import _norm  # external helper for normalisation

from core.io import load_csv, load_meta
from core.parser import scan_result_maps
from core.style import apply_gradient, render_styler

# ──────────────────────────────────────────────────────────────
# Cached result maps – shared across pages in the same session
# ──────────────────────────────────────────────────────────────
DATASETS, MAIN_MAP, RAW_MAP, CAT_MAP = scan_result_maps()


def _filter_by_categories(df: pd.DataFrame, filters: Dict[str, Set[str]]) -> pd.DataFrame:
    """
    Return `df` containing **only** rows where each `col` value is
    inside `filters[col]`.  Columns missing in the dataframe are ignored.
    """
    for col, allowed in filters.items():
        if col in df.columns:
            df = df[df[col].isin(allowed)]
    return df


# ------------------------------------------------------------------ #
# Row‑table helpers
# ------------------------------------------------------------------ #
def _collect_row_tables(
    ds: str,
    models: List[str],
    meta: Dict[str, object],
    cat_filters: Dict[str, Set[str]],
    show_raw: bool,
    shots: int,
    query: str = "",
) -> Tuple[pd.DataFrame | None, List[str]]:
    """
    Build a merged DataFrame where each model contributes a column
    of predictions.  Returns `(merged_df | None, warnings)`.
    """
    warnings: List[str] = []
    merged: pd.DataFrame | None = None

    question_aliases = {meta["question_col"], "Question Body", "question"}
    answer_aliases = {meta["answer_col"], "Gold", "Key", "answer"}
    choice_cols: List[str] = meta["choice_cols"]

    for m in models:
        raw_file = RAW_MAP.get((ds, m, shots))
        if not raw_file:
            warnings.append(f"No raw CSV for model **{m}**")
            warnings.append(f"Raw CSV not found for **{m}** – hiding raw column.")
            continue

        df = load_csv(raw_file)
        df = _filter_by_categories(df, cat_filters)

        # Identify canonical column names present in the current file
        q_col = next((c for c in question_aliases if c in df.columns), None)
        a_col = next((c for c in answer_aliases if c in df.columns), None)

        if q_col is None or a_col is None or "pred" not in df.columns:
            warnings.append(f"Columns missing in *{raw_file.name}* – skipped.")
            continue

        if query:
            df = df[df[q_col].astype(str).str.contains(query, case=False, na=False)]

        keep_cols = [q_col, a_col] + choice_cols + ["pred"]
        if show_raw and "raw" in df.columns:
            keep_cols.append("raw")

        df = df[keep_cols].rename(
            columns={
                q_col: "Question",
                a_col: "Gold",
                "pred": m,
                "raw": f"{m}-raw" if show_raw else "raw",
            }
        )

        if merged is None:
            merged = df
        else:
            # Keep Question/Gold/choice_cols only once (from the first model)
            merged = merged.merge(
                df.drop(columns=["Question", "Gold"] + choice_cols),
                left_index=True,
                right_index=True,
                how="inner",
            )

    return merged, warnings


# ------------------------------------------------------------------ #
# Category‑score helpers
# ------------------------------------------------------------------ #
def _compute_accuracy(df: pd.DataFrame, answer_col: str) -> float:
    """Return accuracy (%) for a grouped dataframe chunk."""
    return (df["pred"].map(_norm) == df[answer_col].map(_norm)).mean() * 100


def _collect_category_tables(
    ds: str,
    models: List[str],
    meta: Dict[str, object],
    cat_filters: Dict[str, Set[str]],
    cat_sel: str,
    shots: int,
) -> Tuple[pd.DataFrame | None, List[str]]:
    """
    Assemble per‑category accuracy tables for each selected model.
    """
    frames, warnings = [], []

    for m in models:
        df_c = None

        if cat_filters:
            # Need to recompute on filtered rows
            raw_file = RAW_MAP.get((ds, m, shots))
            if not raw_file:
                warnings.append(f"No raw CSV for model **{m}**")
                continue
            df_raw = load_csv(raw_file)
            df_raw = _filter_by_categories(df_raw, cat_filters)

            if cat_sel not in df_raw.columns or meta["answer_col"] not in df_raw.columns:
                warnings.append(f"Missing columns in *{raw_file.name}* – skipped.")
                continue

            df_c = (
                df_raw.groupby(cat_sel, dropna=False)
                .apply(lambda g: _compute_accuracy(g, meta["answer_col"]))
                .reset_index(name=m)
                .set_index(cat_sel)
            )
        else:
            # Load cached per‑category CSV if available
            p = CAT_MAP.get((ds, m, shots, cat_sel))
            if p and cat_sel in load_csv(p).columns:
                df_c = load_csv(p).rename(columns={"Accuracy": m}).set_index(cat_sel)
            else:
                warnings.append(f"No category CSV for model **{m}**")

        if df_c is not None:
            frames.append(df_c)

    if frames:
        return pd.concat(frames, axis=1), warnings
    return None, warnings


# ------------------------------------------------------------------ #
# Streamlit page
# ------------------------------------------------------------------ #
def show() -> None:
    """Entry‑point for the Dataset‑view page."""
    if not DATASETS:
        st.error("⚠️ No result CSVs found in `results/` – run evaluations first.")
        return

    # ───── Sidebar controls ───────────────────────────────────────────
    ds_sel = st.sidebar.selectbox("📂 Dataset", DATASETS, key="dataset_sel")
    meta = load_meta(ds_sel)

    shot_opts = (
        sorted({k[2] for k in MAIN_MAP if len(k) == 3 and k[0] == ds_sel}) or [0]
    )
    shots_sel = st.sidebar.selectbox(
        "🎯 Shots", shot_opts, format_func=lambda s: f"{s}-shot"
    )

    models_in_dataset = sorted({
        k[1]
        for k in MAIN_MAP
        if len(k) == 3 and k[0] == ds_sel and k[2] == shots_sel
    })
    # A dataset-specific key guarantees independent widget state
    models_sel = st.sidebar.multiselect(
        "🧠 Models",
        options=models_in_dataset,
        default=models_in_dataset[:1],
        key=f"models_{ds_sel}",
    )

    search_query = st.sidebar.text_input(
        "🔎 Search question text",
        value="",
        key=f"search_{ds_sel}",
        help="Filter rows by keywords (case-insensitive)",
    )
    
    # Auto-select first model if user switched dataset and nothing is ticked
    if not models_sel and models_in_dataset:
        st.session_state[f"models_{ds_sel}"] = [models_in_dataset[0]]
        models_sel = st.session_state[f"models_{ds_sel}"]

    st.title(f"{ds_sel} – {shots_sel}-shot view")

    if not models_sel:
        st.info("Select at least one model to continue.")
        return

    # Category filters (optional)
    cat_filters: Dict[str, Set[str]] = {}
    if meta["category_cols"]:
        with st.sidebar.expander("🔍 Category filters", expanded=False):
            # Pick a sample raw file just to enumerate possible values
            sample_file = RAW_MAP.get((ds_sel, models_sel[0], shots_sel))
            sample_df = load_csv(sample_file) if sample_file else None

            for col in meta["category_cols"]:
                if sample_df is not None and col in sample_df.columns:
                    vals = sorted(sample_df[col].dropna().unique())
                    sel = st.multiselect(
                    col,
                    sorted(vals),
                    key=f"{col}_{ds_sel}",   # ← unique key per dataset
                )
                    if sel:
                        cat_filters[col] = set(sel)

    # Raw output toggle
    show_raw = st.sidebar.checkbox(
    "Show raw model output", value=False, key=f"show_raw_{ds_sel}"
    )

    # Build UI tabs
    row_tab, cat_tab = st.tabs(["📄 Row outputs", "📊 Category scores"])

    # ---------------------------------------------------------------- #
    # Row outputs tab
    # ---------------------------------------------------------------- #
    with row_tab:
        merged_df, warnings = _collect_row_tables(
            ds_sel,
            models_sel,
            meta,
            cat_filters,
            show_raw,
            shots_sel,
            search_query,
        )
        if show_raw and all((ds_sel, m, shots_sel) not in RAW_MAP for m in models_sel):
            st.info("Raw outputs are not available for the selected model(s).")
        for w in warnings:
            st.warning(w)

        if merged_df is None or merged_df.empty:
            st.warning("No compatible rows found for the selected settings.")
        else:
            page_size = st.selectbox("Rows per page", [50, 100, 200], key="rows_ps")
            total_rows = len(merged_df)
            total_pages = max(1, (total_rows + page_size - 1) // page_size)
            page_num = st.number_input(
                "Page", 1, total_pages, 1, key="rows_page", format="%d"
            )
            start, end = (page_num - 1) * page_size, page_num * page_size

            st.caption(f"{shots_sel}-shot results")
            render_styler(apply_gradient(merged_df.iloc[start:end]))
            st.download_button(
                "Download CSV",
                merged_df.to_csv(index=False).encode(),
                file_name=f"{ds_sel}_{shots_sel}shot_rows_compare.csv",
            )

    # ---------------------------------------------------------------- #
    # Category scores tab
    # ---------------------------------------------------------------- #
    with cat_tab:
        cat_names = sorted({
            k[3]
            for k in CAT_MAP
            if len(k) == 4 and k[0] == ds_sel and k[1] in models_sel and k[2] == shots_sel
        })
        if not cat_names and not meta["category_cols"]:
            st.info("No per‑category CSVs available for this dataset.")
            return

        cat_sel = st.selectbox(
            "Category column",
            cat_names if cat_names else meta["category_cols"],
        )

        comp_df, warnings = _collect_category_tables(
            ds_sel, models_sel, meta, cat_filters, cat_sel, shots_sel
        )
        for w in warnings:
            st.warning(w)

        if comp_df is None:
            st.warning("Category data not found for the selected configuration.")
            return

        # Paginated table
        page_size = st.selectbox("Rows per page", [50, 100, 200], key="cat_ps")
        total_rows = len(comp_df)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        page_num = st.number_input(
            "Page", 1, total_pages, 1, key="cat_page", format="%d"
        )
        start, end = (page_num - 1) * page_size, page_num * page_size

        st.caption(f"{shots_sel}-shot results")
        render_styler(apply_gradient(comp_df.iloc[start:end]))

        # Interactive grouped‑bar chart
        chart_df = (
            comp_df.reset_index()
            .rename(columns={comp_df.index.name or "index": cat_sel})
            .melt(id_vars=cat_sel, var_name="Model", value_name="Score")
        )
        width = max(600, 40 * len(comp_df))
        st.altair_chart(
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X(f"{cat_sel}:N", sort=None),
                y="Score:Q",
                color=alt.Color("Model:N", legend=alt.Legend(orient="top-right")),
                xOffset="Model:N",
            )
            .properties(width=width)
            .interactive(bind_y=False),
            use_container_width=True,
        )

        st.download_button(
            "Download CSV",
            comp_df.reset_index().to_csv(index=False).encode(),
            file_name=f"{ds_sel}_{cat_sel}_{shots_sel}shot_compare.csv",
        )
