"""
🤖 LLM‑Judge page – aggregate scores produced by an “evaluator” model.

Functionality:
* list datasets that contain judge/evaluator outputs
* show mean score columns (Score, PredScore, …) per model
* optional per‑category breakdown
"""
from __future__ import annotations

from typing import List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from core.io import load_csv
from core.parser import scan_result_maps
from core.style import apply_gradient, page_header, render_styler

# -------------------------------------------------------------
# Build result maps once (shared across pages)
# -------------------------------------------------------------
DATASETS, MAIN_MAP, _RAW_MAP, CAT_MAP = scan_result_maps()


def _discover_judge_datasets() -> List[str]:
    """Return datasets whose name hints at summarisation / translation."""
    return [
        ds
        for ds in DATASETS
        if any(key in ds.lower() for key in ("summarization", "translation"))
    ]


def _collect_judge_table(ds: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return a `(table, warnings)` tuple summarising judge scores
    for every model available in the dataset.
    """
    warnings: List[str] = []
    rows = []
    metric_names = set()

    models = sorted({mdl for (d, mdl) in MAIN_MAP if d == ds})
    for m in models:
        p = MAIN_MAP.get((ds, m))
        if not p:
            continue
        df = load_csv(p).apply(pd.to_numeric, errors="coerce")

        cols = [
            c
            for c in df.columns
            if (
                c.lower().startswith("score")
                or c.lower().startswith("pred")
                or "judge" in c.lower()
            )
            and not df[c].isna().all()
        ]
        if not cols:  # Fallback – use *any* non‑NA numeric columns
            cols = [c for c in df.columns if not df[c].isna().all()]

        if not cols:
            warnings.append(f"No numeric judge columns found for model **{m}**")
            continue

        scores = {c: df[c].mean() for c in cols}
        if len(scores) > 1:
            scores["Average"] = sum(scores.values()) / len(scores)
        metric_names.update(scores.keys())
        rows.append({"Model": m, **scores})

    metric_order = ["Model"] + [c for c in sorted(metric_names) if c != "Model"]
    df = pd.DataFrame(rows).reindex(columns=metric_order)
    return df, warnings


def _collect_category_breakdown(ds: str, models: List[str], cat_sel: str) -> pd.DataFrame | None:
    """Merge per‑category judge CSVs from all selected models."""
    frames = []
    for m in models:
        p = CAT_MAP.get((ds, m, cat_sel))
        if not p:
            continue
        df = load_csv(p)
        metric_cols = [c for c in df.columns if c != cat_sel]
        if not metric_cols:
            continue
        metric_col = metric_cols[0]  # assume single metric column
        frames.append(df.rename(columns={metric_col: m}).set_index(cat_sel))
    return pd.concat(frames, axis=1) if frames else None


def show() -> None:
    """Render the LLM‑Judge page."""
    judge_datasets = _discover_judge_datasets()
    if not judge_datasets:
        st.info("No LLM‑Judge datasets available.")
        return

    ds_sel = st.sidebar.selectbox("Dataset", judge_datasets)
    table_df, warnings = _collect_judge_table(ds_sel)
    for w in warnings:
        st.warning(w)

    if table_df.empty:
        st.warning("Judge scores not found for this dataset.")
        return

    page_header(
        "LLM Judge",
        f"Review evaluator-model scores and category-level comparisons for {ds_sel}.",
    )
    summary_cols = st.columns(3)
    summary_cols[0].metric("Dataset", ds_sel)
    summary_cols[1].metric("Models scored", f"{len(table_df):,}")
    summary_cols[2].metric(
        "Score fields",
        f"{max(0, len(table_df.columns) - 1):,}",
    )
    st.write("")
    render_styler(apply_gradient(table_df))
    st.download_button(
        "↓  Download CSV",
        table_df.to_csv(index=False).encode(),
        file_name=f"{ds_sel}_judge_scores.csv",
    )

    # Optional per‑category breakdown
    cat_names = sorted({k[2] for k in CAT_MAP if k[0] == ds_sel})
    if cat_names:
        st.subheader("Category breakdown")
        cat_sel = st.selectbox("Category column", cat_names)

        comp_df = _collect_category_breakdown(
            ds_sel, table_df["Model"].tolist(), cat_sel
        )
        if comp_df is None:
            st.warning("No category CSVs found for selected models.")
            return

        # Paginated table
        page_size = st.selectbox("Rows per page", [50, 100, 200], key="judge_cat_ps")
        total_rows = len(comp_df)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        page_num = st.number_input(
            "Page", 1, total_pages, 1, key="judge_cat_page", format="%d"
        )
        start, end = (page_num - 1) * page_size, page_num * page_size

        render_styler(apply_gradient(comp_df.iloc[start:end]))

        # Grouped bar chart
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
                color=alt.Color(
                    "Model:N",
                    legend=alt.Legend(orient="top-right"),
                    scale=alt.Scale(
                        range=[
                            "#23885a",
                            "#6f8774",
                            "#2f3f35",
                            "#a8b9ad",
                            "#53665a",
                        ]
                    ),
                ),
                xOffset="Model:N",
            )
            .properties(width=width)
            .interactive(bind_y=False),
            width="stretch",
        )

        st.download_button(
            "↓  Download CSV",
            comp_df.reset_index().to_csv(index=False).encode(),
            file_name=f"{ds_sel}_{cat_sel}_judge_compare.csv",
        )
