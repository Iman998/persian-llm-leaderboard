"""
📂 Dataset‑view page – row‑level inspection and per‑category scores.

Allows users to:
* pick one dataset
* select one or more models
* optionally filter rows by category columns
* toggle raw model outputs
* view accuracy breakdowns per category
* inspect per-tag dataset statistics
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import colors as mcolors

from leaderboard_lib.data_utils import (
    _norm,
    explode_tag_column,
    filter_rows_by_tags,
    split_tag_value,
    tag_counts,
)

from core.io import load_csv, load_meta
from core.parser import scan_result_maps
from core.paths import DATASETS_DIR
from core.style import SCORE_CMAP, _contrast_color, page_header, render_styler

# ──────────────────────────────────────────────────────────────
# Cached result maps – shared across pages in the same session
# ──────────────────────────────────────────────────────────────
DATASETS, MAIN_MAP, RAW_MAP, CAT_MAP = scan_result_maps()


def _filter_by_categories(df: pd.DataFrame, filters: Dict[str, Set[str]]) -> pd.DataFrame:
    """
    Return rows matching every selected category column.

    Values within one column are ORed, while separate columns are ANDed.
    Comma- and pipe-separated cells are treated as individual tags.
    """
    return filter_rows_by_tags(df, filters)


def _category_values(df: pd.DataFrame, column: str) -> List[str]:
    """Return sorted distinct tags found in a category column."""
    if column not in df.columns:
        return []
    return sorted(
        {
            tag
            for value in df[column].dropna()
            for tag in split_tag_value(value)
        }
    )


def _statistics_columns(
    df: pd.DataFrame, configured_categories: List[str]
) -> List[str]:
    """Return configured categories plus any other tag-like columns."""
    candidates = list(configured_categories)
    candidates.extend(
        col
        for col in df.columns
        if (
            col.lower() in {"tag", "tags"}
            or col.lower().endswith(("_tag", "_tags"))
        )
        and col not in candidates
    )
    return [col for col in candidates if col in df.columns]


def _first_present_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Return the first candidate present, preserving preference order."""
    return next((column for column in candidates if column in df.columns), None)


def _translation_language_columns(meta: Dict[str, object]) -> Tuple[str | None, str | None]:
    """Return configured source and target language columns."""
    return (
        meta.get("source_language_col"),
        meta.get("target_language_col"),
    )


def _translation_display_columns(
    meta: Dict[str, object], language_view: str
) -> Tuple[bool, bool]:
    """Return whether source and target content should be displayed."""
    is_translation = bool(
        meta.get("source_text_col") or meta.get("target_text_col")
    )
    if not is_translation:
        return True, True
    return language_view != "target", language_view != "source"


def _display_prediction(value: object) -> object:
    """Display whole-number model predictions without a decimal suffix."""
    if pd.isna(value):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else value


def _style_row_outputs(
    df: pd.DataFrame, model_columns: List[str]
) -> pd.io.formats.style.Styler:
    """Colour gold answers gold and model predictions by correctness."""
    gold_style = "background-color:#f4e7b2;color:#202521;font-weight:600"
    correct_style = "background-color:#73ad87;color:#102619;font-weight:600"
    incorrect_style = "background-color:#e8a4a4;color:#3a1717;font-weight:600"

    def _styles(data: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        gold_column = "Target Text" if "Target Text" in data.columns else "Gold"
        if gold_column not in data.columns:
            return styles

        styles[gold_column] = gold_style
        gold = data[gold_column].map(_norm)
        for column in model_columns:
            if column not in data.columns:
                continue
            correct = data[column].map(_norm) == gold
            styles[column] = np.where(correct, correct_style, incorrect_style)
        return styles

    return df.style.apply(_styles, axis=None)


def _style_category_scores(
    df: pd.DataFrame, category_column: str, model_columns: List[str]
) -> pd.io.formats.style.Styler:
    """Compare model scores using the leaderboard's soft score scale."""

    def _styles(data: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        columns = [
            column
            for column in model_columns
            if column in data.columns and column != category_column
        ]

        for index, row in data[columns].iterrows():
            scores = pd.to_numeric(row, errors="coerce")
            valid_scores = scores.dropna()
            if valid_scores.empty:
                continue

            low, high = valid_scores.min(), valid_scores.max()
            if low == high:
                colours = {
                    column: mcolors.to_hex(SCORE_CMAP(0.5))
                    for column in valid_scores.index
                }
            else:
                norm = mcolors.Normalize(vmin=low, vmax=high)
                colours = {
                    column: mcolors.to_hex(SCORE_CMAP(norm(score)))
                    for column, score in valid_scores.items()
                }

            for column, colour in colours.items():
                styles.at[index, column] = (
                    f"background-color:{colour};color:{_contrast_color(colour)}"
                )

        return styles

    return df.style.apply(_styles, axis=None)


# ------------------------------------------------------------------ #
# Row‑table helpers
# ------------------------------------------------------------------ #
def _collect_row_tables(
    ds: str,
    models: List[str],
    meta: Dict[str, object],
    cat_filters: Dict[str, Set[str]],
    show_raw: bool,
    language_view: str = "both",
) -> Tuple[pd.DataFrame | None, List[str]]:
    """
    Build a merged DataFrame where each model contributes a column
    of predictions.  Returns `(merged_df | None, warnings)`.
    """
    warnings: List[str] = []
    merged: pd.DataFrame | None = None

    q_name = meta.get("question_col", "question")
    a_name = meta.get("answer_col", "answer")
    question_aliases = [q_name, "Question Body", "question"]
    answer_aliases = [a_name, "Gold", "Key", "answer-key", "answer"]
    choice_cols: List[str] = meta["choice_cols"]
    source_language_col, target_language_col = _translation_language_columns(meta)
    show_source, show_target = _translation_display_columns(meta, language_view)
    is_translation = bool(
        meta.get("source_text_col") or meta.get("target_text_col")
    )

    for m in models:
        raw_file = RAW_MAP.get((ds, m))
        if not raw_file:
            warnings.append(f"No raw CSV for model **{m}**")
            warnings.append(f"Raw CSV not found for **{m}** – hiding raw column.")
            continue

        df = load_csv(raw_file)
        df = _filter_by_categories(df, cat_filters)

        # Identify canonical column names present in the current file
        q_col = _first_present_column(df, question_aliases)
        a_col = _first_present_column(df, answer_aliases)

        if q_col is None:
            st.warning(
                f"Column '{q_name}' missing in {raw_file.name} – skipping model {m}."
            )
            continue
        if a_col is None:
            st.warning(
                f"Column '{a_name}' missing in {raw_file.name} – skipping model {m}."
            )
            continue
        if "pred" not in df.columns:
            warnings.append(f"Column 'pred' missing in *{raw_file.name}* – skipped.")
            continue

        keep_cols = []
        if show_source:
            keep_cols.append(q_col)
            if source_language_col and source_language_col in df.columns:
                keep_cols.append(source_language_col)
        if show_target:
            keep_cols.append(a_col)
            if target_language_col and target_language_col in df.columns:
                keep_cols.append(target_language_col)
            keep_cols.append("pred")
        keep_cols.extend(choice_cols)
        if show_raw and show_target and "raw" in df.columns:
            keep_cols.append("raw")

        rename_map = {
            q_col: "Source Text" if is_translation else "Question",
            a_col: "Target Text" if is_translation else "Gold",
            "pred": m,
            "raw": f"{m}-raw" if show_raw else "raw",
        }
        if source_language_col:
            rename_map[source_language_col] = "Source Language"
        if target_language_col:
            rename_map[target_language_col] = "Target Language"
        df = df[list(dict.fromkeys(keep_cols))].rename(columns=rename_map)
        if m in df.columns:
            df[m] = pd.Series(
                [_display_prediction(value) for value in df[m]],
                index=df.index,
                dtype=object,
            )

        if merged is None:
            merged = df
        else:
            shared_columns = [
                "Question",
                "Gold",
                "Source Text",
                "Target Text",
                "Source Language",
                "Target Language",
                *choice_cols,
            ]
            merged = merged.merge(
                df.drop(
                    columns=[
                        column for column in shared_columns if column in df.columns
                    ]
                ),
                left_index=True,
                right_index=True,
                how="inner",
            )

    return merged, warnings


def _collect_category_tables(
    ds: str,
    models: List[str],
    meta: Dict[str, object],
    cat_filters: Dict[str, Set[str]],
    cat_sel: str,
) -> Tuple[pd.DataFrame | None, List[str]]:
    """
    Assemble per‑category accuracy tables for each selected model.
    """
    frames, warnings = [], []

    for m in models:
        df_c = None

        raw_file = RAW_MAP.get((ds, m))
        if raw_file:
            df_raw = load_csv(raw_file)
            df_raw = _filter_by_categories(df_raw, cat_filters)

            if cat_sel not in df_raw.columns or meta["answer_col"] not in df_raw.columns:
                warnings.append(f"Missing columns in *{raw_file.name}* – skipped.")
                continue

            df_raw = explode_tag_column(df_raw, cat_sel, keep_empty=True)
            df_raw["_correct"] = (
                df_raw["pred"].map(_norm)
                == df_raw[meta["answer_col"]].map(_norm)
            )
            df_c = (
                df_raw.groupby(cat_sel, dropna=False)["_correct"]
                .mean()
                .mul(100)
                .reset_index(name=m)
                .set_index(cat_sel)
            )
        else:
            # Fall back to cached category results when raw rows are unavailable.
            p = CAT_MAP.get((ds, m, cat_sel))
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
    ds_sel = st.sidebar.selectbox("Dataset", DATASETS, key="dataset_sel")
    meta = load_meta(ds_sel)

    models_in_dataset = sorted({mdl for (ds, mdl) in MAIN_MAP if ds == ds_sel})
    # A dataset-specific key guarantees independent widget state
    models_sel = st.sidebar.multiselect(
        "Models",
        options=models_in_dataset,
        default=models_in_dataset[:1],
        key=f"models_{ds_sel}",
    )
    
    # Auto-select first model if user switched dataset and nothing is ticked
    if not models_sel and models_in_dataset:
        st.session_state[f"models_{ds_sel}"] = [models_in_dataset[0]]
        models_sel = st.session_state[f"models_{ds_sel}"]

    page_header(
        "Dataset Explorer",
        f"Inspect row-level outputs, category performance, and tag coverage "
        f"for {ds_sel}.",
    )

    if not models_sel:
        st.info("Select at least one model to continue.")
        return

    # Category filters (optional)
    cat_filters: Dict[str, Set[str]] = {}
    sample_file = RAW_MAP.get((ds_sel, models_sel[0]))
    sample_df = load_csv(sample_file) if sample_file else None
    source_language_col, target_language_col = _translation_language_columns(meta)
    language_columns = {
        column
        for column in (source_language_col, target_language_col)
        if column
    }

    if sample_df is not None and language_columns:
        with st.sidebar.expander("Translation languages", expanded=True):
            for label, column in (
                ("Source language", source_language_col),
                ("Target language", target_language_col),
            ):
                if column and column in sample_df.columns:
                    selected = st.multiselect(
                        label,
                        _category_values(sample_df, column),
                        key=f"{column}_{ds_sel}",
                    )
                    if selected:
                        cat_filters[column] = set(selected)

    if meta["category_cols"]:
        with st.sidebar.expander("Category filters", expanded=False):
            for col in meta["category_cols"]:
                if col in language_columns:
                    continue
                if sample_df is not None and col in sample_df.columns:
                    vals = _category_values(sample_df, col)
                    sel = st.multiselect(
                        col,
                        vals,
                        key=f"{col}_{ds_sel}",
                    )
                    if sel:
                        cat_filters[col] = set(sel)

    # Raw output toggle
    show_raw = st.sidebar.checkbox(
        "Show raw model output", value=False, key=f"show_raw_{ds_sel}"
    )
    language_view = "both"
    if language_columns:
        language_view = st.sidebar.radio(
            "Translation row view",
            options=["both", "source", "target"],
            format_func={
                "both": "Source and target",
                "source": "Source only",
                "target": "Target only",
            }.get,
            horizontal=True,
            key=f"translation_view_{ds_sel}",
        )

    # Build UI tabs
    summary_cols = st.columns(3)
    summary_cols[0].metric("Dataset", ds_sel)
    summary_cols[1].metric("Models selected", f"{len(models_sel):,}")
    summary_cols[2].metric("Active filters", f"{len(cat_filters):,}")
    st.write("")

    row_tab, cat_tab, stats_tab = st.tabs(
        ["Row outputs", "Category scores", "Tag statistics"]
    )

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
            language_view,
        )
        if show_raw and all((ds_sel, m) not in RAW_MAP for m in models_sel):
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

            page_df = merged_df.iloc[start:end]
            render_styler(_style_row_outputs(page_df, models_sel))
            st.download_button(
                "↓  Download CSV",
                merged_df.to_csv(index=False).encode(),
                file_name=f"{ds_sel}_rows_compare.csv",
            )

    # ---------------------------------------------------------------- #
    # Category scores tab
    # ---------------------------------------------------------------- #
    with cat_tab:
        cached_cat_names = {
            k[2] for k in CAT_MAP if k[0] == ds_sel and k[1] in models_sel
        }
        cat_names = sorted(cached_cat_names | set(meta["category_cols"]))
        if not cat_names and not meta["category_cols"]:
            st.info("No per‑category CSVs available for this dataset.")
        else:
            cat_sel = st.selectbox(
                "Category column",
                cat_names,
            )

            comp_df, warnings = _collect_category_tables(
                ds_sel, models_sel, meta, cat_filters, cat_sel
            )
            for w in warnings:
                st.warning(w)

            if comp_df is None:
                st.warning("Category data not found for the selected configuration.")
            else:
                # Paginated table
                page_size = st.selectbox("Rows per page", [50, 100, 200], key="cat_ps")
                total_rows = len(comp_df)
                total_pages = max(1, (total_rows + page_size - 1) // page_size)
                page_num = st.number_input(
                    "Page", 1, total_pages, 1, key="cat_page", format="%d"
                )
                start, end = (page_num - 1) * page_size, page_num * page_size

                page_df = comp_df.iloc[start:end].reset_index()
                category_column = comp_df.index.name or cat_sel
                if page_df.columns[0] != category_column:
                    page_df = page_df.rename(
                        columns={page_df.columns[0]: category_column}
                    )
                render_styler(
                    _style_category_scores(
                        page_df,
                        category_column,
                        models_sel,
                    )
                )

                # Interactive grouped-bar chart
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
                    file_name=f"{ds_sel}_{cat_sel}_compare.csv",
                )

    # ---------------------------------------------------------------- #
    # Tag statistics tab
    # ---------------------------------------------------------------- #
    with stats_tab:
        dataset_file = DATASETS_DIR / ds_sel / "test.csv"
        stats_df = load_csv(dataset_file) if dataset_file.exists() else None
        if stats_df is None:
            sample_file = RAW_MAP.get((ds_sel, models_sel[0]))
            stats_df = load_csv(sample_file) if sample_file else None

        if stats_df is None:
            st.info("Dataset rows are not available for tag statistics.")
        else:
            stat_columns = _statistics_columns(stats_df, meta["category_cols"])
            if not stat_columns:
                st.info("No category or tag columns were found in this dataset.")
            else:
                stat_col = st.selectbox(
                    "Tag/category column",
                    stat_columns,
                    key=f"stats_col_{ds_sel}",
                )
                counts_df = tag_counts(stats_df, stat_col)
                assignments = int(counts_df["Count"].sum())
                missing_rows = int(
                    stats_df[stat_col].map(lambda value: not split_tag_value(value)).sum()
                )

                metric_cols = st.columns(4)
                metric_cols[0].metric("Dataset rows", f"{len(stats_df):,}")
                metric_cols[1].metric("Unique tags", f"{len(counts_df):,}")
                metric_cols[2].metric("Tag assignments", f"{assignments:,}")
                metric_cols[3].metric("Rows without a tag", f"{missing_rows:,}")

                st.caption(
                    "Each tag is counted separately. Percent is the share of dataset "
                    "rows containing that tag, so percentages can total more than 100%."
                )
                st.dataframe(
                    counts_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Percent": st.column_config.NumberColumn(
                            "Percent of rows", format="%.2f%%"
                        )
                    },
                )

                chart_limit = st.selectbox(
                    "Tags shown in chart",
                    [10, 25, 50, 100, "All"],
                    index=1,
                    key=f"stats_limit_{ds_sel}",
                )
                chart_data = (
                    counts_df
                    if chart_limit == "All"
                    else counts_df.head(int(chart_limit))
                )
                st.altair_chart(
                    alt.Chart(chart_data)
                    .mark_bar()
                    .encode(
                        x=alt.X("Count:Q", title="Rows"),
                        y=alt.Y(
                            f"{stat_col}:N",
                            sort="-x",
                            title=stat_col,
                        ),
                        color=alt.Color(
                            "Count:Q",
                            legend=None,
                            scale=alt.Scale(
                                range=["#dfe8e1", "#23885a"]
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip(f"{stat_col}:N", title="Tag"),
                            alt.Tooltip("Count:Q", title="Rows"),
                            alt.Tooltip(
                                "Percent:Q", title="Percent", format=".2f"
                            ),
                        ],
                    )
                    .properties(height=max(240, 24 * len(chart_data))),
                    width="stretch",
                )
                st.download_button(
                    "↓  Download tag statistics",
                    counts_df.to_csv(index=False).encode(),
                    file_name=f"{ds_sel}_{stat_col}_tag_statistics.csv",
                )
