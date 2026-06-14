"""Named sampled Elo league dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

from core.io import load_csv
from core.paths import LEAGUE_CSV, MODELS_DIR, RESULTS_DIR
from core.style import apply_gradient, page_header, render_styler


def _build_league_board_if_missing() -> None:
    if LEAGUE_CSV.exists():
        return
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_league_board.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--results_dir",
            str(RESULTS_DIR),
            "--models_dir",
            str(MODELS_DIR),
            "--out",
            str(LEAGUE_CSV),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        st.error("Failed to build League board")
        st.text(proc.stderr)
        st.stop()


def _rate_table(standings: pd.DataFrame) -> pd.DataFrame:
    """Return chart-ready row outcome rates."""
    return standings[
        ["Model", "Win Rate", "Loss Rate", "Equal Rate"]
    ].copy()


def _elo_history(
    history: pd.DataFrame,
    *,
    models: list[str],
    initial_elo: float,
    display_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Convert wide league history into a chart-ready rating timeline."""
    display_names = display_names or {}
    rows = [
        {
            "Match": 0,
            "Model": display_names.get(model, model),
            "Elo": float(initial_elo),
        }
        for model in models
    ]
    for _, match in history.iterrows():
        for side in (1, 2):
            model = str(match[f"Model {side}"])
            rows.append(
                {
                    "Match": int(match["Match"]),
                    "Model": display_names.get(model, model),
                    "Elo": float(match[f"Model {side} Elo After"]),
                }
            )
    return pd.DataFrame(rows, columns=["Match", "Model", "Elo"])


def _dataset_rate_table(
    history: pd.DataFrame,
    *,
    display_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate row outcomes by dataset and model."""
    display_names = display_names or {}
    rows = []
    for _, match in history.iterrows():
        model_1_wins = int(match["Model 1 Row Wins"])
        model_2_wins = int(match["Model 2 Row Wins"])
        equals = int(match["Equal Rows"])
        for model, wins, losses in (
            (str(match["Model 1"]), model_1_wins, model_2_wins),
            (str(match["Model 2"]), model_2_wins, model_1_wins),
        ):
            rows.append(
                {
                    "Dataset": str(match["Dataset"]),
                    "Model": display_names.get(model, model),
                    "Rows": wins + losses + equals,
                    "Wins": wins,
                    "Losses": losses,
                    "Equals": equals,
                }
            )
    columns = ["Dataset", "Model", "Rows", "Wins", "Losses", "Equals"]
    if not rows:
        return pd.DataFrame(columns=columns + [
            "Win Rate",
            "Loss Rate",
            "Equal Rate",
            "Score Rate",
        ])
    grouped = (
        pd.DataFrame(rows)
        .groupby(["Dataset", "Model"], as_index=False)[
            ["Rows", "Wins", "Losses", "Equals"]
        ]
        .sum()
    )
    grouped["Win Rate"] = grouped["Wins"] / grouped["Rows"]
    grouped["Loss Rate"] = grouped["Losses"] / grouped["Rows"]
    grouped["Equal Rate"] = grouped["Equals"] / grouped["Rows"]
    grouped["Score Rate"] = (
        grouped["Wins"] + 0.5 * grouped["Equals"]
    ) / grouped["Rows"]
    return grouped


def _elo_chart(standings: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(standings)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("Elo:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("Model:N", sort="-x", title=None),
            color=alt.Color(
                "Elo:Q",
                scale=alt.Scale(range=["#9bc8ad", "#196b45"]),
                legend=None,
            ),
            tooltip=["Rank:Q", "Model:N", alt.Tooltip("Elo:Q", format=".2f")],
        )
        .properties(height=max(240, 42 * len(standings)))
    )


def _rate_chart(rate_table: pd.DataFrame) -> alt.Chart:
    chart_df = rate_table.melt(
        id_vars="Model",
        var_name="Outcome",
        value_name="Rate",
    )
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "Rate:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            y=alt.Y("Model:N", title=None),
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(
                    domain=["Win Rate", "Loss Rate", "Equal Rate"],
                    range=["#23885a", "#c75d5d", "#9a8b45"],
                ),
            ),
            yOffset="Outcome:N",
            tooltip=[
                "Model:N",
                "Outcome:N",
                alt.Tooltip("Rate:Q", format=".1%"),
            ],
        )
        .properties(height=max(240, 70 * len(rate_table)))
    )


def show() -> None:
    """Render the persistent Elo League page."""
    _build_league_board_if_missing()
    if not LEAGUE_CSV.exists():
        st.info("No league standings are available yet.")
        return

    board = load_csv(LEAGUE_CSV)
    if board.empty:
        st.info("No league standings are available yet.")
        return

    leagues = board["League"].dropna().astype(str).unique().tolist()
    league_name = st.sidebar.selectbox("League", leagues, key="league_name")
    standings = board[board["League"].astype(str) == league_name].copy()
    standings = standings.sort_values("Rank").reset_index(drop=True)
    first = standings.iloc[0]
    league_slug = str(first["League Slug"])
    history_path = RESULTS_DIR / "league" / league_slug / "history.csv"
    config_path = RESULTS_DIR / "league" / league_slug / "league.yaml"
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()

    page_header(
        league_name,
        "A persistent sampled league. Early matches calibrate every model; "
        "later matches favor nearby Elo ratings while limiting repeat pairs.",
    )
    summary = st.columns(5)
    summary[0].metric("Leader", str(first["Model"]))
    summary[1].metric("Top Elo", f"{float(first['Elo']):.2f}")
    summary[2].metric("Models", f"{len(standings):,}")
    summary[3].metric("Matches", f"{len(history):,}")
    summary[4].metric("Judge", str(first["Judge"]))
    st.caption(
        f"Datasets: {first['Datasets']} | Rows per match: "
        f"{int(first['Rows Per Match'])} | K-factor: {float(first['K Factor']):g}"
    )

    visible_columns = [
        "Rank",
        "Model",
        "Elo",
        "Games",
        "Match Wins",
        "Match Losses",
        "Match Draws",
        "Win Rate",
        "Loss Rate",
        "Equal Rate",
        "Score Rate",
        "Last Elo Change",
    ]
    render_styler(
        apply_gradient(standings[visible_columns]).format(
            {
                "Elo": "{:.2f}",
                "Win Rate": "{:.1%}",
                "Loss Rate": "{:.1%}",
                "Equal Rate": "{:.1%}",
                "Score Rate": "{:.1%}",
                "Last Elo Change": "{:+.2f}",
            }
        )
    )

    ratings_tab, outcomes_tab, datasets_tab, history_tab = st.tabs(
        ["Current Elo", "Outcome rates", "Dataset rates", "Elo history"]
    )
    with ratings_tab:
        st.altair_chart(_elo_chart(standings), width="stretch")
    with outcomes_tab:
        st.altair_chart(_rate_chart(_rate_table(standings)), width="stretch")
    with datasets_tab:
        if history.empty:
            st.info("This league has no completed matches yet.")
        else:
            display_names = dict(
                zip(
                    standings["Model Stub"].astype(str),
                    standings["Model"].astype(str),
                )
            )
            dataset_rates = _dataset_rate_table(
                history,
                display_names=display_names,
            )
            render_styler(
                apply_gradient(dataset_rates).format(
                    {
                        "Win Rate": "{:.1%}",
                        "Loss Rate": "{:.1%}",
                        "Equal Rate": "{:.1%}",
                        "Score Rate": "{:.1%}",
                    }
                )
            )
    with history_tab:
        if history.empty:
            st.info("This league has no completed matches yet.")
        else:
            initial_elo = 1000.0
            if config_path.exists():
                config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                initial_elo = float(config.get("initial_elo", initial_elo))
                models = list(map(str, config.get("models", [])))
            else:
                models = standings["Model Stub"].astype(str).tolist()
            display_names = dict(
                zip(
                    standings["Model Stub"].astype(str),
                    standings["Model"].astype(str),
                )
            )
            timeline = _elo_history(
                history,
                models=models,
                initial_elo=initial_elo,
                display_names=display_names,
            )
            chart = (
                alt.Chart(timeline)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Match:Q", axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y("Elo:Q", scale=alt.Scale(zero=False)),
                    color="Model:N",
                    tooltip=["Match:Q", "Model:N", alt.Tooltip("Elo:Q", format=".2f")],
                )
                .properties(height=430)
            )
            st.altair_chart(chart, width="stretch")

    if not history.empty:
        st.subheader("Recent matches")
        recent = history.sort_values("Match", ascending=False).head(20).copy()
        recent["Model 1"] = recent["Model 1"].map(
            dict(zip(standings["Model Stub"], standings["Model"]))
        ).fillna(recent["Model 1"])
        recent["Model 2"] = recent["Model 2"].map(
            dict(zip(standings["Model Stub"], standings["Model"]))
        ).fillna(recent["Model 2"])
        render_styler(apply_gradient(recent))

    st.download_button(
        "Download League standings",
        standings.to_csv(index=False).encode(),
        file_name=f"{league_slug}_standings.csv",
    )
