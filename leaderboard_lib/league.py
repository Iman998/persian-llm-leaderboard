"""Elo standings and matchmaking helpers for named model leagues."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import pandas as pd


STANDING_COLUMNS = [
    "Model",
    "Elo",
    "Games",
    "Match Wins",
    "Match Losses",
    "Match Draws",
    "Row Battles",
    "Row Wins",
    "Row Losses",
    "Row Equals",
    "Win Rate",
    "Loss Rate",
    "Equal Rate",
    "Score Rate",
    "Last Elo Change",
]

HISTORY_COLUMNS = [
    "Match",
    "Dataset",
    "Model 1",
    "Model 2",
    "Judge",
    "Rows",
    "Model 1 Row Wins",
    "Model 2 Row Wins",
    "Equal Rows",
    "Model 1 Win Rate",
    "Model 2 Win Rate",
    "Equal Rate",
    "Model 1 Score",
    "Model 1 Elo Before",
    "Model 1 Elo After",
    "Model 1 Elo Change",
    "Model 2 Elo Before",
    "Model 2 Elo After",
    "Model 2 Elo Change",
    "Result",
    "Sample Cycle",
]


def initial_standings(
    models: Iterable[str],
    *,
    initial_elo: float = 1000.0,
) -> pd.DataFrame:
    """Return empty standings for a new league."""
    rows = []
    for model in models:
        rows.append(
            {
                "Model": model,
                "Elo": float(initial_elo),
                "Games": 0,
                "Match Wins": 0,
                "Match Losses": 0,
                "Match Draws": 0,
                "Row Battles": 0,
                "Row Wins": 0,
                "Row Losses": 0,
                "Row Equals": 0,
                "Win Rate": 0.0,
                "Loss Rate": 0.0,
                "Equal Rate": 0.0,
                "Score Rate": 0.0,
                "Last Elo Change": 0.0,
            }
        )
    return pd.DataFrame(rows, columns=STANDING_COLUMNS)


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return Elo's expected score for player A."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def pair_key(model_1: str, model_2: str) -> tuple[str, str]:
    return tuple(sorted((model_1, model_2)))


def _pair_counts(history: pd.DataFrame) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    if history.empty:
        return counts
    for model_1, model_2 in history[["Model 1", "Model 2"]].itertuples(
        index=False, name=None
    ):
        key = pair_key(str(model_1), str(model_2))
        counts[key] = counts.get(key, 0) + 1
    return counts


def choose_matchup(
    standings: pd.DataFrame,
    history: pd.DataFrame,
    common_datasets: dict[tuple[str, str], list[str]],
    *,
    calibration_games: int = 2,
    repeat_penalty: float = 64.0,
) -> tuple[str, str]:
    """Choose a balanced pair, then prefer nearby Elo ratings.

    Models with the fewest games act as anchors. During calibration, opponents
    that also need games are preferred. Afterwards, Elo distance plus a repeat
    penalty drives competitive pairings.
    """
    ratings = standings.set_index("Model")["Elo"].astype(float).to_dict()
    games = standings.set_index("Model")["Games"].astype(int).to_dict()
    pair_counts = _pair_counts(history)
    eligible_pairs = [
        pair
        for pair in combinations(sorted(ratings), 2)
        if common_datasets.get(pair_key(*pair))
    ]
    if not eligible_pairs:
        raise ValueError("No league model pair shares an available dataset")

    eligible_models = {model for pair in eligible_pairs for model in pair}
    under_calibration = {
        model for model in eligible_models if games[model] < calibration_games
    }
    anchor_pool = under_calibration or eligible_models
    minimum_games = min(games[model] for model in anchor_pool)
    anchors = {
        model for model in anchor_pool if games[model] == minimum_games
    }
    candidate_pairs = [
        pair for pair in eligible_pairs if pair[0] in anchors or pair[1] in anchors
    ]

    def _key(pair: tuple[str, str]) -> tuple[object, ...]:
        model_1, model_2 = pair
        anchor = model_1 if model_1 in anchors else model_2
        opponent = model_2 if anchor == model_1 else model_1
        repeats = pair_counts.get(pair_key(*pair), 0)
        gap = abs(ratings[anchor] - ratings[opponent])
        if under_calibration:
            return (
                float(games[opponent] >= calibration_games),
                float(repeats),
                float(games[opponent]),
                gap,
                anchor,
                opponent,
            )
        return (
            gap + repeats * repeat_penalty,
            float(games[opponent]),
            float(repeats),
            anchor,
            opponent,
        )

    selected = min(candidate_pairs, key=_key)
    anchor = selected[0] if selected[0] in anchors else selected[1]
    opponent = selected[1] if anchor == selected[0] else selected[0]
    return anchor, opponent


def choose_dataset(
    model_1: str,
    model_2: str,
    datasets: list[str],
    history: pd.DataFrame,
) -> str:
    """Choose the least-used dataset for the pair and then league-wide."""
    if not datasets:
        raise ValueError("No common league dataset is available for this pair")
    if history.empty:
        return sorted(datasets)[0]

    selected_pair = pair_key(model_1, model_2)
    pair_counts = {dataset: 0 for dataset in datasets}
    total_counts = {dataset: 0 for dataset in datasets}
    for _, row in history.iterrows():
        dataset = str(row["Dataset"])
        if dataset not in total_counts:
            continue
        total_counts[dataset] += 1
        row_pair = pair_key(
            str(row["Model 1"]),
            str(row["Model 2"]),
        )
        if row_pair == selected_pair:
            pair_counts[dataset] += 1
    return min(
        datasets,
        key=lambda dataset: (
            pair_counts[dataset],
            total_counts[dataset],
            dataset,
        ),
    )


def update_standings(
    standings: pd.DataFrame,
    *,
    model_1: str,
    model_2: str,
    model_1_row_wins: int,
    model_2_row_wins: int,
    equal_rows: int,
    k_factor: float,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Apply one aggregate row match to Elo and cumulative league statistics."""
    total = model_1_row_wins + model_2_row_wins + equal_rows
    if total <= 0:
        raise ValueError("A league match must contain at least one valid row")

    updated = standings.copy()
    updated = updated.set_index("Model")
    rating_1 = float(updated.loc[model_1, "Elo"])
    rating_2 = float(updated.loc[model_2, "Elo"])
    actual_1 = (model_1_row_wins + 0.5 * equal_rows) / total
    expected_1 = expected_score(rating_1, rating_2)
    delta_1 = float(k_factor) * (actual_1 - expected_1)
    delta_2 = -delta_1

    updated.loc[model_1, "Elo"] = rating_1 + delta_1
    updated.loc[model_2, "Elo"] = rating_2 + delta_2
    updated.loc[model_1, "Games"] += 1
    updated.loc[model_2, "Games"] += 1
    updated.loc[model_1, "Row Battles"] += total
    updated.loc[model_2, "Row Battles"] += total
    updated.loc[model_1, "Row Wins"] += model_1_row_wins
    updated.loc[model_2, "Row Wins"] += model_2_row_wins
    updated.loc[model_1, "Row Losses"] += model_2_row_wins
    updated.loc[model_2, "Row Losses"] += model_1_row_wins
    updated.loc[model_1, "Row Equals"] += equal_rows
    updated.loc[model_2, "Row Equals"] += equal_rows
    updated.loc[model_1, "Last Elo Change"] = delta_1
    updated.loc[model_2, "Last Elo Change"] = delta_2

    if actual_1 > 0.5:
        result = "model_1"
        updated.loc[model_1, "Match Wins"] += 1
        updated.loc[model_2, "Match Losses"] += 1
    elif actual_1 < 0.5:
        result = "model_2"
        updated.loc[model_2, "Match Wins"] += 1
        updated.loc[model_1, "Match Losses"] += 1
    else:
        result = "equal"
        updated.loc[model_1, "Match Draws"] += 1
        updated.loc[model_2, "Match Draws"] += 1

    for model in (model_1, model_2):
        row_battles = int(updated.loc[model, "Row Battles"])
        row_wins = int(updated.loc[model, "Row Wins"])
        row_losses = int(updated.loc[model, "Row Losses"])
        row_equals = int(updated.loc[model, "Row Equals"])
        updated.loc[model, "Win Rate"] = row_wins / row_battles
        updated.loc[model, "Loss Rate"] = row_losses / row_battles
        updated.loc[model, "Equal Rate"] = row_equals / row_battles
        updated.loc[model, "Score Rate"] = (
            row_wins + 0.5 * row_equals
        ) / row_battles

    updated = updated.reset_index()
    updated = updated.sort_values(
        ["Elo", "Score Rate", "Model"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return updated[STANDING_COLUMNS], {
        "model_1_score": actual_1,
        "model_1_rating_before": rating_1,
        "model_1_rating_after": rating_1 + delta_1,
        "model_1_delta": delta_1,
        "model_2_rating_before": rating_2,
        "model_2_rating_after": rating_2 + delta_2,
        "model_2_delta": delta_2,
        "result": result,
    }
