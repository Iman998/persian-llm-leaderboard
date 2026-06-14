from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from leaderboard_lib.league import (
    HISTORY_COLUMNS,
    choose_dataset,
    choose_matchup,
    initial_standings,
    pair_key,
    update_standings,
)


def _history(rows=None):
    return pd.DataFrame(rows or [], columns=HISTORY_COLUMNS)


def test_update_standings_applies_zero_sum_elo_and_rates():
    standings = initial_standings(["m1", "m2"])

    updated, rating = update_standings(
        standings,
        model_1="m1",
        model_2="m2",
        model_1_row_wins=3,
        model_2_row_wins=1,
        equal_rows=0,
        k_factor=32,
    )

    by_model = updated.set_index("Model")
    assert by_model.loc["m1", "Elo"] == 1008
    assert by_model.loc["m2", "Elo"] == 992
    assert by_model.loc["m1", "Win Rate"] == pytest.approx(0.75)
    assert by_model.loc["m2", "Loss Rate"] == pytest.approx(0.75)
    assert rating["model_1_delta"] == pytest.approx(
        -rating["model_2_delta"]
    )
    assert rating["result"] == "model_1"


def test_matchmaking_calibrates_least_played_model():
    standings = initial_standings(["a", "b", "c"])
    standings.loc[standings["Model"] == "a", "Games"] = 2
    standings.loc[standings["Model"] == "b", "Games"] = 1
    common = {
        pair_key("a", "b"): ["ds"],
        pair_key("a", "c"): ["ds"],
        pair_key("b", "c"): ["ds"],
    }

    matchup = choose_matchup(
        standings,
        _history(),
        common,
        calibration_games=2,
    )

    assert "c" in matchup
    assert "b" in matchup


def test_matchmaking_prefers_nearby_elo_and_penalizes_repeats():
    standings = initial_standings(["a", "b", "c"])
    standings["Games"] = 4
    standings.loc[standings["Model"] == "a", "Elo"] = 1000
    standings.loc[standings["Model"] == "b", "Elo"] = 1010
    standings.loc[standings["Model"] == "c", "Elo"] = 1100
    common = {
        pair_key("a", "b"): ["ds"],
        pair_key("a", "c"): ["ds"],
        pair_key("b", "c"): ["ds"],
    }

    matchup = choose_matchup(
        standings,
        _history(),
        common,
        calibration_games=2,
    )
    assert set(matchup) == {"a", "b"}

    history = _history(
        [
            {
                "Model 1": "a",
                "Model 2": "b",
                "Dataset": "ds",
            }
        ]
    )
    matchup = choose_matchup(
        standings,
        history,
        common,
        calibration_games=2,
        repeat_penalty=200,
    )
    assert set(matchup) != {"a", "b"}


def test_dataset_selection_balances_pair_then_whole_league():
    history = _history(
        [
            {"Model 1": "a", "Model 2": "b", "Dataset": "d1"},
            {"Model 1": "a", "Model 2": "c", "Dataset": "d2"},
        ]
    )

    assert choose_dataset("a", "b", ["d1", "d2"], history) == "d2"
