from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
for path in (ROOT, APP):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from app.views.league import _dataset_rate_table, _elo_history, _rate_table


def test_rate_table_and_elo_history_are_chart_ready():
    standings = pd.DataFrame(
        {
            "Model": ["First", "Second"],
            "Win Rate": [0.6, 0.3],
            "Loss Rate": [0.3, 0.6],
            "Equal Rate": [0.1, 0.1],
        }
    )
    history = pd.DataFrame(
        {
            "Match": [1],
            "Model 1": ["m1"],
            "Model 2": ["m2"],
            "Model 1 Elo After": [1008],
            "Model 2 Elo After": [992],
        }
    )

    rates = _rate_table(standings)
    timeline = _elo_history(
        history,
        models=["m1", "m2"],
        initial_elo=1000,
        display_names={"m1": "First", "m2": "Second"},
    )

    assert rates.columns.tolist() == [
        "Model",
        "Win Rate",
        "Loss Rate",
        "Equal Rate",
    ]
    assert timeline["Match"].tolist() == [0, 0, 1, 1]
    assert timeline["Elo"].tolist() == [1000, 1000, 1008, 992]


def test_dataset_rate_table_reports_each_models_dataset_record():
    history = pd.DataFrame(
        {
            "Dataset": ["translation"],
            "Model 1": ["m1"],
            "Model 2": ["m2"],
            "Model 1 Row Wins": [3],
            "Model 2 Row Wins": [1],
            "Equal Rows": [1],
        }
    )

    table = _dataset_rate_table(
        history,
        display_names={"m1": "First", "m2": "Second"},
    ).set_index("Model")

    assert table.loc["First", "Win Rate"] == 0.6
    assert table.loc["Second", "Loss Rate"] == 0.6
    assert table.loc["First", "Equal Rate"] == 0.2
