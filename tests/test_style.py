from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.style import (
    _score_columns,
    apply_gradient,
)


def test_apply_gradient_handles_nan_in_average():
    df = pd.DataFrame(
        {
            "Average": [0.5, np.nan, 0.3],
            "Model": ["m1", "m2", "m3"],
        }
    )
    styler = apply_gradient(df)
    assert isinstance(styler, pd.io.formats.style.Styler)


def test_apply_gradient_supports_grouped_leaderboard_columns():
    columns = pd.MultiIndex.from_tuples(
        [
            ("", "Rank"),
            ("", "Model"),
            ("", "Average"),
            ("", "Parameters"),
            ("", "Organization"),
            ("", "License"),
            ("translation", "BLEU"),
            ("translation", "Average"),
        ]
    )
    df = pd.DataFrame(
        [
            ["1", "model-a", 0.9, "7B", "ZharfaTech", "MIT", 0.8, 0.85],
            ["2", "model-b", 0.7, "13B", "Other", "Proprietary", 0.4, 0.5],
        ],
        columns=columns,
    )

    html = apply_gradient(df).to_html()

    assert "ZharfaTech" in html
    assert "Proprietary" in html
    assert "BLEU" in html


def test_score_columns_exclude_numeric_metadata():
    df = pd.DataFrame(
        {
            "Rank": [1, 2, 3, 4],
            "Parameters": [7, 13, 32, 70],
            "dataset (Accuracy)": [0.9, 0.7, 0.5, 0.2],
            "Average": [0.8, 0.6, 0.4, 0.1],
        }
    )

    assert _score_columns(df) == ["dataset (Accuracy)", "Average"]


def test_each_dataset_score_column_gets_gold_silver_and_bronze():
    df = pd.DataFrame(
        {
            "Model": ["first", "second", "third", "fourth"],
            "dataset-a (Accuracy)": [0.9, 0.8, 0.7, 0.1],
            "dataset-b (F1)": [0.2, 0.6, 0.9, 0.7],
        }
    )

    styler = apply_gradient(df)
    styler._compute()

    gold = ("background-color", "#FFD700")
    silver = ("background-color", "#A7ADB2")
    bronze = ("background-color", "#A66F45")
    assert gold in styler.ctx[(0, 1)]
    assert silver in styler.ctx[(1, 1)]
    assert bronze in styler.ctx[(2, 1)]
    assert gold in styler.ctx[(2, 2)]
    assert silver in styler.ctx[(3, 2)]
    assert bronze in styler.ctx[(1, 2)]
