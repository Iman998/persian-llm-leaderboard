import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
for path in (ROOT, APP):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from app.views.translation_board import (  # noqa: E402
    _add_dataset_averages,
    _group_translation_columns,
    _split_metric_column,
)


def test_split_metric_column():
    assert _split_metric_column("zharfa_translate (BLEU)") == (
        "zharfa_translate",
        "BLEU",
    )
    assert _split_metric_column("Average") is None


def test_group_translation_columns_uses_dataset_and_metric_headers():
    flat = pd.DataFrame(
        {
            "Model": ["model-a"],
            "Average": [0.7],
            "zharfa_translate (BLEU)": [0.5],
            "zharfa_translate (chrF)": [0.9],
            "translation_example (BLEU)": [0.4],
        }
    )

    grouped = _group_translation_columns(flat)

    assert grouped.columns.tolist() == [
        ("", "Model"),
        ("", "Average"),
        ("zharfa_translate", "BLEU"),
        ("zharfa_translate", "chrF"),
        ("translation_example", "BLEU"),
    ]


def test_add_dataset_averages_places_average_after_each_dataset():
    flat = pd.DataFrame(
        {
            "Model": ["model-a", "model-b"],
            "zharfa_translate (BLEU)": [0.5, 0.2],
            "zharfa_translate (chrF)": [0.9, None],
            "translation_example (BLEU)": [0.4, 0.6],
            "translation_example (METEOR)": [0.8, 0.4],
        }
    )

    averaged = _add_dataset_averages(flat)

    assert averaged.columns.tolist() == [
        "Model",
        "zharfa_translate (BLEU)",
        "zharfa_translate (chrF)",
        "zharfa_translate (Average)",
        "translation_example (BLEU)",
        "translation_example (METEOR)",
        "translation_example (Average)",
    ]
    assert averaged["zharfa_translate (Average)"].tolist() == [0.7, 0.2]
    assert averaged["translation_example (Average)"].tolist() == [0.6, 0.5]
