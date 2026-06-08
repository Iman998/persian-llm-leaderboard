from pathlib import Path

import pandas as pd

from app.views import dataset_view


def test_collect_row_tables_prefers_metadata_answer_column(tmp_path, monkeypatch):
    raw_file = tmp_path / "model_raw.csv"
    pd.DataFrame(
        {
            "Question Body": ["question"],
            "option1": ["one"],
            "option2": ["two"],
            "answer-key": [2],
            "answer": ["descriptive answer"],
            "Key": [2],
            "Gold": ["wrong fallback"],
            "pred": [2],
        }
    ).to_csv(raw_file, index=False)
    monkeypatch.setattr(
        dataset_view,
        "RAW_MAP",
        {("dataset", "model"): Path(raw_file)},
    )

    table, warnings = dataset_view._collect_row_tables(
        "dataset",
        ["model"],
        {
            "question_col": "question",
            "answer_col": "answer-key",
            "choice_cols": ["option1", "option2"],
        },
        {},
        False,
    )

    assert warnings == []
    assert table is not None
    assert table["Gold"].tolist() == [2]
