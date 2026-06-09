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
    assert table["model"].tolist() == [2]
    assert isinstance(table["model"].iloc[0], int)


def test_display_prediction_removes_only_integral_decimal_suffixes():
    assert dataset_view._display_prediction(2.0) == 2
    assert isinstance(dataset_view._display_prediction(2.0), int)
    assert dataset_view._display_prediction("3.0") == 3
    assert dataset_view._display_prediction(2.5) == 2.5
    assert dataset_view._display_prediction("option A") == "option A"


def test_row_output_styles_gold_and_prediction_correctness():
    df = pd.DataFrame(
        {
            "Question": ["q1", "q2"],
            "Gold": ["A", "B"],
            "model": ["A", "C"],
        }
    )

    styler = dataset_view._style_row_outputs(df, ["model"])
    styler._compute()

    assert ("background-color", "#f4e7b2") in styler.ctx[(0, 1)]
    assert ("background-color", "#73ad87") in styler.ctx[(0, 2)]
    assert ("background-color", "#e8a4a4") in styler.ctx[(1, 2)]


def test_category_styles_compare_models_within_each_row():
    df = pd.DataFrame(
        {
            "Category": ["easy", "hard"],
            "model-a": [90.0, 20.0],
            "model-b": [40.0, 80.0],
        }
    )

    styler = dataset_view._style_category_scores(
        df, "Category", ["model-a", "model-b"]
    )
    styler._compute()

    assert ("background-color", "#73ad87") in styler.ctx[(0, 1)]
    assert ("background-color", "#e8a4a4") in styler.ctx[(0, 2)]
    assert ("background-color", "#e8a4a4") in styler.ctx[(1, 1)]
    assert ("background-color", "#73ad87") in styler.ctx[(1, 2)]
    assert (0, 0) not in styler.ctx


def test_collect_translation_rows_can_show_source_and_target_separately(
    tmp_path, monkeypatch
):
    raw_file = tmp_path / "model_raw.csv"
    pd.DataFrame(
        {
            "text": ["سلام"],
            "gold_translation": ["hello"],
            "src_language": ["Persian"],
            "tgt_lang": ["English"],
            "pred": ["hello"],
        }
    ).to_csv(raw_file, index=False)
    monkeypatch.setattr(
        dataset_view,
        "RAW_MAP",
        {("zharfa_translate", "model"): Path(raw_file)},
    )
    meta = {
        "question_col": "text",
        "answer_col": "gold_translation",
        "choice_cols": [],
        "source_text_col": "text",
        "target_text_col": "gold_translation",
        "source_language_col": "src_language",
        "target_language_col": "tgt_lang",
    }

    both, _ = dataset_view._collect_row_tables(
        "zharfa_translate", ["model"], meta, {}, False, "both"
    )
    source, _ = dataset_view._collect_row_tables(
        "zharfa_translate", ["model"], meta, {}, True, "source"
    )
    target, _ = dataset_view._collect_row_tables(
        "zharfa_translate", ["model"], meta, {}, False, "target"
    )

    assert both.columns.tolist() == [
        "Source Text",
        "Source Language",
        "Target Text",
        "Target Language",
        "model",
    ]
    assert source.columns.tolist() == ["Source Text", "Source Language"]
    assert target.columns.tolist() == ["Target Text", "Target Language", "model"]


def test_translation_language_filters_are_independent():
    df = pd.DataFrame(
        {
            "src_language": ["Persian", "Persian", "English"],
            "tgt_lang": ["English", "Arabic", "Persian"],
        }
    )

    filtered = dataset_view._filter_by_categories(
        df,
        {
            "src_language": {"Persian"},
            "tgt_lang": {"Arabic"},
        },
    )

    assert filtered.index.tolist() == [1]
