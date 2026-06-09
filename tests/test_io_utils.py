import sys
from pathlib import Path
import types
import importlib

import pandas as pd
import pytest

# Ensure project root is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Mock streamlit cache decorator to avoid side effects
streamlit = types.SimpleNamespace()

def _cache_data_stub(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

streamlit.cache_data = _cache_data_stub
sys.modules["streamlit"] = streamlit

import app.core.io as io
io = importlib.reload(io)


def test_load_csv(tmp_path):
    df_expected = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    csv_file = tmp_path / "test.csv"
    df_expected.to_csv(csv_file, index=False)

    df = io.load_csv(csv_file)
    pd.testing.assert_frame_equal(df, df_expected)


def test_load_meta_fallback(tmp_path, monkeypatch):
    data_dir = tmp_path / "datasets"
    data_dir.mkdir()
    monkeypatch.setattr(io, "DATASETS_DIR", data_dir)

    # Dataset without meta.yaml
    (data_dir / "no_meta").mkdir()
    defaults = io.load_meta("no_meta")
    assert defaults == {
        "question_col": "question",
        "answer_col": "Key",
        "category_cols": [],
        "choice_cols": [],
        "language": "",
        "description": "",
        "board": "leaderboard",
        "source_text_col": None,
        "target_text_col": None,
        "source_language_col": None,
        "target_language_col": None,
    }

    # Dataset with partial meta.yaml
    ds = data_dir / "partial"
    ds.mkdir()
    (ds / "meta.yaml").write_text("question_col: prompt\nlanguage: fa\n")
    partial = io.load_meta("partial")
    assert partial["question_col"] == "prompt"
    assert partial["language"] == "fa"
    assert partial["answer_col"] == "Key"
    assert partial["category_cols"] == []
    assert partial["choice_cols"] == []
    assert partial["description"] == ""
    assert partial["board"] == "leaderboard"
    assert partial["source_text_col"] is None
    assert partial["target_text_col"] is None
    assert partial["source_language_col"] is None
    assert partial["target_language_col"] is None


def test_numeric_cols():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [3.0, 4.5]})
    cols = io.numeric_cols(df)
    assert cols == ["a", "c"]
