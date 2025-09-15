import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from leaderboard_lib import evaluation


def _build_args(**kwargs):
    defaults = dict(
        dataset="ds.csv",
        meta="meta.yaml",
        model="model.yaml",
        prompt=None,
        shots=0,
        workers=1,
        evaluator=None,
        evaluator_class=None,
        out="out.csv",
        n_rows=None,
        verbose=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_parse_args_definitions(monkeypatch):
    added = []

    class DummyParser:
        def add_argument(self, *names, **kwargs):
            added.append((names, kwargs))

        def parse_args(self):
            return argparse.Namespace()

    monkeypatch.setattr(argparse, "ArgumentParser", lambda: DummyParser())
    evaluation.parse_args()

    params = {name: kwargs for names, kwargs in added for name in names}

    required = {k for k, v in params.items() if v.get("required")}
    assert {"--dataset", "--meta", "--model", "--out"} <= required
    assert "--prompt" in params and not params["--prompt"].get("required")
    assert params["--shots"]["type"] is int
    assert params["--workers"]["type"] is int


def test_load_configs_infers_evaluator_class(tmp_path):
    dataset = tmp_path / "data.csv"
    pd.DataFrame({"Key": ["1"], "question": ["q"], "pred": ["1"]}).to_csv(dataset, index=False)

    model = tmp_path / "model.yaml"
    model.write_text("name: dummy\n")

    meta = tmp_path / "meta.yaml"
    meta.write_text("answer_col: Key\nquestion_col: question\n")

    evaluator = tmp_path / "dummy_evaluator.py"
    evaluator.write_text(
        "class DummyEvaluator:\n"
        "    def __init__(self, *a, **k):\n        pass\n"
        "    def evaluate_df(self, df, max_workers):\n        return df\n"
    )

    args = _build_args(dataset=str(dataset), meta=str(meta), model=str(model), evaluator=str(evaluator))
    Evaluator, model_cfg, meta_cfg, df = evaluation.load_configs(args)
    assert Evaluator.__name__ == "DummyEvaluator"
    assert model_cfg["name"] == "dummy"
    assert meta_cfg["answer_col"] == "Key"
    assert not df.empty


def test_load_configs_class_not_found(tmp_path):
    dataset = tmp_path / "data.csv"
    pd.DataFrame({"Key": ["1"], "question": ["q"], "pred": ["1"]}).to_csv(dataset, index=False)

    model = tmp_path / "model.yaml"
    model.write_text("name: dummy\n")

    meta = tmp_path / "meta.yaml"
    meta.write_text("answer_col: Key\nquestion_col: question\n")

    evaluator = tmp_path / "no_class.py"
    evaluator.write_text("class Other:\n    pass\n")

    args = _build_args(dataset=str(dataset), meta=str(meta), model=str(model), evaluator=str(evaluator))
    with pytest.raises(AttributeError):
        evaluation.load_configs(args)


def test_sample_dataset_logs_and_rows(caplog):
    df = pd.DataFrame({"a": range(10)})
    caplog.set_level(logging.INFO)
    sampled = evaluation.sample_dataset(df, 5, verbose=True)
    assert len(sampled) == 5
    assert "dataframe now has 5 rows" in caplog.text


def test_save_results_creates_files_and_handles_missing_columns(tmp_path, capsys):
    df = pd.DataFrame(
        {
            "pred": ["a", "b", "a", "b"],
            "correct": ["a", "a", "a", "b"],
            "question": ["q1", "q2", "q3", "q4"],
            "topic": ["t1", "t1", "t2", "t2"],
        }
    )
    meta_cfg = {"category_cols": ["topic", "missing"]}
    out = tmp_path / "result.csv"

    evaluation.save_results(df, meta_cfg, out, answer_col="correct", question_col="question")
    captured = capsys.readouterr().out

    assert out.exists()
    main = pd.read_csv(out)
    assert "Key" in main.columns
    assert "Question Body" in main.columns

    topic_file = tmp_path / "result_topic.csv"
    assert topic_file.exists()
    topic_df = pd.read_csv(topic_file)
    assert set(topic_df.columns) == {"topic", "Accuracy"}

    missing_file = tmp_path / "result_missing.csv"
    assert not missing_file.exists()
    assert "column 'missing'" in captured

    raw_file = tmp_path / "result_raw.csv"
    assert raw_file.exists()
