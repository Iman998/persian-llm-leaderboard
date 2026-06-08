import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from leaderboard_runner import combo_executor, paths


def setup_paths(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    data_dir.mkdir()
    models_dir.mkdir()
    results_dir.mkdir()
    monkeypatch.setattr(paths, "DATASETS_DIR", data_dir)
    monkeypatch.setattr(paths, "MODELS_DIR", models_dir)
    monkeypatch.setattr(paths, "RESULTS_DIR", results_dir)
    return data_dir, models_dir, results_dir


def test_missing_dataset_returns_early(tmp_path, monkeypatch, caplog):
    data_dir, models_dir, _ = setup_paths(tmp_path, monkeypatch)
    caplog.set_level("WARNING")
    # create model config so dataset missing triggers first
    (models_dir / "m.yaml").write_text("name: dummy\n")
    with patch("leaderboard_runner.combo_executor.subprocess.run") as mock_run:
        combo_executor.run_single_combo(
            model="m",
            dataset="ds",
            n_rows=None,
            shots=0,
            workers=1,
        )
        mock_run.assert_not_called()
    assert "Dataset not found" in caplog.text


def test_missing_model_returns_early(tmp_path, monkeypatch, caplog):
    data_dir, models_dir, _ = setup_paths(tmp_path, monkeypatch)
    caplog.set_level("WARNING")
    ds_dir = data_dir / "ds"
    ds_dir.mkdir()
    (ds_dir / "test.csv").write_text("a\n1\n")
    (ds_dir / "meta.yaml").write_text("")
    # model file intentionally missing
    with patch("leaderboard_runner.combo_executor.subprocess.run") as mock_run:
        combo_executor.run_single_combo(
            model="m",
            dataset="ds",
            n_rows=None,
            shots=0,
            workers=1,
        )
        mock_run.assert_not_called()
    assert "Model config missing" in caplog.text


def test_dry_run_prints_command(tmp_path, monkeypatch, capsys):
    data_dir, models_dir, results_dir = setup_paths(tmp_path, monkeypatch)
    ds_dir = data_dir / "ds"
    ds_dir.mkdir()
    (ds_dir / "test.csv").write_text("a\n1\n")
    (ds_dir / "meta.yaml").write_text("prompt_template: P\nevaluator: E\n")
    (models_dir / "m.yaml").write_text("name: dummy\n")
    monkeypatch.setattr(combo_executor, "build_run_eval_cmd", lambda **_: ["python", "x.py"])
    with patch("leaderboard_runner.combo_executor.subprocess.run") as mock_run:
        combo_executor.run_single_combo(
            model="m",
            dataset="ds",
            n_rows=None,
            shots=0,
            workers=1,
            dry_run=True,
        )
        mock_run.assert_not_called()
    assert capsys.readouterr().out.strip() == "python x.py"


@pytest.mark.parametrize("task", ["text_generation", "summarization", "translation"])
def test_judge_runs_for_supported_tasks(task, tmp_path, monkeypatch):
    data_dir, models_dir, results_dir = setup_paths(tmp_path, monkeypatch)
    ds_dir = data_dir / "ds"
    ds_dir.mkdir()
    (ds_dir / "test.csv").write_text("question,Key\nq,a\n")
    (ds_dir / "meta.yaml").write_text(
        f"task: {task}\n"
        "judge: true\n"
        "prompt_template: P\n"
        "evaluator: E\n"
        "question_col: question\n"
    )
    (models_dir / "m.yaml").write_text("name: dummy\n")

    result_dir = results_dir / "ds" / "m"
    result_dir.mkdir(parents=True)
    (result_dir / "m.csv").write_text("question,Key,pred\nq,a,a\n")

    calls = []
    monkeypatch.setattr(
        combo_executor,
        "build_run_eval_cmd",
        lambda **kwargs: calls.append(kwargs) or ["python", "x.py"],
    )

    with patch("leaderboard_runner.combo_executor.subprocess.run") as mock_run:
        combo_executor.run_single_combo(
            model="m",
            dataset="ds",
            n_rows=None,
            shots=0,
            workers=1,
            judge=True,
        )

    assert mock_run.call_count == 2
    assert calls[1]["evaluator"] == "evaluators/judge_evaluator.py"
    assert calls[1]["out_csv"] == results_dir / "ds_judge" / "m" / "m.csv"


def test_judge_skips_when_dataset_disables_it(tmp_path, monkeypatch):
    data_dir, models_dir, results_dir = setup_paths(tmp_path, monkeypatch)
    ds_dir = data_dir / "ds"
    ds_dir.mkdir()
    (ds_dir / "test.csv").write_text("question,Key\nq,a\n")
    (ds_dir / "meta.yaml").write_text(
        "task: translation\n"
        "judge: false\n"
        "prompt_template: P\n"
        "evaluator: E\n"
        "question_col: question\n"
    )
    (models_dir / "m.yaml").write_text("name: dummy\n")

    result_dir = results_dir / "ds" / "m"
    result_dir.mkdir(parents=True)
    (result_dir / "m.csv").write_text("question,Key,pred\nq,a,a\n")

    monkeypatch.setattr(
        combo_executor, "build_run_eval_cmd", lambda **_: ["python", "x.py"]
    )

    with patch("leaderboard_runner.combo_executor.subprocess.run") as mock_run:
        combo_executor.run_single_combo(
            model="m",
            dataset="ds",
            n_rows=None,
            shots=0,
            workers=1,
            judge=True,
        )

    assert mock_run.call_count == 1
