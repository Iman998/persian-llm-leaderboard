import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

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
    assert (
        calls[1]["out_csv"]
        == results_dir / "ds_judge_reference" / "m" / "m.csv"
    )


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


def test_structured_judge_uses_configured_model_and_prompt(tmp_path, monkeypatch):
    data_dir, models_dir, results_dir = setup_paths(tmp_path, monkeypatch)
    ds_dir = data_dir / "ds"
    ds_dir.mkdir()
    (ds_dir / "test.csv").write_text("text,gold\nhello,سلام\n")
    (ds_dir / "meta.yaml").write_text(
        "task: translation\n"
        "question_col: text\n"
        "answer_col: gold\n"
        "prompt_template: candidate.jinja2\n"
        "evaluator: candidate.py\n"
        "judge:\n"
        "  enabled: true\n"
        "  model: judge-model\n"
        "  prompt_template: judge-translation.jinja2\n"
        "  evaluator: evaluators/judge_evaluator.py\n"
        "  score_min: 0\n"
        "  score_max: 100\n"
    )
    (models_dir / "candidate.yaml").write_text("name: candidate\n")
    (models_dir / "judge-model.yaml").write_text("name: judge\n")

    result_dir = results_dir / "ds" / "candidate"
    result_dir.mkdir(parents=True)
    (result_dir / "candidate.csv").write_text(
        "text,gold,pred\nhello,سلام,درود\n"
    )

    calls = []

    def capture_command(**kwargs):
        if kwargs["evaluator"] == "evaluators/judge_evaluator.py":
            judge_meta = yaml.safe_load(Path(kwargs["meta_path"]).read_text())
            assert judge_meta["metrics"] == ["llm_judge_score"]
            assert judge_meta["judge_score_max"] == 100
        calls.append(kwargs)
        return ["python", "x.py"]

    monkeypatch.setattr(combo_executor, "build_run_eval_cmd", capture_command)

    with patch("leaderboard_runner.combo_executor.subprocess.run"):
        combo_executor.run_single_combo(
            model="candidate",
            dataset="ds",
            n_rows=None,
            shots=0,
            workers=1,
            judge=True,
        )

    assert calls[1]["model_stub"] == "judge-model"
    assert calls[1]["prompt_template"] == "judge-translation.jinja2"
    assert calls[1]["n_rows"] is None


def test_judge_only_reuses_existing_result_and_runs_both_modes(
    tmp_path, monkeypatch
):
    data_dir, models_dir, results_dir = setup_paths(tmp_path, monkeypatch)
    ds_dir = data_dir / "ds"
    ds_dir.mkdir()
    (ds_dir / "meta.yaml").write_text(
        "task: translation\n"
        "question_col: text\n"
        "answer_col: gold\n"
        "judge:\n"
        "  enabled: true\n"
        "  model: judge-model\n"
        "  reference_prompt_template: ref.jinja2\n"
        "  no_reference_prompt_template: noref.jinja2\n"
    )
    (models_dir / "candidate.yaml").write_text("name: candidate\n")
    (models_dir / "judge-model.yaml").write_text("name: judge\n")
    result_dir = results_dir / "ds" / "candidate"
    result_dir.mkdir(parents=True)
    (result_dir / "candidate.csv").write_text(
        "text,gold,pred\nhello,سلام,درود\n"
    )

    calls = []
    monkeypatch.setattr(
        combo_executor,
        "build_run_eval_cmd",
        lambda **kwargs: calls.append(kwargs) or ["python", "judge.py"],
    )

    with patch("leaderboard_runner.combo_executor.subprocess.run") as mock_run:
        combo_executor.run_single_combo(
            model="candidate",
            dataset="ds",
            n_rows=None,
            shots=0,
            workers=1,
            judge=True,
            judge_mode="both",
            judge_only=True,
        )

    assert mock_run.call_count == 2
    assert [call["prompt_template"] for call in calls] == [
        "ref.jinja2",
        "noref.jinja2",
    ]
    assert [call["out_csv"].parent.parent.name for call in calls] == [
        "ds_judge_reference",
        "ds_judge_no_reference",
    ]


def test_judge_only_requires_existing_candidate_result(tmp_path, monkeypatch):
    data_dir, models_dir, _ = setup_paths(tmp_path, monkeypatch)
    ds_dir = data_dir / "ds"
    ds_dir.mkdir()
    (ds_dir / "meta.yaml").write_text(
        "task: translation\njudge: true\n"
    )
    (models_dir / "candidate.yaml").write_text("name: candidate\n")

    with pytest.raises(FileNotFoundError, match="Candidate result missing"):
        combo_executor.run_single_combo(
            model="candidate",
            dataset="ds",
            n_rows=None,
            shots=0,
            workers=1,
            judge=True,
            judge_only=True,
        )
