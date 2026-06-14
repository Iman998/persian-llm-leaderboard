from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from leaderboard_runner import battle_executor, paths


def _setup(tmp_path, monkeypatch):
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


def test_align_results_reorders_matching_rows():
    first = pd.DataFrame(
        {"Question Body": ["q1", "q2"], "answer": ["a1", "a2"], "pred": ["x", "y"]}
    )
    second = pd.DataFrame(
        {"Question Body": ["q2", "q1"], "answer": ["a2", "a1"], "pred": ["v", "u"]}
    )

    left, right = battle_executor._align_results(
        first,
        second,
        meta_cfg={"question_col": "question", "answer_col": "answer"},
    )

    assert left["question"].tolist() == ["q1", "q2"]
    assert right["pred"].tolist() == ["u", "v"]


def test_run_battle_reuses_two_existing_results(tmp_path, monkeypatch):
    data_dir, models_dir, results_dir = _setup(tmp_path, monkeypatch)
    dataset_dir = data_dir / "ds"
    dataset_dir.mkdir()
    (dataset_dir / "meta.yaml").write_text(
        "question_col: question\n"
        "answer_col: answer\n"
        "battle:\n"
        "  enabled: true\n"
        "  model: judge\n"
        "  prompt_template: prompts/battle.jinja2\n"
    )
    for model in ("first", "second", "judge"):
        (models_dir / f"{model}.yaml").write_text(f"model: {model}\n")
    for model, predictions in (("first", ["x", "y"]), ("second", ["u", "v"])):
        output_dir = results_dir / "ds" / model
        output_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "Question Body": ["q1", "q2"],
                "answer": ["a1", "a2"],
                "pred": predictions,
            }
        ).to_csv(output_dir / f"{model}.csv", index=False)

    calls = []

    def capture(**kwargs):
        meta = yaml.safe_load(Path(kwargs["meta_path"]).read_text())
        battle_input = pd.read_csv(kwargs["dataset_path"])
        assert meta["model_1_name"] == "first"
        assert meta["model_2_name"] == "second"
        assert battle_input["model_1_output"].tolist() == ["x", "y"]
        assert battle_input["model_2_output"].tolist() == ["u", "v"]
        calls.append(kwargs)
        return ["python", "battle.py"]

    monkeypatch.setattr(battle_executor, "build_run_eval_cmd", capture)
    with patch("leaderboard_runner.battle_executor.subprocess.run") as run_mock:
        battle_executor.run_battle(
            dataset="ds",
            model_1="first",
            model_2="second",
            judge_model=None,
            n_rows=None,
            workers=2,
        )

    assert run_mock.call_count == 1
    assert calls[0]["model_stub"] == "judge"
    assert calls[0]["out_csv"] == (
        results_dir / "battle" / "ds" / "first__vs__second" / "battle.csv"
    )


def test_run_battle_requires_existing_candidate_results(tmp_path, monkeypatch):
    data_dir, models_dir, _ = _setup(tmp_path, monkeypatch)
    dataset_dir = data_dir / "ds"
    dataset_dir.mkdir()
    (dataset_dir / "meta.yaml").write_text("battle: true\n")
    for model in ("first", "second", "judge"):
        (models_dir / f"{model}.yaml").write_text(f"model: {model}\n")

    with pytest.raises(FileNotFoundError, match="candidate result"):
        battle_executor.run_battle(
            dataset="ds",
            model_1="first",
            model_2="second",
            judge_model="judge",
            n_rows=None,
            workers=1,
        )
