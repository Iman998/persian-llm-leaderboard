from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from leaderboard_runner import league_executor, paths


def _setup(tmp_path, monkeypatch, *, task="translation"):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    dataset_dir = data_dir / "ds"
    dataset_dir.mkdir(parents=True)
    models_dir.mkdir()
    results_dir.mkdir()
    (dataset_dir / "meta.yaml").write_text(
        f"task: {task}\n"
        "question_col: question\n"
        "answer_col: answer\n"
        "battle:\n"
        "  enabled: true\n"
        "  model: judge\n"
        "  prompt_template: prompts/battle.jinja2\n"
    )
    for model in ("m1", "m2", "judge"):
        (models_dir / f"{model}.yaml").write_text(f"model: {model}\n")
    for model, predictions in (
        ("m1", ["a", "b", "c", "d"]),
        ("m2", ["w", "x", "y", "z"]),
    ):
        model_dir = results_dir / "ds" / model
        model_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "Question Body": ["q1", "q2", "q3", "q4"],
                "answer": ["r1", "r2", "r3", "r4"],
                "pred": predictions,
            }
        ).to_csv(model_dir / f"{model}.csv", index=False)
    monkeypatch.setattr(paths, "DATASETS_DIR", data_dir)
    monkeypatch.setattr(paths, "MODELS_DIR", models_dir)
    monkeypatch.setattr(paths, "RESULTS_DIR", results_dir)
    return results_dir


def test_sample_match_rows_rotates_before_repeating():
    frame = pd.DataFrame({"value": list(range(5))})

    first, first_cycle = league_executor._sample_match_rows(
        frame,
        league_seed=7,
        dataset="ds",
        model_1="m1",
        model_2="m2",
        rows_per_match=2,
        prior_pair_dataset_matches=0,
    )
    second, second_cycle = league_executor._sample_match_rows(
        frame,
        league_seed=7,
        dataset="ds",
        model_1="m2",
        model_2="m1",
        rows_per_match=2,
        prior_pair_dataset_matches=1,
    )

    assert set(first["league_source_row"]).isdisjoint(
        set(second["league_source_row"])
    )
    assert first_cycle == second_cycle == 0


def test_run_league_persists_and_continues_named_standings(
    tmp_path, monkeypatch
):
    results_dir = _setup(tmp_path, monkeypatch)
    sampled_rows = []

    def build_command(**kwargs):
        sampled_rows.append(
            pd.read_csv(kwargs["dataset_path"])["league_source_row"].tolist()
        )
        return ["fake-league-eval", str(kwargs["out_csv"])]

    def run_command(command, check):
        assert check is True
        pd.DataFrame({"pred": ["model_1", "model_1"]}).to_csv(
            Path(command[-1]), index=False
        )

    monkeypatch.setattr(league_executor, "build_run_eval_cmd", build_command)
    monkeypatch.setattr(league_executor.subprocess, "run", run_command)

    league_dir = league_executor.run_league(
        name="Test League",
        models=["m1", "m2"],
        datasets=["ds"],
        judge_model="judge",
        matches=1,
        rows_per_match=2,
        workers=1,
        seed=11,
    )
    league_executor.run_league(
        name="Test League",
        models=["m1", "m2"],
        datasets=["ds"],
        judge_model=None,
        matches=1,
        rows_per_match=999,
        workers=1,
        seed=999,
    )

    assert league_dir == results_dir / "league" / "test-league"
    standings = pd.read_csv(league_dir / "standings.csv")
    history = pd.read_csv(league_dir / "history.csv")
    assert len(history) == 2
    assert standings["Games"].tolist() == [2, 2]
    assert standings.iloc[0]["Elo"] > standings.iloc[1]["Elo"]
    assert set(sampled_rows[0]).isdisjoint(set(sampled_rows[1]))
    assert len(list((league_dir / "matches").glob("*.csv"))) == 2


def test_run_league_rejects_non_generation_dataset(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, task="multiple_choice")

    with pytest.raises(ValueError, match="not a generation dataset"):
        league_executor.run_league(
            name="Bad League",
            models=["m1", "m2"],
            datasets=["ds"],
            judge_model="judge",
            matches=1,
            rows_per_match=2,
            workers=1,
        )


def test_run_league_rejects_disconnected_model_pool(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    (paths.MODELS_DIR / "m3.yaml").write_text("model: m3\n")

    with pytest.raises(ValueError, match="pool is disconnected"):
        league_executor.run_league(
            name="Disconnected League",
            models=["m1", "m2", "m3"],
            datasets=["ds"],
            judge_model="judge",
            matches=1,
            rows_per_match=2,
            workers=1,
        )
