import argparse
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


def test_runner_cli_combinations(monkeypatch):
    combo_module = types.ModuleType("leaderboard_runner.combo_executor")
    combo_module.run_single_combo = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "leaderboard_runner.combo_executor", combo_module)

    io_module = types.ModuleType("leaderboard_runner.io_utils")
    io_module.parse_csv_or_file = lambda arg: [x.strip() for x in arg.split(",") if x.strip()]
    monkeypatch.setitem(sys.modules, "leaderboard_runner.io_utils", io_module)

    from leaderboard_runner import cli

    run_mock = MagicMock()
    rebuild_mock = MagicMock()
    monkeypatch.setattr(cli, "run_single_combo", run_mock)
    monkeypatch.setattr(cli, "rebuild_leaderboard", rebuild_mock)

    args = argparse.Namespace(
        models="m1,m2",
        datasets="d1,d2",
        n_rows=None,
        shots=3,
        workers=1,
        judge=False,
        judge_model=None,
        judge_mode="reference",
        judge_only=False,
        battle=False,
        battle_only=False,
        battle_model_1=None,
        battle_model_2=None,
        battle_judge_model=None,
        league=False,
        league_only=False,
        league_name=None,
        league_models=None,
        league_datasets=None,
        league_judge_model=None,
        league_matches=10,
        league_rows_per_match=20,
        league_k_factor=32.0,
        league_initial_elo=1000.0,
        league_calibration_games=2,
        league_repeat_penalty=64.0,
        league_seed=42,
        dry=False,
        debug=False,
    )
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self, args_list=None: args)

    cli.main([])

    assert run_mock.call_count == 4
    combos = {(c.kwargs["model"], c.kwargs["dataset"]) for c in run_mock.call_args_list}
    assert combos == {("m1", "d1"), ("m1", "d2"), ("m2", "d1"), ("m2", "d2")}
    rebuild_mock.assert_called_once_with(dry_run=False)


def test_runner_cli_rejects_invalid_numeric_args(monkeypatch):
    from leaderboard_runner import cli

    with pytest.raises(SystemExit):
        cli.main(["-m", "m", "-d", "d", "-n", "0"])

    with pytest.raises(SystemExit):
        cli.main(["-m", "m", "-d", "d", "-s", "-1"])

    with pytest.raises(SystemExit):
        cli.main(["-m", "m", "-d", "d", "-w", "0"])

    with pytest.raises(SystemExit):
        cli.main(["-m", "m", "-d", "d", "--judge-only"])

    with pytest.raises(SystemExit):
        cli.main(["-m", "m", "-d", "d", "--battle-only"])

    with pytest.raises(SystemExit):
        cli.main(["-m", "m", "-d", "d", "--battle"])

    with pytest.raises(SystemExit):
        cli.main(["-m", "m", "-d", "d", "--league-only"])

    with pytest.raises(SystemExit):
        cli.main(["-m", "m", "-d", "d", "--league"])


def test_runner_cli_battle_only_uses_configured_pair(monkeypatch):
    from leaderboard_runner import cli

    battle_mock = MagicMock()
    rebuild_mock = MagicMock()
    combo_mock = MagicMock()
    monkeypatch.setattr(cli, "run_battle", battle_mock)
    monkeypatch.setattr(cli, "run_single_combo", combo_mock)
    monkeypatch.setattr(cli, "rebuild_leaderboard", rebuild_mock)

    cli.main(
        [
            "-m",
            "unused",
            "-d",
            "ds1,ds2",
            "--battle",
            "--battle-only",
            "--battle-model-1",
            "first",
            "--battle-model-2",
            "second",
            "--battle-judge-model",
            "judge",
        ]
    )

    combo_mock.assert_not_called()
    assert battle_mock.call_count == 2
    assert {
        (call.kwargs["dataset"], call.kwargs["model_1"], call.kwargs["model_2"])
        for call in battle_mock.call_args_list
    } == {("ds1", "first", "second"), ("ds2", "first", "second")}
    rebuild_mock.assert_called_once_with(dry_run=False)


def test_runner_cli_league_only_uses_separate_pool(monkeypatch):
    from leaderboard_runner import cli

    league_mock = MagicMock()
    combo_mock = MagicMock()
    rebuild_mock = MagicMock()
    monkeypatch.setattr(cli, "run_league", league_mock)
    monkeypatch.setattr(cli, "run_single_combo", combo_mock)
    monkeypatch.setattr(cli, "rebuild_leaderboard", rebuild_mock)

    cli.main(
        [
            "-m",
            "unused",
            "-d",
            "unused-dataset",
            "--league",
            "--league-only",
            "--league-name",
            "Demo",
            "--league-models",
            "m1,m2,m3",
            "--league-datasets",
            "d1,d2",
            "--league-judge-model",
            "judge",
            "--league-matches",
            "7",
            "--league-rows-per-match",
            "12",
        ]
    )

    combo_mock.assert_not_called()
    league_mock.assert_called_once()
    call = league_mock.call_args.kwargs
    assert call["name"] == "Demo"
    assert call["models"] == ["m1", "m2", "m3"]
    assert call["datasets"] == ["d1", "d2"]
    assert call["matches"] == 7
    assert call["rows_per_match"] == 12
    rebuild_mock.assert_called_once_with(dry_run=False)
