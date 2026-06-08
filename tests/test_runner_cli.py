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
