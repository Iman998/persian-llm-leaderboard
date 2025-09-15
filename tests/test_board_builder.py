import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from leaderboard_runner import board_builder, paths


def test_rebuild_leaderboard_failure(monkeypatch):
    calls = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        return SimpleNamespace(returncode=1, stderr="boom")

    monkeypatch.setattr(board_builder.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError):
        board_builder.rebuild_leaderboard(dry_run=False)

    expected_cmd = [
        sys.executable,
        str(paths.BUILD_BOARD_SCRIPT),
        "--results_dir",
        str(paths.RESULTS_DIR),
        "--datasets_dir",
        str(paths.DATASETS_DIR),
        "--out",
        str(paths.LEADERBOARD_OUT),
        "--lang",
        "all",
        "--exclude",
        "translat",
        "summar",
        "summary",
    ]

    assert calls[0] == expected_cmd
