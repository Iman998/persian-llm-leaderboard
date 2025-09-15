import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from leaderboard_lib import leaderboard, llm_judge_board


@pytest.mark.parametrize("module", [leaderboard, llm_judge_board])
def test_accuracy(module):
    df_match = pd.DataFrame({"pred": ["a", "b"], "ans": ["a", "b"]})
    assert module.accuracy(df_match, "ans") == 100.0

    df_mismatch = pd.DataFrame({"pred": ["a", "b"], "ans": ["c", "d"]})
    assert module.accuracy(df_mismatch, "ans") == 0.0


@pytest.mark.parametrize("module", [leaderboard, llm_judge_board])
def test_load_metric_success(module, tmp_path, monkeypatch):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "dummy.py").write_text(
        "def compute(preds, labels):\n    return 0.5\n"
    )
    monkeypatch.chdir(tmp_path)
    metric_fn = module.load_metric("dummy")
    assert metric_fn([1], [1]) == 0.5


@pytest.mark.parametrize("module", [leaderboard, llm_judge_board])
def test_load_metric_file_not_found(module, tmp_path, monkeypatch):
    (tmp_path / "metrics").mkdir()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        module.load_metric("missing")


@pytest.mark.parametrize("module", [leaderboard, llm_judge_board])
def test_load_metric_missing_compute(module, tmp_path, monkeypatch):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "bad.py").write_text("x = 1\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(AttributeError):
        module.load_metric("bad")
