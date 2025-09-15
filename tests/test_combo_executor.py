import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import sys
import types
sys.path.append(str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault("pandas", types.ModuleType("pandas"))
yaml_module = types.SimpleNamespace(safe_load=lambda s: {})
sys.modules.setdefault("yaml", yaml_module)

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
