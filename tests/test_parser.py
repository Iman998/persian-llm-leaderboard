from pathlib import Path
import sys
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import app.core.parser as parser


def test_parse_file_new_scheme(tmp_path, monkeypatch):
    monkeypatch.setattr(parser, "RESULTS_DIR", tmp_path)
    assert (
        parser.parse_file(tmp_path / "ds" / "Qwen2.5-7B-Instruct.csv")
        == ("ds", "Qwen2.5-7B-Instruct", "")
    )


def test_parse_file_legacy_scheme():
    assert parser.parse_file(Path("legacy_ds_Qwen2.5-7B-Instruct.csv")) == (
        "legacy_ds",
        "Qwen2.5-7B-Instruct",
        "",
    )


def test_parse_file_invalid_filenames():
    assert parser.parse_file(Path("ds_Qwen2.5-7B-Instruct_123.csv")) is None
    assert parser.parse_file(Path("unknown_model.csv")) is None
