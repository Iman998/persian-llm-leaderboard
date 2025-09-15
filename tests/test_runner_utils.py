import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from leaderboard_runner.io_utils import parse_csv_or_file, sample_csv
from leaderboard_runner.cmd_utils import build_run_eval_cmd
from leaderboard_runner.meta_utils import load_meta_fields
from leaderboard_runner.paths import RUN_EVAL_SCRIPT, MODELS_DIR


def test_parse_csv_or_file_handles_strings_and_files(tmp_path):
    comma_result = parse_csv_or_file("a,b , c")
    assert comma_result == ["a", "b", "c"]

    file = tmp_path / "items.txt"
    file.write_text("foo\nbar\n\n baz \n")
    file_result = parse_csv_or_file(str(file))
    assert file_result == ["foo", "bar", "baz"]

    directory = tmp_path / "dir"
    directory.mkdir()
    with pytest.raises(FileNotFoundError):
        parse_csv_or_file(str(directory))

    with pytest.raises(FileNotFoundError):
        parse_csv_or_file(str(tmp_path / "missing.csv"))


def test_sample_csv_preserves_header_and_row_count(tmp_path):
    src = tmp_path / "data.csv"
    src.write_text("col1,col2\n1,a\n2,b\n3,c\n4,d\n5,e\n")

    out = sample_csv(src, 3)
    try:
        lines = out.read_text().splitlines()
        assert lines[0] == "col1,col2"
        assert len(lines) == 4  # header + 3 rows
    finally:
        out.unlink()


def test_build_run_eval_cmd(tmp_path):
    dataset = tmp_path / "data.csv"
    meta = tmp_path / "meta.yaml"
    out = tmp_path / "out.csv"

    common = [
        sys.executable,
        RUN_EVAL_SCRIPT,
        "--dataset",
        dataset,
        "--meta",
        meta,
        "--model",
        MODELS_DIR / "m.yaml",
        "--prompt",
        "prompts/p.jinja2",
        "--evaluator",
        "evaluators/e.py",
        "--shots",
        "5",
        "--workers",
        "2",
        "--out",
        out,
    ]

    cmd = build_run_eval_cmd(
        model_stub="m",
        dataset_path=dataset,
        meta_path=meta,
        prompt_template="prompts/p.jinja2",
        evaluator="evaluators/e.py",
        shots=5,
        workers=2,
        n_rows=None,
        out_csv=out,
    )
    assert cmd == common

    cmd_rows = build_run_eval_cmd(
        model_stub="m",
        dataset_path=dataset,
        meta_path=meta,
        prompt_template="prompts/p.jinja2",
        evaluator="evaluators/e.py",
        shots=5,
        workers=2,
        n_rows=10,
        out_csv=out,
    )
    assert cmd_rows == common + ["--n_rows", "10"]


def test_load_meta_fields_defaults(tmp_path):
    meta = tmp_path / "meta.yaml"
    meta.write_text("other: value\n")

    prompt_template, evaluator = load_meta_fields(meta)
    assert prompt_template == "prompts/mcq_fewshot.jinja2"
    assert evaluator == "evaluators/mcq_evaluator.py"
