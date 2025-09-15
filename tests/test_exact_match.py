import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "exact_match", Path(__file__).resolve().parents[1] / "metrics" / "exact_match.py"
)
exact_match = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exact_match)


def test_exact_match_perfect_match():
    assert exact_match.compute(["foo", "bar"], ["foo", "bar"]) == 1.0


def test_exact_match_mismatch():
    assert exact_match.compute(["foo"], ["bar"]) == 0.0


def test_exact_match_trimming():
    preds = [" foo ", "bar"]
    labels = ["foo", " bar "]
    assert exact_match.compute(preds, labels) == 1.0


def test_exact_match_empty_lists():
    assert exact_match.compute([], []) == 0.0


def test_exact_match_preds_shorter_than_labels():
    assert exact_match.compute(["foo"], ["foo", "bar"]) == 0.5


def test_exact_match_preds_longer_than_labels():
    assert exact_match.compute(["foo", "bar"], ["foo"]) == 1.0


def test_exact_match_numeric_vs_string():
    preds = [1, 2]
    labels = ["1", "2"]
    assert exact_match.compute(preds, labels) == 1.0
