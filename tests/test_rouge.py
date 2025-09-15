import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "rouge", Path(__file__).resolve().parents[1] / "metrics" / "rouge.py"
)
rouge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rouge)


def test_rouge1_known():
    preds = ["a b c d"]
    labels = ["a b e f"]
    assert rouge.compute_rouge_n(preds, labels, 1) == pytest.approx(0.5)


def test_rouge2_known():
    preds = ["a b c d"]
    labels = ["a b e f"]
    assert rouge.compute_rouge_n(preds, labels, 2) == pytest.approx(1 / 3, rel=1e-6)


def test_sequences_shorter_than_n():
    preds = ["word"]
    labels = ["word"]
    assert rouge.compute_rouge_n(preds, labels, 2) == 0.0


def test_empty_lists():
    assert rouge.compute_rouge_n([], [], 1) == 0.0


def test_mismatched_lengths():
    preds = ["a b", "ignored"]
    labels = ["a b"]
    assert rouge.compute_rouge_n(preds, labels, 1) == 1.0


def test_numeric_inputs_stringified():
    assert rouge.compute_rouge_n([123], [123], 1) == 1.0
    assert rouge.compute_rouge_n([123], [456], 1) == 0.0
