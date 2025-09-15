import importlib.util
from pathlib import Path
import pytest

pytest.importorskip("sklearn")

spec = importlib.util.spec_from_file_location(
    "recall", Path(__file__).resolve().parents[1] / "metrics" / "recall.py",
)
recall = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recall)


def test_recall_perfect():
    preds = [1, 0, 1]
    labels = [1, 0, 1]
    assert recall.compute(preds, labels) == 1.0


def test_recall_zero():
    preds = [0, 0, 0]
    labels = [1, 1, 1]
    assert recall.compute(preds, labels) == 0.0


def test_recall_empty_lists():
    assert recall.compute([], []) == 0.0


def test_recall_no_actual_positives():
    preds = [1, 0, 1]
    labels = [0, 0, 0]
    assert recall.compute(preds, labels) == 0.0
