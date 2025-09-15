import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "tpr", Path(__file__).resolve().parents[1] / "metrics" / "tpr.py"
)
tpr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tpr)


def test_tpr_sanity():
    preds = [1, 0, 1, 0]
    labels = [1, 1, 0, 0]
    assert tpr.compute(preds, labels) == 0.5


def test_tpr_no_positive_labels():
    preds = [0, 1, 0]
    labels = [0, 0, 0]
    assert tpr.compute(preds, labels) == 0.0


def test_tpr_empty_lists():
    assert tpr.compute([], []) == 0.0


def test_tpr_string_labels():
    preds = ["yes", "no", "yes"]
    labels = ["yes", "yes", "no"]
    assert tpr.compute(preds, labels) == 0.5
