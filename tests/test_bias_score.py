import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "bias_score", Path(__file__).resolve().parents[1] / "metrics" / "bias_score.py"
)
bias_score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bias_score)


def test_low_bias_balanced():
    preds = [1, 0, 1, 0]
    labels = [1, 0, 1, 0]
    assert bias_score.compute(preds, labels) == 0.0


def test_high_bias_skewed():
    preds = [1, 1, 1, 1]
    labels = [1, 0, 1, 0]
    assert bias_score.compute(preds, labels) == 1.0


def test_empty_lists():
    assert bias_score.compute([], []) == 1.0


def test_no_negatives():
    preds = [1, 1]
    labels = [1, 1]
    assert bias_score.compute(preds, labels) == 0.0


def test_no_positives():
    preds = [0, 0]
    labels = [0, 0]
    assert bias_score.compute(preds, labels) == 1.0


def test_string_numeric_labels():
    preds = ["yes", "no", "yes"]
    labels_numeric = [1, 0, 0]
    labels_string = ["yes", "no", "no"]
    assert bias_score.compute(preds, labels_numeric) == 0.5
    assert bias_score.compute(preds, labels_string) == 0.5
