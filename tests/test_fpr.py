import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "fpr", Path(__file__).resolve().parents[1] / "metrics" / "fpr.py"
)
fpr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fpr)


def test_fpr_sanity():
    preds = [1, 1, 0, 0]
    labels = [0, 0, 0, 0]
    assert fpr.compute(preds, labels) == 0.5


def test_fpr_no_negative_labels():
    preds = [1, 0]
    labels = [1, 1]
    assert fpr.compute(preds, labels) == 0.0


def test_fpr_empty_lists():
    assert fpr.compute([], []) == 0.0


def test_fpr_type_robustness():
    preds = [True, "yes", 0, "toxic", "no"]
    labels = [False, 0, "no", 0, 0]
    assert fpr.compute(preds, labels) == pytest.approx(0.6)

