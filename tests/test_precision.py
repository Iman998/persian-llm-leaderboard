import importlib.util
from pathlib import Path
import pytest

pytest.importorskip("sklearn")

spec = importlib.util.spec_from_file_location(
    "precision", Path(__file__).resolve().parents[1] / "metrics" / "precision.py"
)
precision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precision)


def test_precision_single_positive():
    assert precision.compute([1], [1]) == 1.0


def test_precision_single_negative():
    assert precision.compute([1], [0]) == 0.0


def test_precision_empty_lists():
    assert precision.compute([], []) == 0.0


def test_precision_no_predicted_positives():
    assert precision.compute([0, 0], [1, 0]) == 0.0
