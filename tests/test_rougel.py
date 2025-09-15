import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "rougel", Path(__file__).resolve().parents[1] / "metrics" / "rougel.py"
)
rougel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rougel)


def test_rougel_identical():
    assert rougel.compute(["سلام دنیا"], ["سلام دنیا"]) == 1.0


def test_rougel_different():
    assert rougel.compute(["abc"], ["xyz"]) == 0.0


def test_rougel_empty_predictions():
    assert rougel.compute([], []) == 0.0


def test_rougel_label_shorter_than_prediction():
    score = rougel.compute(["the cat sat"], ["the cat"])
    assert score == pytest.approx(0.8)


def test_rougel_mismatched_lengths():
    assert rougel.compute(["hello", "world"], ["hello"]) == 1.0


def test_rougel_non_string_inputs():
    assert rougel.compute([123], [123]) == 1.0
