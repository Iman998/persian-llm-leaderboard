import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "chrf", Path(__file__).resolve().parents[1] / "metrics" / "chrf.py"
)
chrf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chrf)


def test_chrf_identical():
    assert chrf.compute(["سلام دنیا"], ["سلام دنیا"]) == 1.0


def test_chrf_different():
    assert chrf.compute(["abc"], ["xyz"]) == 0.0

def test_chrf_empty_lists():
    assert chrf.compute([], []) == 0.0


def test_chrf_unequal_lengths_overlapping():
    assert chrf.compute(["abcdef", "xyz"], ["abcdef"]) == 1.0


def test_chrf_non_string_inputs():
    assert chrf.compute([123456], [123456]) == 1.0
    assert chrf.compute([123456], [789000]) == 0.0


def test_chrf_empty_prediction_or_reference():
    assert chrf.compute(["abc"], []) == 0.0
    assert chrf.compute([], ["abc"]) == 0.0
    assert chrf.compute([""], ["abc"]) == 0.0
    assert chrf.compute(["abc"], [""]) == 0.0
