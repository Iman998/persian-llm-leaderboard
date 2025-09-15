import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "f1", Path(__file__).resolve().parents[1] / "metrics" / "f1.py"
)
f1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f1)


def test_identical_tokens():
    assert f1.compute(["abc def"], ["abc def"]) == 1.0


def test_disjoint_tokens():
    assert f1.compute(["abc"], ["def"]) == 0.0


def test_empty_prediction():
    assert f1.compute([""], ["a b"]) == 0.0


def test_empty_reference():
    assert f1.compute(["a b"], [""]) == 0.0


def test_empty_lists():
    assert f1.compute([], []) == 0.0


def test_uneven_lengths():
    assert f1.compute(["a"], ["a", "b"]) == 1.0
    assert f1.compute(["a", "b"], ["a"]) == 1.0


def test_numeric_and_string_tokenization():
    assert f1.compute([1, "2"], ["1", 2]) == 1.0
