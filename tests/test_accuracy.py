import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "accuracy", Path(__file__).resolve().parents[1] / "metrics" / "accuracy.py"
)
accuracy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accuracy)


def test_accuracy_perfect_matches():
    assert accuracy.compute(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_accuracy_all_mismatches():
    assert accuracy.compute(["a", "b"], ["c", "d"]) == 0.0


def test_accuracy_empty_lists():
    assert accuracy.compute([], []) == 0.0


def test_accuracy_length_mismatch_denominator_len_labels():
    assert accuracy.compute([1, 2], [1, 2, 3]) == pytest.approx(2 / 3)


def test_accuracy_handles_strings_and_numbers():
    preds_num = [1, 0, 1]
    labels_num = [1, 1, 0]
    preds_str = ["1", "0", "1"]
    labels_str = ["1", "1", "0"]
    result_num = accuracy.compute(preds_num, labels_num)
    result_str = accuracy.compute(preds_str, labels_str)
    assert result_num == result_str == pytest.approx(1 / 3)
