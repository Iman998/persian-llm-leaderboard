from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from metrics.llm_judge_score import compute
import pytest


def test_numeric_list_average():
    assert compute([1, 2, 3], []) == 2.0


def test_string_numbers_average():
    assert compute(["1", "2", "3"], []) == 2.0


def test_empty_predictions():
    assert compute([], []) == 0.0


def test_non_numeric_input_raises():
    with pytest.raises(ValueError):
        compute(["x"], [])
