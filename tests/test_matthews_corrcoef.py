import importlib
import math
import pytest

pytest.importorskip("sklearn")

mcc = importlib.import_module("sklearn.metrics").matthews_corrcoef


def test_mcc_known_case():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 1]
    assert mcc(y_true, y_pred) == pytest.approx(1 / math.sqrt(3))


def test_mcc_empty_inputs():
    with pytest.raises(ValueError):
        mcc([], [])


def test_mcc_mismatched_lengths():
    with pytest.raises(ValueError):
        mcc([0, 1], [0])

