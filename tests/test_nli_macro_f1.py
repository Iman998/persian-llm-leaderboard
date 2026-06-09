import math

from metrics.nli_macro_f1 import compute


def test_nli_macro_f1_perfect_predictions():
    labels = ["entailment", "contradiction", "neutral"]
    assert compute(labels, labels) == 1.0


def test_nli_macro_f1_weights_classes_equally():
    labels = ["entailment", "contradiction", "neutral"]
    predictions = ["entailment", "contradiction", "contradiction"]

    assert math.isclose(compute(predictions, labels), (1.0 + 2 / 3) / 3)


def test_nli_macro_f1_handles_empty_and_invalid_predictions():
    assert compute([], []) == 0.0
    assert math.isclose(
        compute([None, "contradiction"], ["entailment", "contradiction"]),
        1.0 / 3,
    )
