from metrics.math_equivalence import compute


def test_math_equivalence_metric():
    preds = [r"\dfrac{1}{2}", r"\sqrt3", "(0, 5)"]
    labels = ["0.5", r"\sqrt{3}", "(0,5)"]
    assert compute(preds, labels) == 1.0


def test_math_equivalence_metric_empty_labels():
    assert compute([], []) == 0.0
