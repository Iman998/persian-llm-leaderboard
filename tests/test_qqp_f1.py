from metrics.qqp_f1 import compute


def test_qqp_f1_for_positive_paraphrase_class():
    assert compute(["1", "0", "1", "0"], ["1", "1", "0", "0"]) == 0.5


def test_qqp_f1_handles_empty_and_no_positive_predictions():
    assert compute([], []) == 0.0
    assert compute(["0", "0"], ["1", "0"]) == 0.0
    assert compute([None, "1"], ["1", "0"]) == 0.0
