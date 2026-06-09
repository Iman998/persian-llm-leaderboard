"""MATH benchmark answer equivalence."""

from leaderboard_lib.math_utils import math_answers_equivalent


def compute(preds, labels):
    """Return the fraction of predictions equivalent to their labels."""
    preds = list(preds)
    labels = list(labels)
    if not labels:
        return 0.0

    correct = sum(
        math_answers_equivalent(pred, label)
        for pred, label in zip(preds, labels)
    )
    return correct / len(labels)
