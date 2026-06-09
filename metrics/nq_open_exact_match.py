"""Official NQ-Open normalized exact-match accuracy."""

from leaderboard_lib.natural_questions_utils import nq_exact_match


def compute(preds, labels):
    """Return mean exact match against all accepted NQ-Open answers."""
    preds = list(preds)
    labels = list(labels)
    if not labels:
        return 0.0
    return sum(
        nq_exact_match(prediction, answers)
        for prediction, answers in zip(preds, labels)
    ) / len(labels)
