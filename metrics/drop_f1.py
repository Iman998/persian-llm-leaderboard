"""Official DROP generalized F1 metric."""

from leaderboard_lib.drop_utils import best_drop_metrics


def compute(preds, labels):
    """Return mean best F1 over validated DROP annotations."""
    preds = list(preds)
    labels = list(labels)
    if not labels:
        return 0.0
    return sum(
        best_drop_metrics(prediction, annotations)[1]
        for prediction, annotations in zip(preds, labels)
    ) / len(labels)
