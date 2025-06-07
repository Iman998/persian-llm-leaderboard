"""Accuracy metric (fraction of exact matches)."""


def compute(preds, labels):
    """Simple accuracy.

    Parameters
    ----------
    preds : list or pandas.Series
        Model predictions.
    labels : list or pandas.Series
        Ground truth labels.
    """

    preds = list(preds)
    labels = list(labels)

    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(labels) if len(labels) > 0 else 0.0
