def compute(preds, labels):
    """Exact string match accuracy.

    Parameters
    ----------
    preds : list or pandas.Series
        Model predictions.
    labels : list or pandas.Series
        Ground truth labels.
    """

    preds = list(preds)
    labels = list(labels)

    correct = sum(str(p).strip() == str(l).strip() for p, l in zip(preds, labels))
    return correct / len(labels) if len(labels) > 0 else 0.0
