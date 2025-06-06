"""Recall score metric."""

from sklearn.metrics import recall_score


def compute(preds, labels, **kwargs):
    """Compute recall score.

    Parameters
    ----------
    preds : list or pandas.Series
        Model predictions.
    labels : list or pandas.Series
        Ground truth labels.
    **kwargs : Any
        Additional keyword arguments passed to ``sklearn.metrics.recall_score``.

    Returns
    -------
    float
        Recall score.
    """
    preds = list(preds)
    labels = list(labels)
    return recall_score(labels, preds, **kwargs) if len(labels) > 0 else 0.0
