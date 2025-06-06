"""Precision score metric."""

from sklearn.metrics import precision_score


def compute(preds, labels, **kwargs):
    """Compute precision score.

    Parameters
    ----------
    preds : list or pandas.Series
        Model predictions.
    labels : list or pandas.Series
        Ground truth labels.
    **kwargs : Any
        Additional keyword arguments passed to ``sklearn.metrics.precision_score``.

    Returns
    -------
    float
        Precision score.
    """
    preds = list(preds)
    labels = list(labels)
    return precision_score(labels, preds, **kwargs) if len(labels) > 0 else 0.0
