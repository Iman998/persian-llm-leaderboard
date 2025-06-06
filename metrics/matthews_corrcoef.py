"""Matthews correlation coefficient metric."""

from sklearn.metrics import matthews_corrcoef


def compute(preds, labels, **kwargs):
    """Compute Matthews correlation coefficient.

    Parameters
    ----------
    preds : list or pandas.Series
        Model predictions.
    labels : list or pandas.Series
        Ground truth labels.
    **kwargs : Any
        Additional keyword arguments passed to ``sklearn.metrics.matthews_corrcoef``.

    Returns
    -------
    float
        Matthews correlation coefficient.
    """
    preds = list(preds)
    labels = list(labels)
    return matthews_corrcoef(labels, preds, **kwargs) if len(labels) > 0 else 0.0
