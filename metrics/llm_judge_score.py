"""Aggregated score for LLM-as-judge outputs."""


def compute(preds, labels):
    """Return the mean of predictions converted to float."""
    values = [float(p) for p in preds]
    return sum(values) / len(values) if values else 0.0

