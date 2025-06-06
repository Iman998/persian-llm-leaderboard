def compute(preds, labels):
    """Exact string match accuracy."""
    correct = sum(str(p).strip() == str(l).strip() for p, l in zip(preds, labels))
    return correct / len(labels) if labels else 0.0
