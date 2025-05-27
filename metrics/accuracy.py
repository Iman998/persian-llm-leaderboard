def compute(preds, labels):
    """Simple accuracy."""
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(labels) if labels else 0.0
