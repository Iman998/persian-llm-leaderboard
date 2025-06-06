def _to_bool(x):
    s = str(x).strip().lower()
    return 1 if s in {"1", "true", "yes", "positive", "toxic", "y"} else 0


def compute(preds, _labels):
    preds = [_to_bool(p) for p in preds]
    return sum(preds) / len(preds) if preds else 0.0
