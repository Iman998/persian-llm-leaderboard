def _to_bool(x):
    s = str(x).strip().lower()
    return 1 if s in {"1", "true", "yes", "positive", "toxic", "y"} else 0


def compute(preds, labels):
    preds = [_to_bool(p) for p in preds]
    labels = [_to_bool(l) for l in labels]
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    denom = tp + fn
    return tp / denom if denom else 0.0
