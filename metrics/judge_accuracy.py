import re

# Extract numbers following an 'accuracy' label like "accuracy: 8".
# Predictions are expected to include "accuracy: <number>" or
# "accuracy=<number>" (case-insensitive).
_ACC_RE = re.compile(r"accuracy\s*[:=]\s*(\d+(?:\.\d+)?)", re.I)


def compute(preds, _labels):
    """Return the average accuracy score extracted from ``preds``."""
    scores = []
    for p in preds:
        m = _ACC_RE.search(str(p))
        if m:
            scores.append(float(m.group(1)))
    return sum(scores) / len(scores) if scores else 0.0
