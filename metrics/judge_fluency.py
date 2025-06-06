import re

# Extract numbers following a 'fluency' label like "fluency: 7.5".
# Predictions are expected to contain a substring of the form
# "fluency: <number>" or "fluency=<number>" (case-insensitive).
_FLU_RE = re.compile(r"fluency\s*[:=]\s*(\d+(?:\.\d+)?)", re.I)


def compute(preds, _labels):
    """Return the average fluency score found in ``preds``."""
    scores = []
    for p in preds:
        m = _FLU_RE.search(str(p))
        if m:
            scores.append(float(m.group(1)))
    return sum(scores) / len(scores) if scores else 0.0
