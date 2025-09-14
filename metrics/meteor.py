"""Token-based METEOR score implementation.

This module implements a lightweight version of METEOR used to
evaluate translation and text generation quality. The :func:`compute`
function returns the corpus-level METEOR score over a sequence of
predictions and reference texts.
"""

from collections import Counter


def _meteor(pred: str, label: str) -> float:
    p_tokens = pred.split()
    l_tokens = label.split()
    if not p_tokens or not l_tokens:
        return 0.0
    p_counts = Counter(p_tokens)
    l_counts = Counter(l_tokens)
    m = sum((p_counts & l_counts).values())
    if m == 0:
        return 0.0
    precision = m / len(p_tokens)
    recall = m / len(l_tokens)
    f_mean = (10 * precision * recall) / (recall + 9 * precision) if (recall + 9 * precision) else 0.0
    positions = {}
    for i, tok in enumerate(p_tokens):
        positions.setdefault(tok, []).append(i)
    used = {tok: 0 for tok in positions}
    matches = []
    for j, tok in enumerate(l_tokens):
        if tok in positions and used[tok] < len(positions[tok]):
            idx = positions[tok][used[tok]]
            used[tok] += 1
            matches.append((idx, j))
    matches.sort()
    ch = 0
    if matches:
        ch = 1
        for i in range(1, len(matches)):
            prev = matches[i - 1]
            curr = matches[i]
            if curr[0] != prev[0] + 1 or curr[1] != prev[1] + 1:
                ch += 1
    penalty = 0.5 * (ch / m) ** 3
    return f_mean * (1 - penalty)


def compute(preds, labels):
    """Return the corpus-level METEOR score for *preds* and *labels*."""
    scores = [_meteor(str(p), str(l)) for p, l in zip(preds, labels)]
    return sum(scores) / len(scores) if scores else 0.0
