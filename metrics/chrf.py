"""Character n-gram F-score (chrF) implementation.

This module provides a lightweight computation of chrF, a character-level
metric used to evaluate text generation quality. The :func:`compute`
function returns the average chrF score over a sequence of predictions and
reference texts.
"""

from collections import Counter


def _ngram_counts(chars: list[str], n: int) -> Counter:
    return Counter(tuple(chars[i : i + n]) for i in range(len(chars) - n + 1))


def _chrf(pred: str, label: str, max_n: int = 6, beta: int = 2) -> float:
    p_chars = list(pred)
    l_chars = list(label)
    if not p_chars or not l_chars:
        return 0.0
    beta2 = beta ** 2
    total_p = 0.0
    total_r = 0.0
    for n in range(1, max_n + 1):
        p_counts = _ngram_counts(p_chars, n)
        l_counts = _ngram_counts(l_chars, n)
        overlap = sum(min(p_counts[k], l_counts[k]) for k in p_counts)
        p_total = max(sum(p_counts.values()), 1)
        l_total = max(sum(l_counts.values()), 1)
        total_p += overlap / p_total
        total_r += overlap / l_total
    precision = total_p / max_n
    recall = total_r / max_n
    if precision == 0 or recall == 0:
        return 0.0
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def compute(preds, labels):
    """Return the average chrF score for *preds* and *labels*."""
    scores = [_chrf(str(p), str(l)) for p, l in zip(preds, labels)]
    return sum(scores) / len(scores) if scores else 0.0
