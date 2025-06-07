"""Token-based BLEU score implementation.

This module implements a lightweight version of BLEU used to
evaluate translation and text generation quality. The :func:`compute`
function returns the corpus-level BLEU score over a sequence of
predictions and reference texts.
"""

from collections import Counter
from math import exp, log


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _bleu(pred: str, label: str, max_n: int = 4) -> float:
    p_tokens = pred.split()
    l_tokens = label.split()
    if not p_tokens or not l_tokens:
        return 0.0
    weights = [1 / max_n] * max_n
    log_scores = []
    for n in range(1, max_n + 1):
        p_counts = _ngram_counts(p_tokens, n)
        l_counts = _ngram_counts(l_tokens, n)
        overlap = sum(min(p_counts[k], l_counts[k]) for k in p_counts)
        total = max(sum(p_counts.values()), 1)
        score = overlap / total
        if score == 0:
            return 0.0
        log_scores.append(log(score))
    geo_mean = exp(sum(w * s for w, s in zip(weights, log_scores)))
    bp = 1.0 if len(p_tokens) > len(l_tokens) else exp(1 - len(l_tokens) / len(p_tokens))
    return bp * geo_mean


def compute(preds, labels):
    """Return the corpus-level BLEU score for *preds* and *labels*."""
    scores = [_bleu(str(p), str(l)) for p, l in zip(preds, labels)]
    return sum(scores) / len(scores) if scores else 0.0
