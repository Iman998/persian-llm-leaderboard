"""ROUGE-N metric implementation."""

from collections import Counter


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _rouge_n(pred: str, label: str, n: int) -> float:
    p_tokens = pred.split()
    l_tokens = label.split()
    if len(p_tokens) < n or len(l_tokens) < n:
        return 0.0
    p_counts = _ngram_counts(p_tokens, n)
    l_counts = _ngram_counts(l_tokens, n)
    overlap = sum(min(p_counts[k], l_counts[k]) for k in p_counts)
    if overlap == 0:
        return 0.0
    prec = overlap / sum(p_counts.values())
    rec = overlap / sum(l_counts.values())
    return 2 * prec * rec / (prec + rec)


def compute_rouge_n(preds, labels, n: int) -> float:
    """Return mean ROUGE-*n* F1 over ``preds`` and ``labels``."""
    scores = [_rouge_n(str(p), str(l), n) for p, l in zip(preds, labels)]
    return sum(scores) / len(scores) if scores else 0.0
