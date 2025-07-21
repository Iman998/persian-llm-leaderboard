"""Convenience wrapper for ROUGE-1."""
from .rouge import compute_rouge_n

def compute(preds, labels):
    return compute_rouge_n(preds, labels, n=1)

