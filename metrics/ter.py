"""Translation Edit Rate (TER) implementation.

This module computes the TER score between predictions and references.
The TER score is defined as ``1 - edits / len(ref_tokens)`` where *edits*
represents the minimal number of insertions, deletions, substitutions, and
shifts required to transform the prediction tokens into the reference tokens.
"""

from functools import lru_cache


def _levenshtein(hyp: tuple[str, ...], ref: tuple[str, ...]) -> int:
    """Return standard Levenshtein edit distance."""
    n, m = len(hyp), len(ref)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if hyp[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],     # deletion
                    dp[i][j - 1],     # insertion
                    dp[i - 1][j - 1], # substitution
                )
    return dp[n][m]


@lru_cache(maxsize=None)
def _ter_distance(hyp: tuple[str, ...], ref: tuple[str, ...]) -> int:
    """Return minimal edit distance allowing shifts.

    A naive search over all possible single shifts is performed. The function
    recurses so multiple shifts can be applied if they reduce the distance.
    To avoid infinite recursion, we only consider shifts that strictly reduce
    the standard Levenshtein distance to the reference.
    """
    best = _levenshtein(hyp, ref)
    n = len(hyp)
    # Try shifting every possible span to every possible new position
    for start in range(n):
        for end in range(start + 1, n + 1):
            span = hyp[start:end]
            remaining = hyp[:start] + hyp[end:]
            for insert in range(len(remaining) + 1):
                candidate = remaining[:insert] + span + remaining[insert:]
                if candidate == hyp:
                    continue
                cand_base = _levenshtein(candidate, ref)
                if cand_base >= best:
                    continue
                dist = 1 + _ter_distance(candidate, ref)
                if dist < best:
                    best = dist
    return best


def _ter(pred: str, label: str) -> float:
    ref_tokens = tuple(str(label).split())
    if not ref_tokens:
        return 0.0
    hyp_tokens = tuple(str(pred).split())
    edits = _ter_distance(hyp_tokens, ref_tokens)
    return 1 - (edits / len(ref_tokens))


def compute(preds, labels):
    """Return the average TER score for *preds* and *labels*."""
    scores = [_ter(p, l) for p, l in zip(preds, labels)]
    return sum(scores) / len(scores) if scores else 0.0
