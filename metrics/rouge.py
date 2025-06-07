"""ROUGE-L metric implementation."""


def _lcs(x: list[str], y: list[str]) -> int:
    table = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i, cx in enumerate(x, 1):
        for j, cy in enumerate(y, 1):
            if cx == cy:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def _rouge_l(pred: str, label: str) -> float:
    p = pred.split()
    l = label.split()
    lcs = _lcs(p, l)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p)
    rec = lcs / len(l)
    return 2 * prec * rec / (prec + rec)


def compute(preds, labels):
    """Return the mean ROUGE-L score for ``preds`` and ``labels``."""
    scores = [_rouge_l(str(p), str(l)) for p, l in zip(preds, labels)]
    return sum(scores) / len(scores) if scores else 0.0
