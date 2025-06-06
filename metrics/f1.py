from collections import Counter


def _f1(pred: str, label: str) -> float:
    pred_tokens = str(pred).split()
    label_tokens = str(label).split()
    common = Counter(pred_tokens) & Counter(label_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(label_tokens)
    return 2 * precision * recall / (precision + recall)


def compute(preds, labels):
    scores = [_f1(p, l) for p, l in zip(preds, labels)]
    return sum(scores) / len(scores) if scores else 0.0
