"""Macro F1 for three-way natural language inference."""


_LABELS = ("entailment", "contradiction", "neutral")


def _class_f1(predictions, labels, target):
    true_positives = sum(
        prediction == target and label == target
        for prediction, label in zip(predictions, labels)
    )
    false_positives = sum(
        prediction == target and label != target
        for prediction, label in zip(predictions, labels)
    )
    false_negatives = sum(
        prediction != target and label == target
        for prediction, label in zip(predictions, labels)
    )
    denominator = 2 * true_positives + false_positives + false_negatives
    return 2 * true_positives / denominator if denominator else 0.0


def compute(preds, labels):
    """Return unweighted mean F1 across the three ParsiNLU NLI labels."""
    labels = [str(label).strip().lower() for label in labels]
    if not labels:
        return 0.0
    predictions = [str(prediction).strip().lower() for prediction in preds]
    return sum(
        _class_f1(predictions, labels, target) for target in _LABELS
    ) / len(_LABELS)
