"""Positive-class F1 for binary question-pair classification."""


def compute(preds, labels):
    """Return F1 for the duplicate/paraphrase class labeled ``1``."""
    labels = [str(label).strip() for label in labels]
    if not labels:
        return 0.0
    predictions = [str(prediction).strip() for prediction in preds]
    true_positives = sum(
        prediction == "1" and label == "1"
        for prediction, label in zip(predictions, labels)
    )
    false_positives = sum(
        prediction == "1" and label != "1"
        for prediction, label in zip(predictions, labels)
    )
    false_negatives = sum(
        prediction != "1" and label == "1"
        for prediction, label in zip(predictions, labels)
    )
    denominator = 2 * true_positives + false_positives + false_negatives
    return 2 * true_positives / denominator if denominator else 0.0
