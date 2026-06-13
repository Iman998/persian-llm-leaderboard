"""Exact numeric accuracy for GSM8K."""

from leaderboard_lib.gsm8k_utils import normalize_gsm8k_answer


def compute(preds, labels):
    """Return exact-match accuracy after numeric normalization."""
    predictions = list(preds)
    references = list(labels)
    if not references:
        return 0.0
    correct = sum(
        normalize_gsm8k_answer(prediction) == normalize_gsm8k_answer(reference)
        for prediction, reference in zip(predictions, references)
    )
    return correct / len(references)
