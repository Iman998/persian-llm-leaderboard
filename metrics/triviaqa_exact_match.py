"""TriviaQA normalized exact-match metric."""

from leaderboard_lib.triviaqa_utils import triviaqa_exact_match


def compute(preds, labels):
    """Return mean normalized exact match against accepted aliases."""
    preds = list(preds)
    labels = list(labels)
    if not labels:
        return 0.0
    return sum(
        triviaqa_exact_match(prediction, aliases)
        for prediction, aliases in zip(preds, labels)
    ) / len(labels)
