"""TriviaQA token-level F1 metric."""

from leaderboard_lib.triviaqa_utils import triviaqa_f1


def compute(preds, labels):
    """Return mean maximum token F1 against accepted aliases."""
    preds = list(preds)
    labels = list(labels)
    if not labels:
        return 0.0
    return sum(
        triviaqa_f1(prediction, aliases)
        for prediction, aliases in zip(preds, labels)
    ) / len(labels)
