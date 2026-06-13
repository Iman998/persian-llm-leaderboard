"""Canonical task-macro accuracy for AGIEval."""

from leaderboard_lib.agieval_utils import agieval_macro_accuracy


def compute(preds, labels):
    """Return macro-average accuracy across AGIEval task variants."""
    return agieval_macro_accuracy(preds, labels)
