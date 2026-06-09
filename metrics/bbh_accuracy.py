"""Canonical 23-task macro accuracy for BIG-Bench Hard."""

from leaderboard_lib.bbh_utils import bbh_macro_accuracy


def compute(preds, labels):
    """Return macro-average exact-match accuracy across BBH task families."""
    return bbh_macro_accuracy(preds, labels)
