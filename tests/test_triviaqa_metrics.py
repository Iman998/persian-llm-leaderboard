import json

from leaderboard_lib.triviaqa_utils import (
    normalize_triviaqa_answer,
    triviaqa_exact_match,
    triviaqa_f1,
)
from metrics.triviaqa_exact_match import compute as compute_exact_match
from metrics.triviaqa_f1 import compute as compute_f1


def test_normalization_removes_articles_punctuation_and_case():
    assert normalize_triviaqa_answer("The, Beatles!") == "beatles"


def test_exact_match_accepts_any_alias():
    aliases = json.dumps(["David Seville", "Ross Bagdasarian Sr."])
    assert triviaqa_exact_match("ross bagdasarian sr", aliases) == 1.0


def test_f1_uses_best_alias():
    aliases = json.dumps(["Mount Everest", "Everest"])
    assert triviaqa_f1("the mount everest", aliases) == 1.0


def test_metric_averages():
    labels = [json.dumps(["Paris"]), json.dumps(["Mount Everest"])]
    preds = ["The Paris.", "Everest Mount"]
    assert compute_exact_match(preds, labels) == 0.5
    assert compute_f1(preds, labels) == 1.0
