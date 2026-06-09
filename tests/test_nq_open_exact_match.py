import json

from leaderboard_lib.natural_questions_utils import (
    normalize_nq_answer,
    nq_exact_match,
)
from metrics.nq_open_exact_match import compute


def test_normalization_matches_official_rules():
    assert normalize_nq_answer("The, Beatles!") == "beatles"


def test_normalization_applies_unicode_nfd():
    assert normalize_nq_answer("Café") == normalize_nq_answer("Cafe\u0301")


def test_exact_match_accepts_any_reference():
    answers = json.dumps(["14 December 1972 UTC", "December 1972"])
    assert nq_exact_match("the december 1972.", answers) == 1.0


def test_metric_average():
    labels = [json.dumps(["Paris"]), json.dumps(["Mount Everest"])]
    assert compute(["The Paris.", "Everest"], labels) == 0.5
