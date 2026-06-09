import json

from leaderboard_lib.drop_utils import (
    best_drop_metrics,
    drop_metrics,
    normalize_drop_answer,
)
from metrics.drop_exact_match import compute as compute_exact_match
from metrics.drop_f1 import compute as compute_f1


def test_normalization_preserves_and_normalizes_numbers():
    assert normalize_drop_answer("The 3.0-yard run") == "3.0 yard run"


def test_single_answer_metrics():
    assert drop_metrics("three", ["three"]) == (1.0, 1.0)
    assert drop_metrics("3", ["3.0"]) == (1.0, 1.0)


def test_multi_span_alignment_is_order_independent():
    assert drop_metrics("Bob; Alice", ["Alice", "Bob"]) == (1.0, 1.0)
    assert drop_metrics("Bob;Alice", ["Alice", "Bob"]) == (1.0, 1.0)


def test_best_validated_annotation_is_used():
    annotations = json.dumps([["Chaz Schilens"], ["JaMarcus Russell"]])
    assert best_drop_metrics("JaMarcus Russell", annotations) == (1.0, 1.0)


def test_html_entities_are_not_split_as_multi_span_answers():
    answer = "Mart&#237;n Gram&#225;tica"
    assert drop_metrics(answer, [answer]) == (1.0, 1.0)


def test_metric_averages():
    labels = [json.dumps([["12"]]), json.dumps([["Alice", "Bob"]])]
    preds = ["12", "Alice"]
    assert compute_exact_match(preds, labels) == 0.5
    assert compute_f1(preds, labels) == 0.75
