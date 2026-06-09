"""Official DROP exact-match and F1 scoring helpers."""

from __future__ import annotations

from collections.abc import Sequence
import json
import re
import string
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_HTML_ENTITY_RE = re.compile(r"&(?:#\d+|#x[0-9a-f]+|[a-z][a-z0-9]+);", re.IGNORECASE)
_PUNCTUATION = set(string.punctuation)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def normalize_drop_answer(text: str) -> str:
    """Normalize an answer using the official DROP evaluation rules."""
    parts: list[str] = []
    for token in re.split(" |-", text.lower()):
        if not _is_number(token):
            token = "".join(char for char in token if char not in _PUNCTUATION)
        if _is_number(token):
            token = str(float(token))
        token = " ".join(_ARTICLES_RE.sub(" ", token).split())
        if token:
            parts.append(token)
    return " ".join(parts).strip()


def parse_drop_prediction(value: Any) -> list[str]:
    """Parse a semicolon-delimited model answer into answer spans."""
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value)
    protected = _HTML_ENTITY_RE.sub(
        lambda match: f"{match.group(0)[:-1]}\0",
        text,
    )
    spans = [part.replace("\0", ";").strip() for part in protected.split(";")]
    return [span for span in spans if span] or [text.strip()]


def parse_drop_annotations(value: Any) -> list[list[str]]:
    """Decode candidate DROP annotations from JSON."""
    if isinstance(value, list):
        decoded = value
    else:
        decoded = json.loads(str(value))
    return [
        [str(span) for span in annotation]
        for annotation in decoded
        if isinstance(annotation, Sequence) and not isinstance(annotation, str)
    ]


def _answer_to_bags(answer: Sequence[str]) -> tuple[list[str], list[set[str]]]:
    normalized = [normalize_drop_answer(span) for span in answer]
    return normalized, [set(span.split()) for span in normalized]


def _numbers_match(gold: set[str], predicted: set[str]) -> bool:
    gold_numbers = {token for token in gold if _is_number(token)}
    predicted_numbers = {token for token in predicted if _is_number(token)}
    return not gold_numbers or bool(gold_numbers & predicted_numbers)


def _bag_f1(predicted: set[str], gold: set[str]) -> float:
    intersection = len(gold & predicted)
    precision = intersection / len(predicted) if predicted else 1.0
    recall = intersection / len(gold) if gold else 1.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _align_bags(predicted: list[set[str]], gold: list[set[str]]) -> list[float]:
    scores = np.zeros((len(gold), len(predicted)))
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _numbers_match(gold_item, pred_item):
                scores[gold_index, pred_index] = _bag_f1(pred_item, gold_item)

    row_indices, column_indices = linear_sum_assignment(-scores)
    max_scores = np.zeros(max(len(gold), len(predicted)))
    for row, column in zip(row_indices, column_indices):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores.tolist()


def drop_metrics(prediction: Any, annotation: Sequence[str]) -> tuple[float, float]:
    """Return official DROP exact match and F1 for one annotation."""
    predicted = parse_drop_prediction(prediction)
    predicted_normalized, predicted_bags = _answer_to_bags(predicted)
    gold_normalized, gold_bags = _answer_to_bags(list(annotation))

    exact_match = float(
        set(predicted_normalized) == set(gold_normalized)
        and len(predicted_normalized) == len(gold_normalized)
    )
    f1 = round(float(np.mean(_align_bags(predicted_bags, gold_bags))), 2)
    return exact_match, f1


def best_drop_metrics(prediction: Any, annotations: Any) -> tuple[float, float]:
    """Return the best EM and F1 over all validated annotations."""
    candidates = parse_drop_annotations(annotations)
    scores = [drop_metrics(prediction, candidate) for candidate in candidates]
    return (
        max((score[0] for score in scores), default=0.0),
        max((score[1] for score in scores), default=0.0),
    )
