"""Normalization and scoring helpers for TriviaQA."""

from __future__ import annotations

from collections import Counter
import json
import re
import string
from typing import Any


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_triviaqa_answer(value: Any) -> str:
    """Apply the standard TriviaQA/SQuAD answer normalization."""
    text = str(value).lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def parse_answer_aliases(value: Any) -> list[str]:
    """Decode a JSON alias list, accepting plain strings as a fallback."""
    if isinstance(value, (list, tuple, set)):
        aliases = [str(alias) for alias in value]
    else:
        text = str(value)
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            decoded = text
        aliases = (
            [str(alias) for alias in decoded]
            if isinstance(decoded, list)
            else [str(decoded)]
        )
    return [alias for alias in aliases if alias.strip()]


def triviaqa_exact_match(prediction: Any, aliases: Any) -> float:
    """Return 1 when a prediction matches any accepted alias."""
    normalized_prediction = normalize_triviaqa_answer(prediction)
    return float(
        any(
            normalized_prediction == normalize_triviaqa_answer(alias)
            for alias in parse_answer_aliases(aliases)
        )
    )


def _token_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_triviaqa_answer(prediction).split()
    ground_truth_tokens = normalize_triviaqa_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    shared = sum(common.values())
    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)
    if shared == 0:
        return 0.0
    precision = shared / len(prediction_tokens)
    recall = shared / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def triviaqa_f1(prediction: Any, aliases: Any) -> float:
    """Return the maximum token F1 against all accepted aliases."""
    return max(
        (_token_f1(str(prediction), alias) for alias in parse_answer_aliases(aliases)),
        default=0.0,
    )
