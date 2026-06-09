"""Official NQ-Open answer normalization and exact-match scoring."""

from __future__ import annotations

import json
import re
import string
from typing import Any
import unicodedata


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def normalize_nq_answer(value: Any) -> str:
    """Normalize an answer using the official NQ-Open evaluation rules."""
    text = unicodedata.normalize("NFD", str(value)).lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def parse_nq_answers(value: Any) -> list[str]:
    """Decode a JSON list of accepted answers."""
    if isinstance(value, (list, tuple, set)):
        answers = [str(answer) for answer in value]
    else:
        text = str(value)
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            decoded = text
        answers = (
            [str(answer) for answer in decoded]
            if isinstance(decoded, list)
            else [str(decoded)]
        )
    return [answer for answer in answers if answer.strip()]


def nq_exact_match(prediction: Any, answers: Any) -> float:
    """Return 1 when the prediction matches any accepted NQ-Open answer."""
    normalized_prediction = normalize_nq_answer(prediction)
    return float(
        any(
            normalized_prediction == normalize_nq_answer(answer)
            for answer in parse_nq_answers(answers)
        )
    )
