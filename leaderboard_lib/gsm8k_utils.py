"""Answer extraction and normalization helpers for GSM8K."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
from typing import Any


_HASH_ANSWER_RE = re.compile(r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?)")
_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*(?:final\s+)?answer\s*:\s*"
    r"(?:\\boxed\{)?\s*([-+]?\$?[\d,]+(?:\.\d+)?)"
)
_NUMBER_RE = re.compile(r"[-+]?\$?[\d,]+(?:\.\d+)?")


def normalize_gsm8k_answer(value: Any) -> str:
    """Return a canonical decimal representation for a GSM8K answer."""
    text = str(value or "").strip().replace("$", "").replace(",", "")
    if not text:
        return ""
    try:
        number = Decimal(text)
    except InvalidOperation:
        return ""
    if number == number.to_integral():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def extract_gsm8k_answer(text: Any) -> str | None:
    """Extract the final numeric answer from common GSM8K response formats."""
    response = str(text or "").strip()
    if not response:
        return None

    hash_answers = _HASH_ANSWER_RE.findall(response)
    if hash_answers:
        answer = hash_answers[-1]
    else:
        answer_lines = _ANSWER_LINE_RE.findall(response)
        if answer_lines:
            answer = answer_lines[-1]
        else:
            numbers = _NUMBER_RE.findall(response)
            answer = numbers[-1] if numbers else ""

    normalized = normalize_gsm8k_answer(answer)
    return normalized or None
