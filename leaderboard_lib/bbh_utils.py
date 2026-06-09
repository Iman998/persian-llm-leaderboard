"""Scoring and answer-extraction helpers for BIG-Bench Hard."""

from __future__ import annotations

from collections import defaultdict
import json
import re
from typing import Any


_ANSWER_LINE_RE = re.compile(
    r"^\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ANSWER_IS_RE = re.compile(
    r"(?:so\s+)?the answer is\s*:?\s*(.+)",
    re.IGNORECASE,
)


def normalize_bbh_answer(value: Any) -> str:
    """Return a stable representation for BBH exact-match scoring."""
    text = " ".join(str(value or "").strip().split())
    if text.startswith(r"\boxed{") and text.endswith("}"):
        text = text[7:-1].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    return text.casefold()


def extract_bbh_answer(text: str | None) -> str | None:
    """Extract the final short answer from a BBH chain-of-thought response."""
    if not text:
        return None

    answer_lines = _ANSWER_LINE_RE.findall(text)
    if answer_lines:
        answer = answer_lines[-1]
    else:
        answer_mentions = _ANSWER_IS_RE.findall(text)
        if answer_mentions:
            answer = answer_mentions[-1]
        else:
            nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
            answer = nonempty_lines[-1] if nonempty_lines else ""

    normalized = normalize_bbh_answer(answer)
    return normalized or None


def make_bbh_scoring_key(task: str, family: str, target: Any) -> str:
    """Encode task identity and its normalized target into a stable JSON key."""
    return json.dumps(
        {
            "task": str(task),
            "family": str(family),
            "target": normalize_bbh_answer(target),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def parse_bbh_scoring_key(value: Any) -> dict[str, str]:
    """Decode a BBH scoring key, accepting plain answers as a fallback."""
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        decoded = None
    if isinstance(decoded, dict):
        return {
            "task": str(decoded.get("task", "")),
            "family": str(decoded.get("family", "")),
            "target": normalize_bbh_answer(decoded.get("target", "")),
        }
    return {"task": "", "family": "", "target": normalize_bbh_answer(value)}


def bbh_macro_accuracy(predictions: Any, labels: Any) -> float:
    """Return the macro-average exact-match accuracy over BBH task families."""
    family_scores: dict[str, list[float]] = defaultdict(list)
    for prediction, label in zip(predictions, labels):
        predicted = parse_bbh_scoring_key(prediction)
        expected = parse_bbh_scoring_key(label)
        family = expected["family"] or expected["task"]
        if not family:
            family = "unknown"
        same_task = not predicted["task"] or predicted["task"] == expected["task"]
        family_scores[family].append(
            float(same_task and predicted["target"] == expected["target"])
        )

    if not family_scores:
        return 0.0
    per_family = [sum(scores) / len(scores) for scores in family_scores.values()]
    return sum(per_family) / len(per_family)
