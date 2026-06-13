"""Answer normalization and task-macro scoring for AGIEval."""

from __future__ import annotations

from collections import defaultdict
import json
import re
from typing import Any

from .math_utils import normalize_math_answer


_LETTERS_RE = re.compile(r"(?<![A-Za-z])([A-G])(?![A-Za-z])")
_COMPACT_CHOICES_RE = re.compile(r"^[\(\[\{\s]*([A-G][A-G\s,;/、]*)[\)\]\}\s.]*$")


def normalize_agieval_choice(value: Any) -> str:
    """Return sorted unique option letters from an AGIEval answer."""
    if isinstance(value, (list, tuple, set)):
        value = " ".join(str(item) for item in value)
    text = str(value or "")
    compact = _COMPACT_CHOICES_RE.fullmatch(text)
    letters = (
        re.findall(r"[A-G]", compact.group(1))
        if compact
        else _LETTERS_RE.findall(text)
    )
    return "".join(sorted(set(letters)))


def normalize_agieval_target(question_type: str, value: Any) -> str:
    """Normalize an MCQ or mathematical cloze answer."""
    if question_type == "cloze":
        return normalize_math_answer(value)
    return normalize_agieval_choice(value)


def make_agieval_scoring_key(
    task: str,
    question_type: str,
    target: Any,
) -> str:
    """Encode task identity and normalized target for macro scoring."""
    return json.dumps(
        {
            "task": str(task),
            "question_type": str(question_type),
            "target": normalize_agieval_target(question_type, target),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def parse_agieval_scoring_key(value: Any) -> dict[str, str]:
    """Decode a scoring key, accepting plain answers as a fallback."""
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        decoded = None
    if isinstance(decoded, dict):
        question_type = str(decoded.get("question_type", "mcq"))
        return {
            "task": str(decoded.get("task", "")),
            "question_type": question_type,
            "target": normalize_agieval_target(
                question_type,
                decoded.get("target", ""),
            ),
        }
    return {
        "task": "",
        "question_type": "mcq",
        "target": normalize_agieval_choice(value),
    }


def agieval_macro_accuracy(predictions: Any, labels: Any) -> float:
    """Return macro-average accuracy over AGIEval task variants."""
    task_scores: dict[str, list[float]] = defaultdict(list)
    for prediction, label in zip(predictions, labels):
        predicted = parse_agieval_scoring_key(prediction)
        expected = parse_agieval_scoring_key(label)
        task = expected["task"] or "unknown"
        same_task = not predicted["task"] or predicted["task"] == task
        task_scores[task].append(
            float(same_task and predicted["target"] == expected["target"])
        )

    if not task_scores:
        return 0.0
    per_task = [sum(scores) / len(scores) for scores in task_scores.values()]
    return sum(per_task) / len(per_task)
