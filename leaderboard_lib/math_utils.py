"""Answer extraction and normalization helpers for the MATH benchmark."""

from __future__ import annotations

import re
from typing import Any


_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$"
)
_STAR_ANSWER_RE = re.compile(r"\*{4}\s*(.+?)\s*\*{4}", re.DOTALL)


def last_boxed_only_string(text: str | None) -> str | None:
    r"""Return the final balanced ``\boxed{...}`` or ``\fbox{...}`` expression."""
    if not text:
        return None

    candidates: list[tuple[int, str]] = []
    for command in (r"\boxed", r"\fbox"):
        start = 0
        while True:
            command_pos = text.find(command, start)
            if command_pos < 0:
                break

            brace_pos = command_pos + len(command)
            while brace_pos < len(text) and text[brace_pos].isspace():
                brace_pos += 1
            if brace_pos >= len(text) or text[brace_pos] != "{":
                start = command_pos + len(command)
                continue

            depth = 0
            for end_pos in range(brace_pos, len(text)):
                char = text[end_pos]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append((command_pos, text[command_pos : end_pos + 1]))
                        start = end_pos + 1
                        break
            else:
                start = brace_pos + 1

    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def remove_boxed(text: str | None) -> str | None:
    r"""Remove one outer ``\boxed`` or ``\fbox`` wrapper."""
    if text is None:
        return None

    for command in (r"\boxed", r"\fbox"):
        prefix = f"{command}{{"
        if text.startswith(prefix) and text.endswith("}"):
            return text[len(prefix) : -1]
    return text


def extract_math_answer(text: Any) -> str | None:
    """Extract a final answer from common MATH response formats."""
    if text is None:
        return None

    response = str(text).strip()
    if not response:
        return None

    boxed = last_boxed_only_string(response)
    if boxed:
        return remove_boxed(boxed)

    starred = _STAR_ANSWER_RE.findall(response)
    if starred:
        return starred[-1].strip()

    answer_lines = _ANSWER_LINE_RE.findall(response)
    if answer_lines:
        return answer_lines[-1].strip()

    nonempty_lines = [line.strip() for line in response.splitlines() if line.strip()]
    return nonempty_lines[-1] if nonempty_lines else None


def _fix_fracs(value: str) -> str:
    parts = value.split(r"\frac")
    if len(parts) == 1:
        return value

    fixed = parts[0]
    for part in parts[1:]:
        fixed += r"\frac"
        if not part:
            return value
        if part[0] == "{":
            fixed += part
            continue
        if len(part) < 2:
            return value

        numerator, denominator = part[0], part[1]
        remainder = part[2:]
        if denominator == "{":
            fixed += "{" + numerator + "}" + part[1:]
        else:
            fixed += "{" + numerator + "}{" + denominator + "}" + remainder
    return fixed


def _fix_a_slash_b(value: str) -> str:
    parts = value.split("/")
    if len(parts) != 2:
        return value
    try:
        numerator, denominator = (int(part) for part in parts)
    except ValueError:
        return value
    if value != f"{numerator}/{denominator}":
        return value
    return rf"\frac{{{numerator}}}{{{denominator}}}"


def _fix_sqrt(value: str) -> str:
    parts = value.split(r"\sqrt")
    if len(parts) == 1:
        return value

    fixed = parts[0]
    for part in parts[1:]:
        if not part:
            return value
        fixed += r"\sqrt" + (
            part if part[0] == "{" else "{" + part[0] + "}" + part[1:]
        )
    return fixed


def _remove_right_units(value: str) -> str:
    if r"\text{ " not in value:
        return value
    return value.split(r"\text{ ", 1)[0]


def normalize_math_answer(value: Any) -> str:
    """Normalize a MATH answer using the benchmark's reference conventions."""
    if value is None:
        return ""

    answer = extract_math_answer(value) or ""
    answer = answer.replace("\n", "")
    answer = answer.replace(r"\!", "")
    answer = answer.replace(r"\\", "\\")
    answer = answer.replace("tfrac", "frac").replace("dfrac", "frac")
    answer = answer.replace(r"\left", "").replace(r"\right", "")
    answer = answer.replace(r"^{\circ}", "").replace(r"^\circ", "")
    answer = answer.replace(r"\$", "").replace("$", "")
    answer = _remove_right_units(answer)
    answer = answer.replace(r"\%", "").replace(r"\%", "")
    answer = answer.replace(" .", " 0.").replace("{.", "{0.")
    answer = answer.strip()

    while answer and answer[-1] in ".,;":
        answer = answer[:-1].rstrip()
    if not answer:
        return ""
    if answer.startswith("."):
        answer = "0" + answer

    equals_parts = answer.split("=")
    if len(equals_parts) == 2 and len(equals_parts[0].strip()) <= 2:
        answer = equals_parts[1]

    answer = _fix_sqrt(answer)
    answer = answer.replace(" ", "")
    answer = _fix_fracs(answer)
    if answer == "0.5":
        answer = r"\frac{1}{2}"
    return _fix_a_slash_b(answer)


def math_answers_equivalent(first: Any, second: Any) -> bool:
    """Return whether two answers match after MATH normalization."""
    return normalize_math_answer(first) == normalize_math_answer(second)
