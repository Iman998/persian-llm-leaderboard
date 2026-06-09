from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.math_evaluator import MathEvaluator
from leaderboard_lib.math_utils import (
    extract_math_answer,
    math_answers_equivalent,
    normalize_math_answer,
)


@pytest.fixture
def evaluator(tmp_path):
    prompt_path = tmp_path / "template.jinja2"
    prompt_path.write_text("{{ question }}")
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text("question_col: problem\nanswer_col: answer\n")
    model_cfg = {
        "api_key": "test",
        "base_url": "http://localhost",
        "model": "dummy",
    }
    return MathEvaluator(
        model_cfg=model_cfg,
        prompt_path=prompt_path,
        meta_path=meta_path,
    )


def test_extracts_last_balanced_boxed_answer():
    text = (
        r"First try: \boxed{2}. "
        r"After simplifying, the final answer is \boxed{\frac{9}{7}}."
    )
    assert extract_math_answer(text) == r"\frac{9}{7}"


@pytest.mark.parametrize(
    "text, expected",
    [
        (r"Answer: \boxed{\dfrac{1}{2}}", r"\frac{1}{2}"),
        ("Answer: **** 0.5 ****", r"\frac{1}{2}"),
        (r"Answer: \boxed{x = \sqrt3}", r"\sqrt{3}"),
        (r"Answer: \boxed{(3, -1)}", "(3,-1)"),
    ],
)
def test_normalizes_math_answers(text, expected):
    assert normalize_math_answer(text) == expected


def test_math_equivalence_handles_common_latex_variants():
    assert math_answers_equivalent(r"\dfrac{1}{2}", "0.5")
    assert math_answers_equivalent(r"\sqrt3", r"\sqrt{3}")
    assert not math_answers_equivalent(r"\frac{1}{2}", r"\frac{2}{3}")


def test_evaluator_extracts_final_answer_not_first_number(evaluator):
    response = r"First calculate 12 - 3 = 9. Answer: \boxed{\frac{9}{7}}"
    assert evaluator._extract(response) == r"\frac{9}{7}"
