from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.gsm8k_evaluator import Gsm8kEvaluator
from leaderboard_lib.gsm8k_utils import (
    extract_gsm8k_answer,
    normalize_gsm8k_answer,
)
from metrics.gsm8k_accuracy import compute


def _make_evaluator(tmp_path, shots=1):
    root = Path(__file__).resolve().parents[1]
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: question\n"
        "answer_col: answer\n"
        "solution_col: solution\n"
    )
    return Gsm8kEvaluator(
        model_cfg={
            "api_key": "test",
            "base_url": "http://localhost",
            "model": "dummy",
        },
        prompt_path=root / "prompts" / "gsm8k.jinja2",
        meta_path=meta_path,
        shots=shots,
    )


def test_prompt_includes_canonical_few_shot_solution(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    df = pd.DataFrame(
        [
            {
                "question": "Current problem?",
                "solution": "Current work.\n#### 10",
                "answer": "10",
            },
            {
                "question": "Example problem?",
                "solution": "Example work.\n#### 7",
                "answer": "7",
            },
        ]
    )

    prompt = evaluator._build_prompt(df, df.iloc[0])

    assert "Example problem?" in prompt
    assert "Example work." in prompt
    assert prompt.endswith("Question: Current problem?\nSolution:")


def test_extracts_final_numeric_answer():
    assert extract_gsm8k_answer("Work: 9 * 2 = 18\n#### 18") == "18"
    assert extract_gsm8k_answer("Reasoning with 3.\nFinal answer: $1,200") == "1200"
    assert extract_gsm8k_answer("After 5 steps, the result is -2.") == "-2"
    assert normalize_gsm8k_answer("18.0") == "18"


def test_evaluator_uses_last_final_answer(tmp_path):
    evaluator = _make_evaluator(tmp_path, shots=0)
    assert evaluator._extract("First answer: 4\nFinal answer: 11") == "11"


def test_metric_normalizes_string_and_numeric_answers():
    assert compute(["1,200", "-2", "18.0"], [1200, -2, 18]) == 1.0
