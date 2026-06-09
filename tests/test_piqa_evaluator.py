from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.piqa_evaluator import PIQAEvaluator


def _make_evaluator(tmp_path):
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "piqa.jinja2"
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: goal\n"
        "answer_col: answer\n"
        "choice_cols: [choice1, choice2]\n"
    )
    model_cfg = {
        "api_key": "test",
        "base_url": "http://localhost",
        "model": "dummy",
    }
    return PIQAEvaluator(
        model_cfg=model_cfg,
        prompt_path=prompt_path,
        meta_path=meta_path,
    )


def test_build_prompt_contains_goal_and_two_solutions(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    df = pd.DataFrame(
        [
            {
                "goal": "Keep a paper from blowing away.",
                "choice1": "Place a heavy book on top of it.",
                "choice2": "Put it in front of a running fan.",
                "answer": 1,
            }
        ]
    )

    prompt = evaluator._build_prompt(df, df.iloc[0])

    assert "Goal: Keep a paper from blowing away." in prompt
    assert "1) Place a heavy book on top of it." in prompt
    assert "2) Put it in front of a running fan." in prompt
    assert "Correct option: ****n****" in prompt


def test_piqa_uses_physical_reasoning_system_prompt(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    assert "physical commonsense reasoning" in evaluator.SYSTEM_PROMPT


def test_extract_returns_numeric_option_for_accuracy_metric(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    prediction = evaluator._extract("Correct option: ****1****")
    assert prediction == 1
    assert isinstance(prediction, int)
