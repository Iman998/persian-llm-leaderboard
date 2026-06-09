from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.socialiqa_evaluator import SocialIQAEvaluator


def _make_evaluator(tmp_path):
    prompt_path = (
        Path(__file__).resolve().parents[1] / "prompts" / "socialiqa.jinja2"
    )
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: input\n"
        "answer_col: answer\n"
        "choice_cols: [choice1, choice2, choice3]\n"
    )
    model_cfg = {
        "api_key": "test",
        "base_url": "http://localhost",
        "model": "dummy",
    }
    return SocialIQAEvaluator(
        model_cfg=model_cfg,
        prompt_path=prompt_path,
        meta_path=meta_path,
    )


def test_build_prompt_contains_context_question_and_three_answers(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    df = pd.DataFrame(
        [
            {
                "input": (
                    "Context: Alex apologized to Robin for arriving late.\n\n"
                    "Question: How would Robin likely feel?"
                ),
                "choice1": "appreciative",
                "choice2": "invisible",
                "choice3": "hungry",
                "answer": 1,
            }
        ]
    )

    prompt = evaluator._build_prompt(df, df.iloc[0])

    assert "Context: Alex apologized to Robin for arriving late." in prompt
    assert "Question: How would Robin likely feel?" in prompt
    assert "1) appreciative" in prompt
    assert "3) hungry" in prompt


def test_socialiqa_uses_social_reasoning_system_prompt(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    assert "social commonsense reasoning" in evaluator.SYSTEM_PROMPT


def test_extract_returns_numeric_option_for_accuracy_metric(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    prediction = evaluator._extract("Correct option: ****3****")
    assert prediction == 3
    assert isinstance(prediction, int)
