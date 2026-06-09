from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.hellaswag_evaluator import HellaSwagEvaluator


def _make_evaluator(tmp_path):
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "hellaswag.jinja2"
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: context\n"
        "answer_col: answer\n"
        "choice_cols: [choice1, choice2, choice3, choice4]\n"
    )
    model_cfg = {
        "api_key": "test",
        "base_url": "http://localhost",
        "model": "dummy",
    }
    return HellaSwagEvaluator(
        model_cfg=model_cfg,
        prompt_path=prompt_path,
        meta_path=meta_path,
    )


def test_build_prompt_uses_context_and_four_endings(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    df = pd.DataFrame(
        [
            {
                "context": "A person opens a refrigerator.",
                "choice1": "They take out a carton of milk.",
                "choice2": "The mountain begins singing.",
                "choice3": "They drive the refrigerator.",
                "choice4": "The room turns into a boat.",
                "answer": 1,
            }
        ]
    )

    prompt = evaluator._build_prompt(df, df.iloc[0])

    assert "Context: A person opens a refrigerator." in prompt
    assert "1) They take out a carton of milk." in prompt
    assert "4) The room turns into a boat." in prompt
    assert "Correct option: ****n****" in prompt


def test_hellaswag_uses_continuation_specific_system_prompt(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    assert "commonsense sentence completion" in evaluator.SYSTEM_PROMPT


def test_extract_returns_numeric_option_for_accuracy_metric(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    prediction = evaluator._extract("Correct option: ****4****")
    assert prediction == 4
    assert isinstance(prediction, int)
