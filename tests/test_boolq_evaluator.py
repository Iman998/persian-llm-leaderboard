from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.boolq_evaluator import BoolQEvaluator


def _make_evaluator(tmp_path):
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "boolq.jinja2"
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: input\n"
        "answer_col: answer\n"
        "choice_cols: [choice1, choice2]\n"
    )
    model_cfg = {
        "api_key": "test",
        "base_url": "http://localhost",
        "model": "dummy",
    }
    return BoolQEvaluator(
        model_cfg=model_cfg,
        prompt_path=prompt_path,
        meta_path=meta_path,
    )


def test_build_prompt_contains_passage_question_and_yes_no_options(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    df = pd.DataFrame(
        [
            {
                "input": (
                    "Passage: Water freezes at zero degrees Celsius.\n\n"
                    "Question: does water freeze at zero degrees celsius?"
                ),
                "choice1": "No",
                "choice2": "Yes",
                "answer": 2,
            }
        ]
    )

    prompt = evaluator._build_prompt(df, df.iloc[0])

    assert "Passage: Water freezes at zero degrees Celsius." in prompt
    assert "Question: does water freeze at zero degrees celsius?" in prompt
    assert "1) No" in prompt
    assert "2) Yes" in prompt


def test_boolq_uses_reading_comprehension_system_prompt(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    assert "yes/no reading comprehension" in evaluator.SYSTEM_PROMPT


def test_extract_returns_numeric_option_for_accuracy_metric(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    prediction = evaluator._extract("Correct option: ****2****")
    assert prediction == 2
    assert isinstance(prediction, int)
