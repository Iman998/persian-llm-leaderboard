from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.winogrande_evaluator import WinoGrandeEvaluator


def _make_evaluator(tmp_path):
    prompt_path = (
        Path(__file__).resolve().parents[1] / "prompts" / "winogrande.jinja2"
    )
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: sentence\n"
        "answer_col: answer\n"
        "choice_cols: [choice1, choice2]\n"
    )
    return WinoGrandeEvaluator(
        model_cfg={
            "api_key": "test",
            "base_url": "http://localhost",
            "model": "dummy",
        },
        prompt_path=prompt_path,
        meta_path=meta_path,
    )


def test_build_prompt_contains_sentence_and_candidates(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    df = pd.DataFrame(
        [
            {
                "sentence": "The trophy did not fit in the suitcase because _ was too big.",
                "choice1": "the trophy",
                "choice2": "the suitcase",
                "answer": 1,
            }
        ]
    )

    prompt = evaluator._build_prompt(df, df.iloc[0])

    assert "Sentence: The trophy did not fit" in prompt
    assert "1) the trophy" in prompt
    assert "2) the suitcase" in prompt
    assert "Correct option: ****n****" in prompt


def test_winogrande_uses_coreference_system_prompt(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    assert "commonsense coreference resolution" in evaluator.SYSTEM_PROMPT


def test_extract_returns_numeric_option_for_accuracy_metric(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    prediction = evaluator._extract("Correct option: ****2****")
    assert prediction == 2
    assert isinstance(prediction, int)
