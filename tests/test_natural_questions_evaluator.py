from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.natural_questions_evaluator import NaturalQuestionsEvaluator


def _make_evaluator(tmp_path, shots=0):
    prompt_path = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "natural_questions.jinja2"
    )
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: question\n"
        "answer_col: accepted_answers\n"
        "canonical_answer_col: canonical_answer\n"
    )
    return NaturalQuestionsEvaluator(
        model_cfg={
            "api_key": "test",
            "base_url": "http://localhost",
            "model": "dummy",
        },
        prompt_path=prompt_path,
        meta_path=meta_path,
        shots=shots,
    )


def test_extract_uses_final_answer_line(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    assert evaluator._extract("Answer: December 1972") == "December 1972"


def test_build_prompt_uses_first_answer_for_few_shot(tmp_path):
    evaluator = _make_evaluator(tmp_path, shots=1)
    df = pd.DataFrame(
        [
            {
                "question": "When was the last time anyone was on the moon?",
                "canonical_answer": "14 December 1972 UTC",
                "accepted_answers": '["14 December 1972 UTC", "December 1972"]',
            },
            {
                "question": "Who wrote the lyrics?",
                "canonical_answer": "Bobby Scott",
                "accepted_answers": '["Bobby Scott", "Bob Russell"]',
            },
        ]
    )

    prompt = evaluator._build_prompt(df, df.iloc[0])

    assert "Answer: Bobby Scott" in prompt
    assert '["Bobby Scott"' not in prompt
    assert "Question: When was the last time anyone was on the moon?" in prompt
