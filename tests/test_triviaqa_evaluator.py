from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.triviaqa_evaluator import TriviaQAEvaluator


def _make_evaluator(tmp_path, shots=0):
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "triviaqa.jinja2"
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: question\n"
        "answer_col: accepted_answers\n"
        "canonical_answer_col: canonical_answer\n"
    )
    return TriviaQAEvaluator(
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
    response = "I recall the stage name.\nAnswer: David Seville"
    assert evaluator._extract(response) == "David Seville"


def test_extract_preserves_none_answer_through_csv_round_trip(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    assert evaluator._extract("Answer: None") == "None."


def test_build_prompt_uses_canonical_answer_for_few_shot(tmp_path):
    evaluator = _make_evaluator(tmp_path, shots=1)
    df = pd.DataFrame(
        [
            {
                "question": "Who was the man behind The Chipmunks?",
                "canonical_answer": "David Seville",
                "accepted_answers": '["David Seville", "Ross Bagdasarian"]',
            },
            {
                "question": "What is the capital of France?",
                "canonical_answer": "Paris",
                "accepted_answers": '["Paris"]',
            },
        ]
    )

    prompt = evaluator._build_prompt(df, df.iloc[0])

    assert "Answer: Paris" in prompt
    assert '["Paris"]' not in prompt
    assert "Question: Who was the man behind The Chipmunks?" in prompt


def test_build_prompt_recovers_none_answer_from_aliases(tmp_path):
    evaluator = _make_evaluator(tmp_path, shots=1)
    df = pd.DataFrame(
        [
            {
                "question": "Which answer represents no selection?",
                "canonical_answer": pd.NA,
                "accepted_answers": '["None", "None (disambiguation)"]',
            },
            {
                "question": "What is the capital of France?",
                "canonical_answer": "Paris",
                "accepted_answers": '["Paris"]',
            },
        ]
    )

    prompt = evaluator._build_prompt(df, df.iloc[1])
    assert "Answer: None" in prompt
