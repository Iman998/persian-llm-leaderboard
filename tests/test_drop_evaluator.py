from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.drop_evaluator import DROPEvaluator


def _make_evaluator(tmp_path, shots=0):
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "drop.jinja2"
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: question\n"
        "passage_col: passage\n"
        "answer_col: accepted_answers\n"
        "canonical_answer_col: canonical_answer\n"
    )
    return DROPEvaluator(
        model_cfg={
            "api_key": "test",
            "base_url": "http://localhost",
            "model": "dummy",
        },
        prompt_path=prompt_path,
        meta_path=meta_path,
        shots=shots,
    )


def test_build_prompt_contains_passage_question_and_canonical_answer(tmp_path):
    evaluator = _make_evaluator(tmp_path, shots=1)
    df = pd.DataFrame(
        [
            {
                "passage": "Alice scored 7 points and Bob scored 5.",
                "question": "How many points did they score altogether?",
                "canonical_answer": "12",
                "accepted_answers": '[[\"12\"]]',
            },
            {
                "passage": "The match was played in June.",
                "question": "When was the match played?",
                "canonical_answer": "June",
                "accepted_answers": '[[\"June\"]]',
            },
        ]
    )

    prompt = evaluator._build_prompt(df, df.iloc[0])

    assert "Passage: The match was played in June." in prompt
    assert "Question: When was the match played?" in prompt
    assert "Answer: June" in prompt
    assert "Passage: Alice scored 7 points" in prompt


def test_canonical_multi_span_answer_uses_semicolons(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    row = pd.Series({"accepted_answers": '[["Alice", "Bob"]]'})
    assert evaluator._canonical_answer(row) == "Alice; Bob"
