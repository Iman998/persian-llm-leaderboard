from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from evaluators.gpqa_evaluator import GpqaEvaluator


def _make_evaluator(tmp_path, shots=1):
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: question\n"
        "answer_col: answer\n"
        "explanation_col: explanation\n"
        "domain_col: domain\n"
        "choice_cols: [choice1, choice2, choice3, choice4]\n",
        encoding="utf-8",
    )
    return GpqaEvaluator(
        model_cfg={
            "api_key": "test",
            "base_url": "http://localhost",
            "model": "dummy",
        },
        prompt_path=ROOT / "prompts" / "gpqa.jinja2",
        meta_path=meta_path,
        shots=shots,
    )


def test_prompt_uses_same_domain_example_and_rationale(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    df = pd.DataFrame(
        [
            {
                "domain": "Physics",
                "question": "Current question?",
                "choice1": "One",
                "choice2": "Two",
                "choice3": "Three",
                "choice4": "Four",
                "answer": 2,
                "explanation": "Current rationale.",
            },
            {
                "domain": "Physics",
                "question": "Physics example?",
                "choice1": "A",
                "choice2": "B",
                "choice3": "C",
                "choice4": "D",
                "answer": 3,
                "explanation": "Official physics rationale.",
            },
            {
                "domain": "Biology",
                "question": "Biology example?",
                "choice1": "A",
                "choice2": "B",
                "choice3": "C",
                "choice4": "D",
                "answer": 1,
                "explanation": "Biology rationale.",
            },
        ]
    )

    prompt = evaluator._build_prompt(df, df.iloc[0])

    assert "Physics example?" in prompt
    assert "Official physics rationale." in prompt
    assert "Biology example?" not in prompt
    assert "Correct option: ****3****" in prompt


def test_extracts_numeric_option(tmp_path):
    evaluator = _make_evaluator(tmp_path, shots=0)

    assert evaluator._extract("Reasoning\nCorrect option: ****4****") == 4
    assert evaluator._extract("Correct option: ****۲****") == 2
    assert evaluator._extract("Answer: 3") is None
