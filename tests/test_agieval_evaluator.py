from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.agieval_evaluator import AgievalEvaluator
from leaderboard_lib.agieval_utils import (
    agieval_macro_accuracy,
    make_agieval_scoring_key,
    normalize_agieval_choice,
)


def _make_evaluator(tmp_path, shots=1):
    root = Path(__file__).resolve().parents[1]
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: question\n"
        "answer_col: scoring_key\n"
        "target_col: target\n"
        "passage_col: passage\n"
        "task_col: task\n"
        "question_type_col: question_type\n"
        "language_col: language\n"
        "choice_cols: [choice1, choice2, choice3, choice4, choice5]\n"
    )
    return AgievalEvaluator(
        model_cfg={
            "api_key": "test",
            "base_url": "http://localhost",
            "model": "dummy",
        },
        prompt_path=root / "prompts" / "agieval.jinja2",
        meta_path=meta_path,
        shots=shots,
    )


def test_prompt_uses_same_task_few_shot_and_row_language(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    df = pd.DataFrame(
        [
            {
                "task": "sat-math",
                "language": "en",
                "question_type": "mcq",
                "passage": "",
                "question": "Current question?",
                "choice1": "(A) One",
                "choice2": "(B) Two",
                "target": "B",
            },
            {
                "task": "sat-math",
                "language": "en",
                "question_type": "mcq",
                "passage": "",
                "question": "Same-task example?",
                "choice1": "(A) Yes",
                "choice2": "(B) No",
                "target": "A",
            },
            {
                "task": "gaokao-mathqa",
                "language": "zh",
                "question_type": "mcq",
                "passage": "",
                "question": "Different task?",
                "choice1": "(A) Yes",
                "choice2": "(B) No",
                "target": "A",
            },
        ]
    )

    prompt = evaluator._build_prompt(df, df.iloc[0])

    assert "Same-task example?" in prompt
    assert "Different task?" not in prompt
    assert prompt.endswith("Final answer:")


def test_extracts_mcq_and_cloze_answers(tmp_path):
    evaluator = _make_evaluator(tmp_path, shots=0)
    assert evaluator._extract("Reasoning\nFinal answer: (C)") == "C"
    assert evaluator._extract("解析\n答案：A, D") == "AD"
    assert evaluator._extract(r"Therefore \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert normalize_agieval_choice("The correct option is C.") == "C"
    assert normalize_agieval_choice("A B D") == "ABD"


def test_macro_accuracy_weights_tasks_equally():
    labels = [
        make_agieval_scoring_key("large-task", "mcq", "A"),
        make_agieval_scoring_key("large-task", "mcq", "A"),
        make_agieval_scoring_key("large-task", "mcq", "A"),
        make_agieval_scoring_key("small-task", "cloze", "1/2"),
    ]
    predictions = [
        make_agieval_scoring_key("large-task", "mcq", "A"),
        make_agieval_scoring_key("large-task", "mcq", "A"),
        make_agieval_scoring_key("large-task", "mcq", "B"),
        make_agieval_scoring_key("small-task", "cloze", "0.5"),
    ]

    assert agieval_macro_accuracy(predictions, labels) == (2 / 3 + 1) / 2
