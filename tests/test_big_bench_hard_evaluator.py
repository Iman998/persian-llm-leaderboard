from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.big_bench_hard_evaluator import BigBenchHardEvaluator
from leaderboard_lib.bbh_utils import make_bbh_scoring_key


def _make_evaluator(tmp_path, shots=3):
    root = Path(__file__).resolve().parents[1]
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: input\n"
        "answer_col: scoring_key\n"
        "task_col: task\n"
        "task_family_col: task_family\n"
        f"cot_prompt_dir: {root / 'prompts' / 'big_bench_hard'}\n"
    )
    return BigBenchHardEvaluator(
        model_cfg={
            "api_key": "test",
            "base_url": "http://localhost",
            "model": "dummy",
        },
        prompt_path=root / "prompts" / "big_bench_hard.jinja2",
        meta_path=meta_path,
        shots=shots,
    )


def test_build_prompt_uses_official_task_demonstrations(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    df = pd.DataFrame(
        [
            {
                "task": "boolean_expressions",
                "task_family": "boolean_expressions",
                "input": "True and False is",
                "scoring_key": "",
            }
        ]
    )

    prompt = evaluator._build_prompt(df, df.iloc[0])

    assert prompt.count("Q:") == 4
    assert "Evaluate the result of a random Boolean expression." in prompt
    assert "Q: True and False is" in prompt
    assert prompt.endswith("A: Let's think step by step.")


def test_zero_shot_prompt_omits_demonstrations(tmp_path):
    evaluator = _make_evaluator(tmp_path, shots=0)
    row = pd.Series(
        {
            "task": "word_sorting",
            "task_family": "word_sorting",
            "input": "Sort the following words alphabetically: List: beta alpha",
        }
    )

    prompt = evaluator._build_prompt(pd.DataFrame([row]), row)

    assert prompt.count("Q:") == 1
    assert "oven costume counterpart" not in prompt


def test_extracts_official_and_explicit_answer_formats(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    assert evaluator._extract("Reasoning. So the answer is (B).") == "(b)"
    assert evaluator._extract("Final answer: True") == "true"


def test_evaluate_df_encodes_task_identity_for_macro_scoring(tmp_path):
    evaluator = _make_evaluator(tmp_path, shots=0)
    evaluator._query_model = lambda prompt: "So the answer is True."
    df = pd.DataFrame(
        [
            {
                "task": "boolean_expressions",
                "task_family": "boolean_expressions",
                "input": "True is",
                "scoring_key": make_bbh_scoring_key(
                    "boolean_expressions", "boolean_expressions", "True"
                ),
            }
        ]
    )

    result = evaluator.evaluate_df(df, max_workers=1)

    assert result.loc[0, "answer_prediction"] == "true"
    assert result.loc[0, "pred"] == result.loc[0, "scoring_key"]
