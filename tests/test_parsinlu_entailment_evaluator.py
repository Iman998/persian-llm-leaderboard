from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.parsinlu_entailment_evaluator import ParsinLUEntailmentEvaluator


def _make_evaluator(tmp_path, shots=0):
    root = Path(__file__).resolve().parents[1]
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: sentence_1\n"
        "hypothesis_col: sentence_2\n"
        "answer_col: relation_label\n"
    )
    return ParsinLUEntailmentEvaluator(
        model_cfg={
            "api_key": "test",
            "base_url": "http://localhost",
            "model": "dummy",
        },
        prompt_path=root / "prompts" / "parsinlu_entailment.jinja2",
        meta_path=meta_path,
        shots=shots,
    )


def test_build_prompt_contains_premise_hypothesis_and_few_shot_label(tmp_path):
    evaluator = _make_evaluator(tmp_path, shots=1)
    df = pd.DataFrame(
        [
            {
                "sentence_1": "تهران پایتخت ایران است.",
                "sentence_2": "تهران در ایران قرار دارد.",
                "relation_label": "entailment",
            },
            {
                "sentence_1": "علی در خانه است.",
                "sentence_2": "علی بیرون از خانه است.",
                "relation_label": "contradiction",
            },
        ]
    )

    prompt = evaluator._build_prompt(df, df.iloc[0])

    assert "مقدمه: تهران پایتخت ایران است." in prompt
    assert "فرضیه: تهران در ایران قرار دارد." in prompt
    assert "برچسب: ****contradiction****" in prompt


def test_extracts_three_canonical_labels(tmp_path):
    evaluator = _make_evaluator(tmp_path)

    assert evaluator._extract("برچسب: ****entailment****") == "entailment"
    assert evaluator._extract("Reasoning\nLabel: contradiction") == "contradiction"
    assert evaluator._extract("neutral") == "neutral"
    assert evaluator._extract("equivalent") is None
