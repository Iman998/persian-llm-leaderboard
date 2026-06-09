from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.parsinlu_qqp_evaluator import ParsinLUQQPEvaluator


def _make_evaluator(tmp_path, shots=0):
    root = Path(__file__).resolve().parents[1]
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: question_1\n"
        "question_2_col: question_2\n"
        "answer_col: is_duplicate\n"
    )
    return ParsinLUQQPEvaluator(
        model_cfg={
            "api_key": "test",
            "base_url": "http://localhost",
            "model": "dummy",
        },
        prompt_path=root / "prompts" / "parsinlu_qqp.jinja2",
        meta_path=meta_path,
        shots=shots,
    )


def test_build_prompt_contains_both_questions_and_few_shot_label(tmp_path):
    evaluator = _make_evaluator(tmp_path, shots=1)
    df = pd.DataFrame(
        [
            {
                "question_1": "چگونه زبان انگلیسی یاد بگیرم؟",
                "question_2": "بهترین روش یادگیری انگلیسی چیست؟",
                "is_duplicate": "1",
            },
            {
                "question_1": "پایتخت ایران کجاست؟",
                "question_2": "جمعیت ایران چقدر است؟",
                "is_duplicate": "0",
            },
        ]
    )

    prompt = evaluator._build_prompt(df, df.iloc[0])

    assert "پرسش اول: چگونه زبان انگلیسی یاد بگیرم؟" in prompt
    assert "پرسش دوم: بهترین روش یادگیری انگلیسی چیست؟" in prompt
    assert "برچسب: ****0****" in prompt


def test_extracts_ascii_and_persian_binary_labels(tmp_path):
    evaluator = _make_evaluator(tmp_path)

    assert evaluator._extract("برچسب: ****1****") == "1"
    assert evaluator._extract("توضیح کوتاه\nبرچسب: ****۰****") == "0"
    assert evaluator._extract("Label: 1") == "1"
    assert evaluator._extract("این دو پرسش متفاوت‌اند.") is None
