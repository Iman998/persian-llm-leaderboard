from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.zharfa_translation_evaluator import ZharfaTranslationEvaluator


def _make_evaluator(tmp_path, shots=1):
    root = Path(__file__).resolve().parents[1]
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: text\n"
        "answer_col: gold_translation\n"
        "source_language_col: src_language\n"
        "target_language_col: tgt_lang\n"
    )
    return ZharfaTranslationEvaluator(
        model_cfg={
            "api_key": "test",
            "base_url": "http://localhost",
            "model": "dummy",
        },
        prompt_path=root / "prompts" / "zharfa_translate.jinja2",
        meta_path=meta_path,
        shots=shots,
    )


def test_prompt_uses_row_languages_and_same_pair_few_shot(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    df = pd.DataFrame(
        [
            {
                "text": "سلام",
                "gold_translation": "hello",
                "src_language": "Persian",
                "tgt_lang": "English",
            },
            {
                "text": "خداحافظ",
                "gold_translation": "goodbye",
                "src_language": "Persian",
                "tgt_lang": "English",
            },
            {
                "text": "hello",
                "gold_translation": "hola",
                "src_language": "English",
                "tgt_lang": "Spanish",
            },
        ]
    )

    prompt = evaluator._build_prompt(df, df.iloc[0])

    assert "Translate the provided text from Persian to English." in prompt
    assert "Source (Persian): خداحافظ" in prompt
    assert "Translation (English): goodbye" in prompt
    assert "Translation (Spanish): hola" not in prompt
