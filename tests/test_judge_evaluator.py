from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.judge_evaluator import JudgeEvaluator


def _make_evaluator(tmp_path):
    root = Path(__file__).resolve().parents[1]
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: text\n"
        "answer_col: gold_translation\n"
        "candidate_col: candidate\n"
        "source_language_col: src_language\n"
        "target_language_col: tgt_lang\n"
        "judge_score_min: 0\n"
        "judge_score_max: 100\n"
    )
    return JudgeEvaluator(
        model_cfg={
            "api_key": "test",
            "base_url": "http://localhost",
            "model": "dummy",
        },
        prompt_path=root / "prompts" / "judge_zharfa_translation.jinja2",
        meta_path=meta_path,
        shots=0,
    )


def test_extracts_embedded_json_and_section_scores(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    details = evaluator._extract_details(
        '```json\n{"score": 87, "reason": "Faithful", '
        '"section_scores": {"Adequacy": 90, "Fluency": 84}}\n```'
    )

    assert details["score"] == "87"
    assert details["reason"] == "Faithful"
    assert details["section_scores"] == {"adequacy": "90", "fluency": "84"}


def test_rejects_score_outside_configured_range(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    assert evaluator._extract('{"score": 101, "reason": "invalid"}')[0] is None


def test_prompt_contains_language_pair_and_evaluation_keeps_sections(
    tmp_path, monkeypatch
):
    evaluator = _make_evaluator(tmp_path)
    df = pd.DataFrame(
        [
            {
                "text": "hello",
                "gold_translation": "سلام",
                "candidate": "درود",
                "src_language": "English",
                "tgt_lang": "Persian",
            }
        ]
    )
    prompt = evaluator._build_prompt(df, df.iloc[0])
    assert "Source language: English" in prompt
    assert "Target language: Persian" in prompt

    monkeypatch.setattr(
        evaluator,
        "_query_model",
        lambda _: (
            '{"score": 75, "reason": "Good", '
            '"section_scores": {"adequacy": 80, "fluency": 70}}'
        ),
    )
    result = evaluator.evaluate_df(df, max_workers=1)
    assert result.loc[0, "pred"] == "75"
    assert result.loc[0, "reason"] == "Good"
    assert result.loc[0, "judge_adequacy_score"] == "80"
    assert result.loc[0, "judge_fluency_score"] == "70"
