from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluators.battle_evaluator import BattleEvaluator


def _make_evaluator(tmp_path):
    root = Path(__file__).resolve().parents[1]
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "question_col: text\n"
        "answer_col: gold\n"
        "model_1_col: model_1_output\n"
        "model_2_col: model_2_output\n"
        "model_1_name: first\n"
        "model_2_name: second\n"
    )
    return BattleEvaluator(
        model_cfg={
            "api_key": "test",
            "base_url": "http://localhost",
            "model": "judge",
        },
        prompt_path=root / "prompts" / "battle.jinja2",
        meta_path=meta_path,
    )


def test_extract_choice_supports_json_and_ties():
    assert BattleEvaluator._extract_choice(
        '{"winner": "candidate_b", "reason": "More accurate"}'
    ) == ("candidate_b", "More accurate")
    assert BattleEvaluator._extract_choice("tie: equivalent answers")[0] == "equal"


def test_swapped_candidate_winner_maps_back_to_real_model(tmp_path, monkeypatch):
    evaluator = _make_evaluator(tmp_path)
    monkeypatch.setattr(evaluator, "_swap_candidates", lambda row: True)
    monkeypatch.setattr(
        evaluator,
        "_query_model",
        lambda prompt: '{"winner": "candidate_a", "reason": "Better"}',
    )
    df = pd.DataFrame(
        [
            {
                "text": "Translate hello",
                "gold": "سلام",
                "model_1_output": "درود",
                "model_2_output": "سلام",
            }
        ]
    )

    result = evaluator.evaluate_df(df, max_workers=1)

    assert result.loc[0, "pred"] == "model_2"
    assert result.loc[0, "winner_model"] == "second"
    assert result.loc[0, "presentation_order"] == "model_2|model_1"


def test_equal_outcome_has_no_model_winner(tmp_path, monkeypatch):
    evaluator = _make_evaluator(tmp_path)
    monkeypatch.setattr(
        evaluator,
        "_query_model",
        lambda prompt: '{"winner": "equal", "reason": "Equivalent"}',
    )
    df = pd.DataFrame(
        [
            {
                "text": "q",
                "gold": "a",
                "model_1_output": "a",
                "model_2_output": "a",
            }
        ]
    )

    result = evaluator.evaluate_df(df, max_workers=1)

    assert result.loc[0, "pred"] == "equal"
    assert result.loc[0, "winner_model"] == "equal"
