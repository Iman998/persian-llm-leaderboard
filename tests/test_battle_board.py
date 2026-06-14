from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from leaderboard_lib.battle_board import build_battle_board, summarize_battle


def test_summarize_battle_computes_both_models_rates(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "m1.yaml").write_text("display_name: Model One\n")
    (models / "m2.yaml").write_text("display_name: Model Two\n")
    (models / "judge.yaml").write_text("display_name: Judge\n")
    df = pd.DataFrame(
        {
            "model_1": ["m1"] * 5,
            "model_2": ["m2"] * 5,
            "judge_model": ["judge"] * 5,
            "pred": ["model_1", "model_1", "model_2", "equal", "invalid"],
        }
    )

    result = summarize_battle(df, dataset="translation", models_dir=models)

    assert result["Evaluated Rows"] == 4
    assert result["Model 1 Win Rate"] == 0.5
    assert result["Model 1 Loss Rate"] == 0.25
    assert result["Model 2 Win Rate"] == 0.25
    assert result["Model 2 Loss Rate"] == 0.5
    assert result["Equal Rate"] == 0.25


def test_build_battle_board_discovers_dataset_matchups(tmp_path):
    results = tmp_path / "results"
    models = tmp_path / "models"
    battle_dir = results / "battle" / "ds" / "m1__vs__m2"
    battle_dir.mkdir(parents=True)
    models.mkdir()
    pd.DataFrame(
        {
            "model_1": ["m1", "m1"],
            "model_2": ["m2", "m2"],
            "judge_model": ["judge", "judge"],
            "pred": ["model_1", "equal"],
        }
    ).to_csv(battle_dir / "battle.csv", index=False)

    board = build_battle_board(results_dir=results, models_dir=models)

    assert board["Dataset"].tolist() == ["ds"]
    assert board.loc[0, "Model 1 Wins"] == 1
    assert board.loc[0, "Equal"] == 1
