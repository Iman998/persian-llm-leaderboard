from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from leaderboard_lib.league import initial_standings
from leaderboard_lib.league_board import build_league_board


def test_build_league_board_discovers_and_ranks_named_league(tmp_path):
    results = tmp_path / "results"
    models = tmp_path / "models"
    league_dir = results / "league" / "demo"
    league_dir.mkdir(parents=True)
    models.mkdir()
    (models / "m1.yaml").write_text("display_name: Model One\n")
    (models / "m2.yaml").write_text("display_name: Model Two\n")
    (models / "judge.yaml").write_text("display_name: Judge\n")
    (league_dir / "league.yaml").write_text(
        "name: Demo League\n"
        "slug: demo\n"
        "judge_model: judge\n"
        "datasets: [d1, d2]\n"
        "rows_per_match: 20\n"
        "k_factor: 32\n"
    )
    standings = initial_standings(["m1", "m2"])
    standings.loc[standings["Model"] == "m2", "Elo"] = 1040
    standings.to_csv(league_dir / "standings.csv", index=False)

    board = build_league_board(results_dir=results, models_dir=models)

    assert board["League"].tolist() == ["Demo League", "Demo League"]
    assert board["Rank"].tolist() == [1, 2]
    assert board["Model"].tolist() == ["Model Two", "Model One"]
    assert board["Judge"].tolist() == ["Judge", "Judge"]
    assert board.loc[0, "Datasets"] == "d1, d2"
