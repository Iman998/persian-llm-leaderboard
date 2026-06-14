"""Aggregate persistent league standings for the dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


BOARD_COLUMNS = [
    "League",
    "League Slug",
    "Rank",
    "Model",
    "Model Stub",
    "Elo",
    "Games",
    "Match Wins",
    "Match Losses",
    "Match Draws",
    "Row Battles",
    "Row Wins",
    "Row Losses",
    "Row Equals",
    "Win Rate",
    "Loss Rate",
    "Equal Rate",
    "Score Rate",
    "Last Elo Change",
    "Judge",
    "Datasets",
    "Rows Per Match",
    "K Factor",
]


def _display_name(models_dir: Path, model_stub: str) -> str:
    model_path = models_dir / f"{model_stub}.yaml"
    if not model_path.exists():
        return model_stub
    config = yaml.safe_load(model_path.read_text(encoding="utf-8")) or {}
    return str(config.get("display_name", config.get("name", model_stub)))


def summarize_league(
    *,
    league_dir: Path,
    models_dir: Path,
) -> list[dict[str, Any]]:
    """Return dashboard rows for one league directory."""
    config_path = league_dir / "league.yaml"
    standings_path = league_dir / "standings.csv"
    if not config_path.exists() or not standings_path.exists():
        return []

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    standings = pd.read_csv(standings_path)
    required = {"Model", "Elo"}
    if standings.empty or not required.issubset(standings.columns):
        return []

    sort_columns = ["Elo"]
    ascending = [False]
    if "Score Rate" in standings:
        sort_columns.append("Score Rate")
        ascending.append(False)
    sort_columns.append("Model")
    ascending.append(True)
    standings = standings.sort_values(
        sort_columns,
        ascending=ascending,
    ).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for rank, standing in standings.iterrows():
        model_stub = str(standing["Model"])
        row = {
            "League": str(config.get("name", league_dir.name)),
            "League Slug": str(config.get("slug", league_dir.name)),
            "Rank": rank + 1,
            "Model": _display_name(models_dir, model_stub),
            "Model Stub": model_stub,
            "Judge": _display_name(
                models_dir, str(config.get("judge_model", ""))
            ),
            "Datasets": ", ".join(map(str, config.get("datasets", []))),
            "Rows Per Match": int(config.get("rows_per_match", 0)),
            "K Factor": float(config.get("k_factor", 0)),
        }
        for column in BOARD_COLUMNS:
            if column not in row and column in standing:
                row[column] = standing[column]
        rows.append(row)
    return rows


def build_league_board(
    *,
    results_dir: Path,
    models_dir: Path,
) -> pd.DataFrame:
    """Build a dataframe with one ranked row per model and named league."""
    rows: list[dict[str, Any]] = []
    league_root = results_dir / "league"
    for league_dir in sorted(path for path in league_root.glob("*") if path.is_dir()):
        rows.extend(summarize_league(league_dir=league_dir, models_dir=models_dir))
    return pd.DataFrame(rows, columns=BOARD_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    board = build_league_board(
        results_dir=Path(args.results_dir),
        models_dir=Path(args.models_dir),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(out_path, index=False)
    print(f"League board written -> {out_path}")


if __name__ == "__main__":
    main()
