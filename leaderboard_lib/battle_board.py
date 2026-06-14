"""Aggregate row-level pairwise battle results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


BOARD_COLUMNS = [
    "Dataset",
    "Model 1",
    "Model 2",
    "Judge",
    "Evaluated Rows",
    "Model 1 Wins",
    "Model 2 Wins",
    "Equal",
    "Model 1 Win Rate",
    "Model 1 Loss Rate",
    "Model 2 Win Rate",
    "Model 2 Loss Rate",
    "Equal Rate",
]


def _display_name(models_dir: Path, model_stub: str) -> str:
    model_path = models_dir / f"{model_stub}.yaml"
    if not model_path.exists():
        return model_stub
    config = yaml.safe_load(model_path.read_text(encoding="utf-8")) or {}
    return str(config.get("display_name", model_stub))


def summarize_battle(
    df: pd.DataFrame,
    *,
    dataset: str,
    models_dir: Path,
) -> dict[str, Any] | None:
    """Return one matchup summary from a row-level battle dataframe."""
    required = {"model_1", "model_2", "judge_model", "pred"}
    if not required.issubset(df.columns) or df.empty:
        return None
    model_1_stub = str(df["model_1"].iloc[0])
    model_2_stub = str(df["model_2"].iloc[0])
    judge_stub = str(df["judge_model"].iloc[0])
    outcomes = df["pred"].astype(str).str.strip().str.lower()
    valid = outcomes.isin({"model_1", "model_2", "equal"})
    outcomes = outcomes[valid]
    total = len(outcomes)
    if total == 0:
        return None

    model_1_wins = int((outcomes == "model_1").sum())
    model_2_wins = int((outcomes == "model_2").sum())
    equal = int((outcomes == "equal").sum())
    model_1_win_rate = model_1_wins / total
    model_2_win_rate = model_2_wins / total
    equal_rate = equal / total
    return {
        "Dataset": dataset,
        "Model 1": _display_name(models_dir, model_1_stub),
        "Model 2": _display_name(models_dir, model_2_stub),
        "Judge": _display_name(models_dir, judge_stub),
        "Evaluated Rows": total,
        "Model 1 Wins": model_1_wins,
        "Model 2 Wins": model_2_wins,
        "Equal": equal,
        "Model 1 Win Rate": model_1_win_rate,
        "Model 1 Loss Rate": model_2_win_rate,
        "Model 2 Win Rate": model_2_win_rate,
        "Model 2 Loss Rate": model_1_win_rate,
        "Equal Rate": equal_rate,
    }


def build_battle_board(
    *,
    results_dir: Path,
    models_dir: Path,
) -> pd.DataFrame:
    """Build a dataframe with one row per dataset matchup."""
    rows = []
    battle_root = results_dir / "battle"
    for battle_path in sorted(battle_root.glob("*/*/battle.csv")):
        dataset = battle_path.relative_to(battle_root).parts[0]
        summary = summarize_battle(
            pd.read_csv(battle_path),
            dataset=dataset,
            models_dir=models_dir,
        )
        if summary is not None:
            rows.append(summary)
    return pd.DataFrame(rows, columns=BOARD_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    board = build_battle_board(
        results_dir=Path(args.results_dir),
        models_dir=Path(args.models_dir),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(out_path, index=False)
    print(f"Battle board written -> {out_path}")


if __name__ == "__main__":
    main()
