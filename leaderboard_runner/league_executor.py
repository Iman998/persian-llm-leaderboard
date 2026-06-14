"""Persistent named leagues with sampled battles and Elo matchmaking."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import random
import re
import subprocess
import tempfile
from typing import Any

import pandas as pd
import yaml

from leaderboard_lib.league import (
    HISTORY_COLUMNS,
    choose_dataset,
    choose_matchup,
    initial_standings,
    pair_key,
    update_standings,
)

from . import paths
from .battle_executor import (
    _battle_config,
    _result_path,
    build_battle_meta,
    load_battle_inputs,
)
from .cmd_utils import build_run_eval_cmd


logger = logging.getLogger(__name__)
GENERATION_TASKS = {"text_generation", "summarization", "translation"}


def league_slug(name: str) -> str:
    """Return a stable filesystem-safe identifier for a league name."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-._").lower()
    if slug:
        return slug
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]
    return f"league-{digest}"


def _league_dir(name: str) -> Path:
    return paths.RESULTS_DIR / "league" / league_slug(name)


def _read_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.read_csv(path).reindex(columns=HISTORY_COLUMNS)


def _write_state(
    league_dir: Path,
    standings: pd.DataFrame,
    history: pd.DataFrame,
) -> None:
    standings.to_csv(league_dir / "standings.csv", index=False)
    history.reindex(columns=HISTORY_COLUMNS).to_csv(
        league_dir / "history.csv", index=False
    )


def _load_or_create_config(
    *,
    league_dir: Path,
    name: str,
    models: list[str],
    datasets: list[str],
    judge_model: str | None,
    rows_per_match: int,
    k_factor: float,
    initial_elo: float,
    calibration_games: int,
    repeat_penalty: float,
    seed: int,
    dry_run: bool,
) -> dict[str, Any]:
    config_path = league_dir / "league.yaml"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if config.get("name") != name:
            raise ValueError(
                f"League slug collision: '{name}' conflicts with "
                f"'{config.get('name')}'"
            )
        if list(config.get("models", [])) != models:
            raise ValueError("Existing league uses a different model list")
        if list(config.get("datasets", [])) != datasets:
            raise ValueError("Existing league uses a different dataset list")
        if judge_model and config.get("judge_model") != judge_model:
            raise ValueError("Existing league uses a different judge model")
        return config

    if not judge_model:
        raise ValueError("A new league requires --league-judge-model")
    config = {
        "name": name,
        "slug": league_slug(name),
        "models": models,
        "datasets": datasets,
        "judge_model": judge_model,
        "rows_per_match": rows_per_match,
        "k_factor": float(k_factor),
        "initial_elo": float(initial_elo),
        "calibration_games": calibration_games,
        "repeat_penalty": float(repeat_penalty),
        "seed": seed,
        "strategy": (
            "Calibrate least-played models first; then prefer nearby Elo "
            "ratings with a repeat-pair penalty and balanced dataset usage."
        ),
    }
    if not dry_run:
        league_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return config


def _validate_pool(
    models: list[str],
    datasets: list[str],
    *,
    judge_model: str,
) -> dict[tuple[str, str], list[str]]:
    if len(models) < 2:
        raise ValueError("A league requires at least two models")
    if len(set(models)) != len(models):
        raise ValueError("League model names must be unique")
    if not datasets:
        raise ValueError("A league requires at least one dataset")

    for model in models:
        model_path = paths.MODELS_DIR / f"{model}.yaml"
        if not model_path.exists():
            raise FileNotFoundError(f"League model config missing: {model_path}")
    judge_path = paths.MODELS_DIR / f"{judge_model}.yaml"
    if not judge_path.exists():
        raise FileNotFoundError(f"League judge model missing: {judge_path}")

    valid_datasets = []
    for dataset in datasets:
        meta_path = paths.DATASETS_DIR / dataset / "meta.yaml"
        if not meta_path.exists():
            raise FileNotFoundError(f"League dataset metadata missing: {meta_path}")
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        if meta.get("task") not in GENERATION_TASKS:
            raise ValueError(
                f"League dataset '{dataset}' is not a generation dataset"
            )
        if _battle_config(meta, judge_model=judge_model) is None:
            raise ValueError(
                f"Battles are disabled for league dataset '{dataset}'"
            )
        valid_datasets.append(dataset)

    common: dict[tuple[str, str], list[str]] = {}
    for index, model_1 in enumerate(models):
        for model_2 in models[index + 1 :]:
            available = [
                dataset
                for dataset in valid_datasets
                if _result_path(dataset, model_1, None).exists()
                and _result_path(dataset, model_2, None).exists()
            ]
            common[pair_key(model_1, model_2)] = available

    unused_datasets = [
        dataset
        for dataset in valid_datasets
        if not any(dataset in available for available in common.values())
    ]
    if unused_datasets:
        raise FileNotFoundError(
            "No league model pair has existing results for dataset(s): "
            + ", ".join(unused_datasets)
        )

    adjacency = {model: set() for model in models}
    for (model_1, model_2), available in common.items():
        if available:
            adjacency[model_1].add(model_2)
            adjacency[model_2].add(model_1)
    connected = set()
    frontier = [models[0]]
    while frontier:
        model = frontier.pop()
        if model in connected:
            continue
        connected.add(model)
        frontier.extend(adjacency[model] - connected)
    if connected != set(models):
        disconnected = sorted(set(models) - connected)
        raise ValueError(
            "League model pool is disconnected; no comparable matchup path "
            f"exists for: {', '.join(disconnected)}"
        )
    return common


def _sample_match_rows(
    battle_df: pd.DataFrame,
    *,
    league_seed: int,
    dataset: str,
    model_1: str,
    model_2: str,
    rows_per_match: int,
    prior_pair_dataset_matches: int,
) -> tuple[pd.DataFrame, int]:
    """Rotate through a deterministic shuffled row order before repeating."""
    if battle_df.empty:
        raise ValueError("Cannot sample a league match from an empty result")
    count = min(rows_per_match, len(battle_df))
    material = (
        f"{league_seed}:{dataset}:{pair_key(model_1, model_2)}"
    ).encode("utf-8")
    pair_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    order = list(range(len(battle_df)))
    random.Random(pair_seed).shuffle(order)
    offset = prior_pair_dataset_matches * count
    selected = [order[(offset + index) % len(order)] for index in range(count)]
    cycle = offset // len(order)
    sampled = battle_df.iloc[selected].copy().reset_index(drop=False)
    sampled.rename(columns={"index": "league_source_row"}, inplace=True)
    return sampled, cycle


def _pair_dataset_match_count(
    history: pd.DataFrame,
    *,
    dataset: str,
    model_1: str,
    model_2: str,
) -> int:
    if history.empty:
        return 0
    pair = pair_key(model_1, model_2)
    matches = history[history["Dataset"].astype(str) == dataset]
    return sum(
        pair_key(str(row["Model 1"]), str(row["Model 2"])) == pair
        for _, row in matches.iterrows()
    )


def _run_match(
    *,
    league_name: str,
    league_dir: Path,
    match_number: int,
    dataset: str,
    model_1: str,
    model_2: str,
    judge_model: str,
    rows_per_match: int,
    league_seed: int,
    prior_pair_dataset_matches: int,
    workers: int,
    dry_run: bool,
) -> tuple[Path, int]:
    battle_df, meta_cfg = load_battle_inputs(
        dataset=dataset,
        model_1=model_1,
        model_2=model_2,
    )
    config = _battle_config(meta_cfg, judge_model=judge_model)
    if config is None:
        raise ValueError(f"Battles are disabled for league dataset '{dataset}'")
    sampled, sample_cycle = _sample_match_rows(
        battle_df,
        league_seed=league_seed,
        dataset=dataset,
        model_1=model_1,
        model_2=model_2,
        rows_per_match=rows_per_match,
        prior_pair_dataset_matches=prior_pair_dataset_matches,
    )
    sampled["judge_model"] = config["model"]
    sampled["league_name"] = league_name
    sampled["league_match"] = match_number
    battle_meta = build_battle_meta(
        meta_cfg,
        model_1=model_1,
        model_2=model_2,
        use_reference=config["use_reference"],
    )

    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset)
    safe_pair = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{model_1}__vs__{model_2}")
    output_path = (
        league_dir
        / "matches"
        / f"{match_number:06d}_{safe_dataset}_{safe_pair}.csv"
    )
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="llm-league-") as tmp_dir:
        input_path = Path(tmp_dir) / "match.csv"
        meta_path = Path(tmp_dir) / "meta.yaml"
        sampled.to_csv(input_path, index=False)
        meta_path.write_text(
            yaml.safe_dump(battle_meta, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        cmd = build_run_eval_cmd(
            model_stub=config["model"],
            dataset_path=input_path,
            meta_path=meta_path,
            prompt_template=config["prompt_template"],
            evaluator=config["evaluator"],
            n_rows=None,
            shots=0,
            workers=workers,
            out_csv=output_path,
        )
        if dry_run:
            print(" ".join(map(str, cmd)))
        else:
            subprocess.run([str(part) for part in cmd], check=True)
    return output_path, sample_cycle


def run_league(
    *,
    name: str,
    models: list[str],
    datasets: list[str],
    judge_model: str | None,
    matches: int,
    rows_per_match: int,
    workers: int,
    k_factor: float = 32.0,
    initial_elo: float = 1000.0,
    calibration_games: int = 2,
    repeat_penalty: float = 64.0,
    seed: int = 42,
    dry_run: bool = False,
) -> Path:
    """Run or continue a named league and return its directory."""
    if not name.strip():
        raise ValueError("League name cannot be empty")
    if matches <= 0 or rows_per_match <= 0:
        raise ValueError("League matches and rows per match must be positive")
    if calibration_games < 0 or k_factor <= 0 or repeat_penalty < 0:
        raise ValueError("Invalid league rating or scheduling configuration")

    league_dir = _league_dir(name)
    config = _load_or_create_config(
        league_dir=league_dir,
        name=name,
        models=models,
        datasets=datasets,
        judge_model=judge_model,
        rows_per_match=rows_per_match,
        k_factor=k_factor,
        initial_elo=initial_elo,
        calibration_games=calibration_games,
        repeat_penalty=repeat_penalty,
        seed=seed,
        dry_run=dry_run,
    )
    common_datasets = _validate_pool(
        models,
        datasets,
        judge_model=str(config["judge_model"]),
    )
    standings_path = league_dir / "standings.csv"
    history_path = league_dir / "history.csv"
    standings = (
        pd.read_csv(standings_path)
        if standings_path.exists()
        else initial_standings(models, initial_elo=float(config["initial_elo"]))
    )
    history = _read_history(history_path)

    for _ in range(matches):
        match_number = len(history) + 1
        model_1, model_2 = choose_matchup(
            standings,
            history,
            common_datasets,
            calibration_games=int(config["calibration_games"]),
            repeat_penalty=float(config["repeat_penalty"]),
        )
        datasets_for_pair = common_datasets[pair_key(model_1, model_2)]
        dataset = choose_dataset(
            model_1,
            model_2,
            datasets_for_pair,
            history,
        )
        prior_count = _pair_dataset_match_count(
            history,
            dataset=dataset,
            model_1=model_1,
            model_2=model_2,
        )
        logger.info(
            "LEAGUE %s match %d: %s vs %s on %s",
            name,
            match_number,
            model_1,
            model_2,
            dataset,
        )
        match_path, sample_cycle = _run_match(
            league_name=name,
            league_dir=league_dir,
            match_number=match_number,
            dataset=dataset,
            model_1=model_1,
            model_2=model_2,
            judge_model=str(config["judge_model"]),
            rows_per_match=int(config["rows_per_match"]),
            league_seed=int(config["seed"]),
            prior_pair_dataset_matches=prior_count,
            workers=workers,
            dry_run=dry_run,
        )

        if dry_run:
            row_wins_1 = row_wins_2 = 0
            equal_rows = int(config["rows_per_match"])
        else:
            match_df = pd.read_csv(match_path)
            outcomes = match_df["pred"].astype(str).str.strip().str.lower()
            row_wins_1 = int((outcomes == "model_1").sum())
            row_wins_2 = int((outcomes == "model_2").sum())
            equal_rows = int((outcomes == "equal").sum())
            if row_wins_1 + row_wins_2 + equal_rows == 0:
                raise RuntimeError(f"League match has no valid outcomes: {match_path}")

        standings, rating = update_standings(
            standings,
            model_1=model_1,
            model_2=model_2,
            model_1_row_wins=row_wins_1,
            model_2_row_wins=row_wins_2,
            equal_rows=equal_rows,
            k_factor=float(config["k_factor"]),
        )
        history_row = {
            "Match": match_number,
            "Dataset": dataset,
            "Model 1": model_1,
            "Model 2": model_2,
            "Judge": config["judge_model"],
            "Rows": row_wins_1 + row_wins_2 + equal_rows,
            "Model 1 Row Wins": row_wins_1,
            "Model 2 Row Wins": row_wins_2,
            "Equal Rows": equal_rows,
            "Model 1 Win Rate": row_wins_1
            / (row_wins_1 + row_wins_2 + equal_rows),
            "Model 2 Win Rate": row_wins_2
            / (row_wins_1 + row_wins_2 + equal_rows),
            "Equal Rate": equal_rows
            / (row_wins_1 + row_wins_2 + equal_rows),
            "Model 1 Score": rating["model_1_score"],
            "Model 1 Elo Before": rating["model_1_rating_before"],
            "Model 1 Elo After": rating["model_1_rating_after"],
            "Model 1 Elo Change": rating["model_1_delta"],
            "Model 2 Elo Before": rating["model_2_rating_before"],
            "Model 2 Elo After": rating["model_2_rating_after"],
            "Model 2 Elo Change": rating["model_2_delta"],
            "Result": rating["result"],
            "Sample Cycle": sample_cycle,
        }
        history = pd.concat(
            [history, pd.DataFrame([history_row])],
            ignore_index=True,
        )
        if not dry_run:
            _write_state(league_dir, standings, history)

    return league_dir
