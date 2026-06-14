#!/usr/bin/env bash
#------------------------------------------------------------------------------
# run_all.sh  —  Thin wrapper that delegates heavy lifting to scripts/main.py
#
# HOW TO USE
# ----------
# 1) Adjust the CONFIGURATION block below and save the file.
# 2) Run:
#       bash run_all.sh          # normal mode
#       bash run_all.sh --dry    # just print commands (no execution)
#
# Any extra CLI flags (e.g. --dry) are forwarded unchanged to the Python
# orchestrator.  All comments are in English for consistency.
#
# Judge behavior is configured in the CONFIGURATION block below.
#------------------------------------------------------------------------------

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ────────────────────────── CONFIGURATION ──────────────────────────── #
# Space-separated list of model stubs located in models/<name>.yaml
MODELS=( "gemma-3-12b-president-pt-sft-v2.1" )

# Space-separated list of dataset folder names inside data/
DATASETS=( "MATH" "GSM8K" "HellaSwag" "BoolQ" "PIQA" "SocialIQA" "TriviaQA" "NaturalQuestions" "WinoGrande" "DROP" "BIG-Bench-Hard" "AGIEval" "GPQA" "parsinlu-qqp" "parsinlu-entailment" "zharfa_translate" )

# Sample size per dataset:
#   * Empty string  → evaluate full CSV (default)
#   * Positive int  → sample exactly N rows (header preserved)
N_ROWS=""

# Number of few-shot examples to include in each prompt
SHOTS=3

# Maximum number of Python worker threads (affects evaluator concurrency)
WORKERS=200

# LLM-as-judge configuration
#   RUN_JUDGE=true       → run judge evaluation for tagged datasets
#   JUDGE_ONLY=true      → reuse existing results; never regenerate model output
#   JUDGE_MODE options   → reference | no-reference | both
#   JUDGE_MODEL          → model stub from models/<name>.yaml
RUN_JUDGE=true
JUDGE_ONLY=true
JUDGE_MODE="both"
JUDGE_MODEL="deepseek-chat-judge"

# Pairwise Battle configuration
# Battle always consumes existing result CSVs for the two configured models.
RUN_BATTLE=false
BATTLE_ONLY=true
BATTLE_MODEL_1="gemma-3-4b-it"
BATTLE_MODEL_2="zharfa-mini"
BATTLE_JUDGE_MODEL="deepseek-chat-judge"

# Named Elo League configuration
# League reuses existing generation result CSVs and samples only a small row
# set per match. Re-running the same name continues its saved standings.
RUN_LEAGUE=false
LEAGUE_ONLY=true
LEAGUE_NAME="zharfa-generation-league"
LEAGUE_MODELS=( "gemma-3-4b-it" "zharfa-mini" )
LEAGUE_DATASETS=( "zharfa_translate" )
LEAGUE_JUDGE_MODEL="deepseek-chat-judge"
LEAGUE_MATCHES=10
LEAGUE_ROWS_PER_MATCH=20
LEAGUE_K_FACTOR=32
LEAGUE_INITIAL_ELO=1000
LEAGUE_CALIBRATION_GAMES=2
LEAGUE_REPEAT_PENALTY=64
LEAGUE_SEED=42
# ───────────────────────────────────────────────────────────────────── #

#
# Construct comma-separated strings required by the Python CLI
#
MODELS_CSV="$(IFS=,; echo "${MODELS[*]}")"
DATASETS_CSV="$(IFS=,; echo "${DATASETS[*]}")"
LEAGUE_MODELS_CSV="$(IFS=,; echo "${LEAGUE_MODELS[*]}")"
LEAGUE_DATASETS_CSV="$(IFS=,; echo "${LEAGUE_DATASETS[*]}")"
JUDGE_ARGS=()
if [[ "${RUN_JUDGE}" == "true" ]]; then
  JUDGE_ARGS+=(--judge --judge-mode "${JUDGE_MODE}")
  if [[ -n "${JUDGE_MODEL}" ]]; then
    JUDGE_ARGS+=(--judge-model "${JUDGE_MODEL}")
  fi
  if [[ "${JUDGE_ONLY}" == "true" ]]; then
    JUDGE_ARGS+=(--judge-only)
  fi
fi
BATTLE_ARGS=()
if [[ "${RUN_BATTLE}" == "true" ]]; then
  BATTLE_ARGS+=(
    --battle
    --battle-model-1 "${BATTLE_MODEL_1}"
    --battle-model-2 "${BATTLE_MODEL_2}"
    --battle-judge-model "${BATTLE_JUDGE_MODEL}"
  )
  if [[ "${BATTLE_ONLY}" == "true" ]]; then
    BATTLE_ARGS+=(--battle-only)
  fi
fi
LEAGUE_ARGS=()
if [[ "${RUN_LEAGUE}" == "true" ]]; then
  LEAGUE_ARGS+=(
    --league
    --league-name "${LEAGUE_NAME}"
    --league-models "${LEAGUE_MODELS_CSV}"
    --league-datasets "${LEAGUE_DATASETS_CSV}"
    --league-judge-model "${LEAGUE_JUDGE_MODEL}"
    --league-matches "${LEAGUE_MATCHES}"
    --league-rows-per-match "${LEAGUE_ROWS_PER_MATCH}"
    --league-k-factor "${LEAGUE_K_FACTOR}"
    --league-initial-elo "${LEAGUE_INITIAL_ELO}"
    --league-calibration-games "${LEAGUE_CALIBRATION_GAMES}"
    --league-repeat-penalty "${LEAGUE_REPEAT_PENALTY}"
    --league-seed "${LEAGUE_SEED}"
  )
  if [[ "${LEAGUE_ONLY}" == "true" ]]; then
    LEAGUE_ARGS+=(--league-only)
  fi
fi

#
# Forward everything to the orchestrator.  If N_ROWS is set (non-empty),
# we add the -n flag; otherwise we omit it.
#
python3 "${SCRIPT_DIR}/scripts/main.py" \
  -m "${MODELS_CSV}" \
  -d "${DATASETS_CSV}" \
  ${N_ROWS:+-n "${N_ROWS}"} \
  -s "${SHOTS}" \
  -w "${WORKERS}" \
  "${JUDGE_ARGS[@]}" \
  "${BATTLE_ARGS[@]}" \
  "${LEAGUE_ARGS[@]}" \
  "$@"
