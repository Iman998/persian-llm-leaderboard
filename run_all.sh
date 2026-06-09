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
# Optional environment variables:
#   RUN_JUDGE=true   → pass the --judge flag to scripts/main.py
#------------------------------------------------------------------------------

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JUDGE_FLAG=""
if [[ "${RUN_JUDGE:-}" == "true" ]]; then
  JUDGE_FLAG="--judge"
fi

# ────────────────────────── CONFIGURATION ──────────────────────────── #
# Space-separated list of model stubs located in models/<name>.yaml
MODELS=( "gemma-3-12b-president-pt-sft-v2.1" )

# Space-separated list of dataset folder names inside data/
DATASETS=( "MATH" "HellaSwag" "BoolQ" "PIQA" "SocialIQA" "TriviaQA" "NaturalQuestions" "WinoGrande" "DROP" "BIG-Bench-Hard" )

# Sample size per dataset:
#   * Empty string  → evaluate full CSV (default)
#   * Positive int  → sample exactly N rows (header preserved)
N_ROWS=""

# Number of few-shot examples to include in each prompt
SHOTS=3

# Maximum number of Python worker threads (affects evaluator concurrency)
WORKERS=200
# ───────────────────────────────────────────────────────────────────── #

#
# Construct comma-separated strings required by the Python CLI
#
MODELS_CSV="$(IFS=,; echo "${MODELS[*]}")"
DATASETS_CSV="$(IFS=,; echo "${DATASETS[*]}")"

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
  ${JUDGE_FLAG:+$JUDGE_FLAG} "$@"
