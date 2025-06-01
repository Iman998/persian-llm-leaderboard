#!/usr/bin/env bash
#─────────────────────────────────────────────────────────────────────────────
# run_all.sh
#
# Evaluate each (MODEL × DATASET) combination.  Multi-threading happens
# inside Python; the number of worker threads is passed via --workers.
#─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#──────────────────────────── EDIT THESE LISTS ↓ ────────────────────────────#
MODEL_LIST=( "gemma-3-27b-it" )           # →  models/Qwen30.yaml
DATASET_LIST=( "khayyam_challenge" ) # →  data/khayyam_challenge/*.*
WORKERS=100                           # threads per Python evaluation
#────────────────────────────────────────────────────────────────────────────#

# simple color logger (disable with NO_COLOR env)
if [[ -z "${NO_COLOR:-}" ]]; then
  BOLD=$'\e[1m'; GRN=$'\e[32m'; YEL=$'\e[33m'; NC=$'\e[0m'
else
  BOLD=''; GRN=''; YEL=''; NC=''
fi
log(){ printf "%s[%s]%s %s\n" "$BOLD" "$1" "$NC" "$2"; }

log "INFO" "Worker threads per run: $WORKERS"

for model in "${MODEL_LIST[@]}"; do
  for ds in "${DATASET_LIST[@]}"; do
    csv="data/${ds}/${ds}.csv"
    meta="data/${ds}/meta.yaml"
    if [[ ! -f "$csv" ]]; then
      log "${YEL}SKIP${NC}" "$csv not found"
      continue
    fi

    out="results/${ds}_${model}.csv"
    log "${GRN}RUN${NC}" "$model × $ds → $out"

    python "$ROOT/scripts/run_eval.py" \
      --dataset "$csv" \
      --meta    "$meta" \
      --model   "models/${model}.yaml" \
      --prompt  "prompts/mcq_fewshot.jinja2" \
      --shots   3 \
      --workers "$WORKERS" \
      --out     "$out"
  done
done

python "$ROOT/scripts/build_leaderboard.py" \
  --results_dir results \
  --out dashboard/leaderboard.csv

log "${GRN}DONE${NC}" "Dashboard updated → dashboard/leaderboard.csv"
