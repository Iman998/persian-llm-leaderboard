#!/usr/bin/env bash
#─────────────────────────────────────────────────────────────────────────────
# run_all.sh
#
# Evaluate every (MODEL × DATASET) combination.
# Optional flag:
#   -n|--n_rows N   → evaluate on a random sample of N rows (header kept)
#─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

##############################################################################
# Edit these lists as needed                                                 #
##############################################################################
MODEL_LIST=( "gemma-3-27b-it" )     # ← models/<name>.yaml
DATASET_LIST=( "mmlu" )             # ← data/<dataset>/test.csv
WORKERS=100                         # Python worker threads
##############################################################################

##############################################################################
# Parse CLI arguments (only -n / --n_rows is supported)                      #
##############################################################################
N_ROWS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--n_rows) N_ROWS="$2"; shift 2 ;;
    *) echo "Usage: $0 [-n N_ROWS]" >&2; exit 1 ;;
  esac
done

##############################################################################
# Helper: colored logger                                                     #
##############################################################################
if [[ -z "${NO_COLOR:-}" ]]; then
  BOLD=$'\e[1m'; GRN=$'\e[32m'; YEL=$'\e[33m'; NC=$'\e[0m'
else
  BOLD=''; GRN=''; YEL=''; NC=''
fi
log(){ printf "%s[%s]%s %s\n" "$BOLD" "$1" "$NC" "$2"; }

##############################################################################
# Helper: create a temporary CSV with N random rows (header preserved)       #
##############################################################################
make_sample () {
  local src_csv="$1" n="$2"
  local tmp
  tmp="$(mktemp --suffix=.csv)"
  { head -n 1 "$src_csv"; tail -n +2 "$src_csv" | shuf -n "$n"; } > "$tmp"
  echo "$tmp"
}

log "INFO" "Worker threads per run: $WORKERS"
[[ -n "$N_ROWS" ]] && log "INFO" "Sampling $N_ROWS random rows"

##############################################################################
# Main loop: MODEL × DATASET                                                 #
##############################################################################
for model in "${MODEL_LIST[@]}"; do
  for ds in "${DATASET_LIST[@]}"; do
    csv="data/${ds}/test.csv"
    meta="data/${ds}/meta.yaml"
    [[ ! -f "$csv" ]] && { log "${YEL}SKIP${NC}" "$csv not found"; continue; }

    # Sample if requested
    if [[ -n "$N_ROWS" ]]; then
      csv_sample="$(make_sample "$csv" "$N_ROWS")"
      dataset_file="$csv_sample"
      suffix="_${N_ROWS}"
    else
      dataset_file="$csv"
      suffix=""
    fi

    out="results/${ds}_${model}${suffix}.csv"
    log "${GRN}RUN${NC}" "$model × $ds → $out"

    python "$ROOT/scripts/run_eval.py" \
      --dataset "$dataset_file" \
      --meta    "$meta" \
      --model   "models/${model}.yaml" \
      --prompt  "prompts/mcq_fewshot.jinja2" \
      --shots   3 \
      --workers "$WORKERS" \
      --out     "$out"

    # Remove temporary sample
    [[ -n "${csv_sample:-}" && -f "$csv_sample" ]] && rm -f "$csv_sample"
    unset csv_sample
  done
done

##############################################################################
# Re-build the leaderboard                                                   #
##############################################################################
python "$ROOT/scripts/build_leaderboard.py" \
  --results_dir results \
  --out dashboard/leaderboard.csv

log "${GRN}DONE${NC}" "Dashboard updated → dashboard/leaderboard.csv"
