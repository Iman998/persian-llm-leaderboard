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
MODEL_LIST=( "gpt-4.1-nano-2025-04-14" )     # ← models/<name>.yaml
DATASET_LIST=(
  "mmlu-pro" \
  "sentiment" \
  "nli" \
  "paraphrase" \
  "reading_comprehension" \
)             # ← data/<dataset>/test.csv
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
# Helper: create a clean CSV sample with N random rows (header preserved)    #
# Uses Python/Pandas so quoting & embedded newlines are handled correctly.   #
##############################################################################
make_sample () {
  local src_csv="$1" n="$2"
  local tmp
  tmp="$(mktemp --suffix=.csv)"
  python - "$src_csv" "$n" "$tmp" <<'PY'
import sys, pandas as pd, csv
src, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]

# Read with the tolerant python engine
df = pd.read_csv(
    src,
    engine="python",
    quoting=csv.QUOTE_MINIMAL,
    quotechar='"',
    escapechar="\\",
    on_bad_lines="skip",
)

n = min(n, len(df))
sample = df.sample(n, random_state=42)          # deterministic shuffle
sample.to_csv(out, index=False)
PY
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

    out_dir="results/${ds}/${model}"
    mkdir -p "$out_dir"
    out="${out_dir}/${model}${suffix}.csv"
    log "${GRN}RUN${NC}" "$model × $ds → $out"

    prompt="$(python - <<'PY' "$meta"
import sys, yaml; print(yaml.safe_load(open(sys.argv[1])).get('prompt_template', 'prompts/mcq_fewshot.jinja2'))
PY
)"
    evaluator="$(python - <<'PY' "$meta"
import sys, yaml; print(yaml.safe_load(open(sys.argv[1])).get('evaluator', 'evaluators/mcq_evaluator.py'))
PY
)"

    python "$ROOT/scripts/run_eval.py" \
      --dataset "$dataset_file" \
      --meta    "$meta" \
      --model   "models/${model}.yaml" \
      --prompt  "$prompt" \
      --evaluator "$evaluator" \
      --shots   3 \
      --workers "$WORKERS" \
      --out     "$out"

    if [[ -n "$N_ROWS" ]]; then
      cp "$out" "results/${ds}/${model}/${model}.csv"
      for f in "${out%.csv}"_*.csv; do
        [[ -f "$f" ]] && cp "$f" "results/${ds}/${model}${f#${out%.csv}}"
      done
    fi

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
  --datasets_dir data \
  --out dashboard/leaderboard.csv

log "${GRN}DONE${NC}" "Dashboard updated → dashboard/leaderboard.csv"
