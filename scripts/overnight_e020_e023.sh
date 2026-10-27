#!/usr/bin/env bash
# Overnight chain for E020-E023. Usage (from the repository root, inside tmux):
#   bash scripts/overnight_e020_e023.sh part1    # E020 (anomaly reruns, short)
#   bash scripts/overnight_e020_e023.sh part2    # E021 (missing Pareto points) -> E022 (baselines) -> E021 summary again
#   bash scripts/overnight_e020_e023.sh part3    # E023 (split robustness; members from the E021 summary)
#   bash scripts/overnight_e020_e023.sh all      # part1, part2, part3
# Env: JOBS, THREADS, DEVICE (default auto from resources.json), SEEDS, POLITE_MAX_WAIT_S.
# Every experiment runs in its own subshell: control first; a failed control or stage
# prints "ОШИБКА: <стадия> прерван" and the chain continues with the next experiment.
set -uo pipefail
set +H
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
part="${1:-}"
if [[ "$part" != "part1" && "$part" != "part2" && "$part" != "part3" && "$part" != "all" ]]; then
  sed -n '2,9p' "$0"
  exit 2
fi
mkdir -p logs
LOG="logs/overnight_${part}_$(date +%Y%m%d_%H%M%S).log"
export JOBS="${JOBS:-auto}" THREADS="${THREADS:-auto}" DEVICE="${DEVICE:-auto}"
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
export PYTHON

experiment() {           # experiment <folder> <stage>...
  local folder="$1"
  shift
  local stage
  echo "=== $(date -Is) $folder: $* ==="
  # explicit exit: `set -e` is ignored inside a subshell that is the left side of `||`
  (
    for stage in "$@"; do
      bash "experiments/$folder/run.sh" "$stage" || { echo "стадия $stage: код $?"; exit 1; }
    done
  ) || echo "ОШИБКА: $folder прерван (см. experiments/$folder/logs/)"
}

run_part1() {
  experiment E020_e017_anomaly control main summarize
}

run_part2() {
  experiment E021_pareto control main summarize
  experiment E022_baselines control main summarize
  experiment E021_pareto summarize          # adds the E022 baselines to the Pareto plot
}

run_part3() {
  experiment E023_split_robustness control main summarize
}

{
  echo "=== $(date -Is) overnight $part: JOBS=$JOBS THREADS=$THREADS DEVICE=$DEVICE host=$(hostname) commit=$(git rev-parse --short HEAD) ==="
  if [[ ! -f resources.json ]]; then
    echo "resources.json нет: запускаю probe"
    bash experiments/E020_e017_anomaly/run.sh probe || echo "ОШИБКА: probe прерван"
  fi
  case "$part" in
    part1) run_part1 ;;
    part2) run_part2 ;;
    part3) run_part3 ;;
    all) run_part1; run_part2; run_part3 ;;
  esac
  echo "=== $(date -Is) overnight $part завершён; строки с ОШИБКА: $(grep -c '^ОШИБКА' "$LOG" 2>/dev/null || echo 0) ==="
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
