#!/usr/bin/env bash
# Overnight chain for E015-E019. Usage (from the repository root, inside tmux):
#   bash scripts/overnight_e015_e018.sh part1    # E015 -> E016 -> E017
#   bash scripts/overnight_e015_e018.sh part2    # E018 -> E019 (confirm) -> final summary
#   bash scripts/overnight_e015_e018.sh all      # part1 then part2
# Env: JOBS, THREADS, DEVICE (default auto from resources.json), SEEDS, POLITE_MAX_WAIT_S.
# Every experiment runs in its own subshell: control first; a failed control or stage
# prints "ОШИБКА: <стадия> прерван" and the chain continues with the next experiment.
set -uo pipefail
set +H
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
part="${1:-}"
if [[ "$part" != "part1" && "$part" != "part2" && "$part" != "all" ]]; then
  sed -n '2,8p' "$0"
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
  experiment E015_physics_diagnostics diagnostics
  experiment E016_phys_kerr_only control main summarize
  experiment E017_phys_plus_mlp control main summarize
}

run_part2() {
  experiment E018_dbp_depth control main summarize
  experiment E019_confirm_long_runs control main summarize
  echo "=== $(date -Is) итоговая сводка ==="
  PYTHONPATH=src "$PYTHON" scripts/recompute_q2.py || echo "ОШИБКА: recompute_q2 прерван"
  PYTHONPATH=src "$PYTHON" scripts/summarize_E015_E018.py || echo "ОШИБКА: summarize_E015_E018 прерван"
}

{
  echo "=== $(date -Is) overnight $part: JOBS=$JOBS THREADS=$THREADS DEVICE=$DEVICE host=$(hostname) commit=$(git rev-parse --short HEAD) ==="
  if [[ ! -f resources.json ]]; then
    echo "resources.json нет: запускаю probe"
    bash experiments/E016_phys_kerr_only/run.sh probe || echo "ОШИБКА: probe прерван"
  fi
  case "$part" in
    part1) run_part1 ;;
    part2) run_part2 ;;
    all) run_part1; run_part2 ;;
  esac
  echo "=== $(date -Is) overnight $part завершён; строки с ОШИБКА: $(grep -c '^ОШИБКА' "$LOG" 2>/dev/null || echo 0) ==="
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
