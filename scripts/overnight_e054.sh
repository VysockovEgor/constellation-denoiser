#!/usr/bin/env bash
# E054 after E050 start: waits for E050 results/start_best.csv and the end of the E050 start session; then control, fit, summarize.
# Usage: bash scripts/remote.sh go e054 all. JOBS_ADAPTIVE=1 and POLITE_EXCLUDE_OWN=1 as in E048.
set -uo pipefail
set +H
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs
LOG="logs/queue_e054_$(date +%Y%m%d_%H%M%S).log"
export THREADS=1 DEVICE=cpu JOBS_ADAPTIVE=1 POLITE_EXCLUDE_OWN=1
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
export PYTHON PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
gate() { "$PYTHON" -m constellation_denoiser.gate "$@"; }
stage() {
  local name="$1" kind="$2" rc
  gate halted || return 1
  gate memory || return 1
  JOBS="$(gate jobs --total)"; export JOBS
  echo "=== $(date -Is) E054_dbpx_pruning: stage $name ($kind) JOBS=$JOBS ==="
  bash experiments/E054_dbpx_pruning/run.sh "$name"; rc=$?
  case "$kind" in
    control|critical) [[ "$rc" == 0 ]] || { gate set --reason "E054 $name: код $rc"; return 1; } ;;
    train) if [[ "$rc" != 0 && "$rc" != 1 ]]; then gate set --reason "E054 $name: код $rc"; return 1; fi
           gate stage --exp E054 --stage "$name" || return 1 ;;
    other) [[ "$rc" == 0 ]] || echo "ОШИБКА: E054 $name (код $rc), очередь продолжается" ;;
  esac
}
{
  echo "=== $(date -Is) queue e054: host=$(hostname) commit=$(git rev-parse --short HEAD) ==="
  while [[ ! -f experiments/E050_dbpx_physical/results/start_best.csv ]] || pgrep -u "$(id -u)" -f "dbpx start" > /dev/null; do
    echo "$(date -Is) жду E050 start (start_best.csv)"; sleep 300
  done
  if stage control control; then
    for s in fit:other summarize:other; do stage "${s%%:*}" "${s##*:}"; done
  else
    echo "ОШИБКА: E054 прерван на control"
  fi
  echo "=== $(date -Is) queue e054 завершена; строк ОШИБКА: $(grep -c '^ОШИБКА' "$LOG" 2>/dev/null || echo 0) ==="
} 2>&1 | tee -a "$LOG"
