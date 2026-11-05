#!/usr/bin/env bash
# E050 (DBP-X) after E048: waits until the E048 queue has finished, then unittest, start, control, train + ref, summarize.
# Usage: bash scripts/remote.sh go e050 all. JOBS_ADAPTIVE=1 and POLITE_EXCLUDE_OWN=1 as in E048.
set -uo pipefail
set +H
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs
LOG="logs/queue_e050_$(date +%Y%m%d_%H%M%S).log"
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
  echo "=== $(date -Is) E050_dbpx_physical: stage $name ($kind) JOBS=$JOBS ==="
  bash experiments/E050_dbpx_physical/run.sh "$name"; rc=$?
  case "$kind" in
    control|critical) [[ "$rc" == 0 ]] || { gate set --reason "E050 $name: код $rc"; return 1; } ;;
    train) if [[ "$rc" != 0 && "$rc" != 1 ]]; then gate set --reason "E050 $name: код $rc"; return 1; fi
           gate stage --exp E050 --stage "$name" || return 1 ;;
    other) [[ "$rc" == 0 ]] || echo "ОШИБКА: E050 $name (код $rc), очередь продолжается" ;;
  esac
}
{
  echo "=== $(date -Is) queue e050: host=$(hostname) commit=$(git rev-parse --short HEAD) ==="
  while pgrep -u "$(id -u)" -f "overnight_e048.sh" > /dev/null; do
    echo "$(date -Is) жду окончания очереди E048"; sleep 600
  done
  ok=1
  for s in unittest:critical start:critical control:control; do
    stage "${s%%:*}" "${s##*:}" || { echo "ОШИБКА: E050 прерван на ${s%%:*}"; ok=0; break; }
  done
  if [[ "$ok" == 1 ]]; then
    ( stage train other ) &
    trained=$!
    stage ref train || echo "ОШИБКА: E050 ref"
    wait "$trained"
    stage summarize other
  fi
  echo "=== $(date -Is) queue e050 завершена; строк ОШИБКА: $(grep -c '^ОШИБКА' "$LOG" 2>/dev/null || echo 0) ==="
} 2>&1 | tee -a "$LOG"
