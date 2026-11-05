#!/usr/bin/env bash
# E048 next to the running queues: closed-form A (M = 11, 15) and DBP + off-axis triplets. Usage: bash scripts/remote.sh go e048 all.
# JOBS_ADAPTIVE=1 and POLITE_EXCLUDE_OWN=1: processes follow the free cores, our other queues are not foreign load.
set -uo pipefail
set +H
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs
LOG="logs/queue_e048_$(date +%Y%m%d_%H%M%S).log"
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
  echo "=== $(date -Is) E048_kerr_offaxis_dbp: stage $name ($kind) JOBS=$JOBS ==="
  bash experiments/E048_kerr_offaxis_dbp/run.sh "$name"; rc=$?
  case "$kind" in
    control|critical) [[ "$rc" == 0 ]] || { gate set --reason "E048 $name: код $rc"; return 1; } ;;
    train) if [[ "$rc" != 0 && "$rc" != 1 ]]; then gate set --reason "E048 $name: код $rc"; return 1; fi
           gate stage --exp E048 --stage "$name" || return 1 ;;
    other) [[ "$rc" == 0 ]] || echo "ОШИБКА: E048 $name (код $rc), очередь продолжается" ;;
  esac
}
{
  echo "=== $(date -Is) queue e048: host=$(hostname) commit=$(git rev-parse --short HEAD) ==="
  ( for s in fit:other fit_summary:other splits:other fit_summary:other; do stage "${s%%:*}" "${s##*:}"; done ) &
  closed=$!
  for s in control:control select:critical main:train; do stage "${s%%:*}" "${s##*:}" || { echo "ОШИБКА: E048 прерван на ${s%%:*}"; break; }; done
  wait "$closed"
  stage summarize other
  echo "=== $(date -Is) queue e048 завершена; строк ОШИБКА: $(grep -c '^ОШИБКА' "$LOG" 2>/dev/null || echo 0) ==="
} 2>&1 | tee -a "$LOG"
