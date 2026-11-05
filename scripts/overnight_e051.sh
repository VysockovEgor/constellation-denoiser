#!/usr/bin/env bash
# E051 (strict A-M, closed form) after E048: waits until the E048 queue has finished, then control, fit, fit_summary, splits, summarize.
# Usage: bash scripts/remote.sh go e051 all. JOBS_ADAPTIVE=1 and POLITE_EXCLUDE_OWN=1 as in E048.
set -uo pipefail
set +H
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs
LOG="logs/queue_e051_$(date +%Y%m%d_%H%M%S).log"
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
  echo "=== $(date -Is) E051_am_liang_kumar: stage $name ($kind) JOBS=$JOBS ==="
  bash experiments/E051_am_liang_kumar/run.sh "$name"; rc=$?
  case "$kind" in
    control|critical) [[ "$rc" == 0 ]] || { gate set --reason "E051 $name: код $rc"; return 1; } ;;
    train) if [[ "$rc" != 0 && "$rc" != 1 ]]; then gate set --reason "E051 $name: код $rc"; return 1; fi
           gate stage --exp E051 --stage "$name" || return 1 ;;
    other) [[ "$rc" == 0 ]] || echo "ОШИБКА: E051 $name (код $rc), очередь продолжается" ;;
  esac
}
{
  echo "=== $(date -Is) queue e051: host=$(hostname) commit=$(git rev-parse --short HEAD) ==="
  while pgrep -u "$(id -u)" -f "overnight_e048.sh" > /dev/null; do
    echo "$(date -Is) жду окончания очереди E048"; sleep 600
  done
  if stage control control; then
    for s in fit:other fit_summary:other splits:other summarize:other; do stage "${s%%:*}" "${s##*:}"; done
  else
    echo "ОШИБКА: E051 прерван на control"
  fi
  echo "=== $(date -Is) queue e051 завершена; строк ОШИБКА: $(grep -c '^ОШИБКА' "$LOG" 2>/dev/null || echo 0) ==="
} 2>&1 | tee -a "$LOG"
