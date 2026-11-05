#!/usr/bin/env bash
# E053 (chirps, closed form): waits while the E048 queue runs and fewer than 4 cores are free; then control, fit, splits, summarize.
# Usage: bash scripts/remote.sh go e053 all. JOBS_ADAPTIVE=1 and POLITE_EXCLUDE_OWN=1 as in E048.
set -uo pipefail
set +H
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs
LOG="logs/queue_e053_$(date +%Y%m%d_%H%M%S).log"
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
  echo "=== $(date -Is) E053_chirp_triplets: stage $name ($kind) JOBS=$JOBS ==="
  bash experiments/E053_chirp_triplets/run.sh "$name"; rc=$?
  case "$kind" in
    control|critical) [[ "$rc" == 0 ]] || { gate set --reason "E053 $name: код $rc"; return 1; } ;;
    train) if [[ "$rc" != 0 && "$rc" != 1 ]]; then gate set --reason "E053 $name: код $rc"; return 1; fi
           gate stage --exp E053 --stage "$name" || return 1 ;;
    other) [[ "$rc" == 0 ]] || echo "ОШИБКА: E053 $name (код $rc), очередь продолжается" ;;
  esac
}
{
  echo "=== $(date -Is) queue e053: host=$(hostname) commit=$(git rev-parse --short HEAD) ==="
  while pgrep -u "$(id -u)" -f "overnight_e048.sh" > /dev/null && [[ "$(gate jobs --total 2>/dev/null)" -lt 4 ]]; do
    echo "$(date -Is) E048 идёт и свободных ядер меньше 4: жду"; sleep 600
  done
  if stage control control; then
    for s in fit:other splits:other summarize:other; do stage "${s%%:*}" "${s##*:}"; done
  else
    echo "ОШИБКА: E053 прерван на control"
  fi
  echo "=== $(date -Is) queue e053 завершена; строк ОШИБКА: $(grep -c '^ОШИБКА' "$LOG" 2>/dev/null || echo 0) ==="
} 2>&1 | tee -a "$LOG"
