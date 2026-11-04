#!/usr/bin/env bash
# E046 (architecture A, closed form + L-BFGS) then E047 (cascade variants), next to the running E039-E043 queue.
# Usage: bash scripts/remote.sh go e044 all. JOBS_ADAPTIVE=1: the number of processes follows the free cores
# (all running threads counted, ours included), no upper cap beyond the machine.
set -uo pipefail
set +H
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs
LOG="logs/queue_e044_$(date +%Y%m%d_%H%M%S).log"
export THREADS=1 DEVICE=cpu JOBS_ADAPTIVE=1
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
export PYTHON PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
gate() { "$PYTHON" -m constellation_denoiser.gate "$@"; }
stage() {   # stage <folder> <stage> <kind>
  local folder="$1" name="$2" kind="$3" rc
  gate halted || return 1
  gate memory || return 1
  JOBS="$(gate jobs --total)"; export JOBS
  echo "=== $(date -Is) $folder: stage $name ($kind) JOBS=$JOBS ==="
  bash "experiments/$folder/run.sh" "$name"; rc=$?
  case "$kind" in
    control|critical) [[ "$rc" == 0 ]] || { gate set --reason "$folder $name: код $rc"; return 1; } ;;
    train) if [[ "$rc" != 0 && "$rc" != 1 ]]; then gate set --reason "$folder $name: код $rc"; return 1; fi
           gate stage --exp "${folder:0:4}" --stage "$name" || return 1 ;;
    other) [[ "$rc" == 0 ]] || echo "ОШИБКА: $folder $name (код $rc), очередь продолжается" ;;
  esac
}
{
  echo "=== $(date -Is) queue e044: host=$(hostname) commit=$(git rev-parse --short HEAD) ==="
  for item in main:other summarize:other splits:other summarize:other; do stage E046_kerr_offaxis "${item%%:*}" "${item##*:}"; done
  for item in control:control init:critical main:train summarize:other; do
    stage E047_cascade_variants "${item%%:*}" "${item##*:}" || { echo "ОШИБКА: E047 прерван на ${item%%:*}"; break; }
  done
  echo "=== $(date -Is) queue e044 завершена; строк ОШИБКА: $(grep -c '^ОШИБКА' "$LOG" 2>/dev/null || echo 0) ==="
} 2>&1 | tee -a "$LOG"
