#!/usr/bin/env bash
# E036 (triplets ceiling) and E037 (triplets on three splits), closed form on CPU (BLAS 16 threads), in parallel with
# E038 (DBP + selected triplets, training; JOBS = free cores minus 16). Started by `bash scripts/remote.sh go q y`.
# Same gates as overnight_e024_e033.sh (halt file, memory, > 15 % failed runs, controls).
set -uo pipefail
set +H
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs
LOG="logs/queue_y_$(date +%Y%m%d_%H%M%S).log"
export THREADS=1 DEVICE=cpu
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
export PYTHON PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
gate() { "$PYTHON" -m constellation_denoiser.gate "$@"; }

stage() {   # stage <folder> <stage> <kind> [jobs args]
  local folder="$1" name="$2" kind="$3" rc
  shift 3
  gate halted || return 1
  gate memory || return 1
  JOBS="$(gate jobs "$@")"; export JOBS
  echo "=== $(date -Is) $folder: stage $name ($kind) JOBS=$JOBS ==="
  bash "experiments/$folder/run.sh" "$name"; rc=$?
  case "$kind" in
    control|critical) [[ "$rc" == 0 ]] || { gate set --reason "$folder $name: код $rc"; return 1; } ;;
    train)
      if [[ "$rc" != 0 && "$rc" != 1 ]]; then gate set --reason "$folder $name: код $rc"; return 1; fi
      gate stage --exp "${folder:0:4}" --stage "$name" || return 1 ;;
    other) [[ "$rc" == 0 ]] || echo "ОШИБКА: $folder $name (код $rc), очередь продолжается" ;;
  esac
}

{
  echo "=== $(date -Is) queue y (E036, E037, E038): host=$(hostname) commit=$(git rev-parse --short HEAD) ==="
  (
    export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
    stage E037_triplets_splits main other --max 1
    stage E036_triplets_ceiling main other --max 1
  ) &
  closed=$!
  for item in control:control select:critical main:train summarize:other; do
    stage E038_dbp_triplets "${item%%:*}" "${item##*:}" --reserve 16 || { echo "ОШИБКА: E038 прерван на ${item%%:*}"; break; }
  done
  wait "$closed"
  echo "=== $(date -Is) queue y завершена; строк ОШИБКА: $(grep -c '^ОШИБКА' "$LOG" 2>/dev/null || echo 0) ==="
} 2>&1 | tee -a "$LOG"
