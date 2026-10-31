#!/usr/bin/env bash
# E034 (triplets: larger M, pruning; closed form) and E035 (triplets by gradient, + Kerr MLP), in parallel with the
# E024-E033 queue. Started by `bash scripts/remote.sh go q x` as a tmux session named q_conv_<date>, so that the main
# queue reserves 16 cores for it and E033 (p5) waits for its results. Same gates as overnight_e024_e033.sh.
set -uo pipefail
set +H
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs
LOG="logs/queue_x_$(date +%Y%m%d_%H%M%S).log"
export THREADS=1 DEVICE=cpu OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
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
  JOBS="$(gate jobs --max 16)"; export JOBS
  echo "=== $(date -Is) $folder: stage $name ($kind) JOBS=$JOBS ==="
  bash "experiments/$folder/run.sh" "$name"; rc=$?
  case "$kind" in
    control|critical) [[ "$rc" == 0 ]] || { gate set --reason "$folder $name: код $rc"; return 1; } ;;
    train|search)
      if [[ "$rc" != 0 && "$rc" != 1 ]]; then gate set --reason "$folder $name: код $rc"; return 1; fi
      local flag=""; [[ "$kind" == search ]] && flag="--search"
      gate stage --exp "${folder:0:4}" --stage "$name" $flag || return 1 ;;
    other) [[ "$rc" == 0 ]] || echo "ОШИБКА: $folder $name (код $rc), очередь продолжается" ;;
  esac
}

{
  echo "=== $(date -Is) queue x (E034, E035): host=$(hostname) commit=$(git rev-parse --short HEAD) ==="
  stage E034_triplets_extended main other
  for item in control:control fit:critical s1:search s2:train summarize:other; do
    stage E035_triplet_hybrid "${item%%:*}" "${item##*:}" || { echo "ОШИБКА: E035 прерван на ${item%%:*}"; break; }
  done
  echo "=== $(date -Is) queue x завершена; строк ОШИБКА: $(grep -c '^ОШИБКА' "$LOG" 2>/dev/null || echo 0) ==="
} 2>&1 | tee -a "$LOG"
