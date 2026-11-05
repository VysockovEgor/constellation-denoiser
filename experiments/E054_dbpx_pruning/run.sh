#!/usr/bin/env bash
# E054: DBP-X with trainable FIR, progressive pruning 255 -> 7. Usage: bash experiments/E054_dbpx_pruning/run.sh <stage>
# Stages: control | phase1 | gate | phase2 | fit (all arms) | summarize | smoke
set -euo pipefail
EXP=E054
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_STAGES=(control)
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
stage="${1:-}"
if rc_is_common "$stage"; then rc_common "$@"; exit 0; fi
case "$stage" in
  control) rc_runner control --no-control-gate; rc_common check-control ;;
  fit) rc_run fit "$PYTHON" -u -m constellation_denoiser.e054_prune all ;;
  phase1) rc_run phase1 "$PYTHON" -u -m constellation_denoiser.e054_prune phase1 ;;
  gate) rc_run gate "$PYTHON" -u -m constellation_denoiser.e054_prune gate ;;
  phase2) rc_run phase2 "$PYTHON" -u -m constellation_denoiser.e054_prune phase2 ;;
  summarize) rc_run summarize "$PYTHON" -m constellation_denoiser.e054_prune summarize ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"
         rc_runner control --no-control-gate; rc_common check-control
         rc_run fit "$PYTHON" -u -m constellation_denoiser.e054_prune all
         rc_run summarize "$PYTHON" -m constellation_denoiser.e054_prune summarize ;;
  *) sed -n '2,3p' "$0"; [[ -z "$stage" ]] && exit 0 || exit 2 ;;
esac
