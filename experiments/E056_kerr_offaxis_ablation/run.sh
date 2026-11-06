#!/usr/bin/env bash
# E056: ablation Kerr memory vs off-axis triplets (closed form). Usage: bash experiments/E056_kerr_offaxis_ablation/run.sh <stage>
# Stages: fit | control | summarize | smoke
set -euo pipefail
EXP=E056
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_STAGES=(control)
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup
export TRIPLET_KERR_EXP=E056 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
stage="${1:-}"
if rc_is_common "$stage"; then export PLACEHOLDER_SELECTION=1; rc_common "$@"; exit 0; fi
case "$stage" in
  fit) rc_lock fit; rc_run fit "$PYTHON" -u -m constellation_denoiser.e046_fit all ;;
  control) rc_runner control --no-control-gate; rc_common check-control ;;
  summarize) rc_run summarize "$PYTHON" -m constellation_denoiser.summarize_E056 ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"
         rc_run fit "$PYTHON" -u -m constellation_denoiser.e046_fit all
         rc_runner control --no-control-gate; rc_common check-control
         rc_run summarize "$PYTHON" -m constellation_denoiser.summarize_E056 ;;
  *) sed -n '2,3p' "$0"; [[ -z "$stage" ]] && exit 0 || exit 2 ;;
esac
