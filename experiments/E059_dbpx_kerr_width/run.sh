#!/usr/bin/env bash
# E059: trained DBP-X, Kerr window R = 1, 3, 7, 15 at Ns = 8, 20. Usage: bash experiments/E059_dbpx_kerr_width/run.sh <stage>
# Stages: control | fit | summarize | weights | smoke
set -euo pipefail
EXP=E059
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_STAGES=(control)
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
stage="${1:-}"
if rc_is_common "$stage"; then rc_common "$@"; exit 0; fi
case "$stage" in
  control) rc_runner control --no-control-gate; rc_common check-control ;;
  fit) rc_run fit "$PYTHON" -u -m constellation_denoiser.e059_kerr_width train ;;
  summarize) rc_run summarize "$PYTHON" -m constellation_denoiser.e059_kerr_width summarize ;;
  weights) rc_run weights "$PYTHON" -m constellation_denoiser.e059_kerr_weights ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"
         rc_runner control --no-control-gate; rc_common check-control
         rc_run fit "$PYTHON" -u -m constellation_denoiser.e059_kerr_width train
         rc_run summarize "$PYTHON" -m constellation_denoiser.e059_kerr_width summarize ;;
  *) sed -n '2,3p' "$0"; [[ -z "$stage" ]] && exit 0 || exit 2 ;;
esac
