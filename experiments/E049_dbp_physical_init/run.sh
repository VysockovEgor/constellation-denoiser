#!/usr/bin/env bash
# E049: DBP physical initialization. Usage: bash experiments/E049_dbp_physical_init/run.sh <stage>
# Stages: calib (fit c on trained DBP checkpoints) | control | main | summarize | all | smoke | plan | estimate
set -euo pipefail
EXP=E049
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_STAGES=(control main)
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup
stage="${1:-}"
if rc_is_common "$stage"; then export PLACEHOLDER_SELECTION=1; rc_common "$@"; exit 0; fi
case "$stage" in
  calib) rc_run calib "$PYTHON" -u -m constellation_denoiser.e049_physical calib ;;
  control) rc_runner control --no-control-gate; rc_common check-control ;;
  main) rc_runner main ;;
  summarize) rc_run summarize "$PYTHON" -m constellation_denoiser.summarize_E049 ;;
  all) rc_run calib "$PYTHON" -u -m constellation_denoiser.e049_physical calib
       rc_runner control --no-control-gate; rc_common check-control; rc_runner main
       rc_run summarize "$PYTHON" -m constellation_denoiser.summarize_E049 ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"
         rc_run calib "$PYTHON" -u -m constellation_denoiser.e049_physical calib
         rc_runner control --no-control-gate; rc_common check-control; rc_runner main
         rc_run summarize "$PYTHON" -m constellation_denoiser.summarize_E049 ;;
  *) sed -n '2,3p' "$0"; [[ -z "$stage" ]] && exit 0 || exit 2 ;;
esac
