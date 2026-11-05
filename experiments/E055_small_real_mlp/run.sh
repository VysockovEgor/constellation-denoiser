#!/usr/bin/env bash
# E055: small real tanh MLP (one hidden layer, ~1-2k parameters), F10 recipe. Usage: bash experiments/E055_small_real_mlp/run.sh <stage>
# Stages: control | main | summarize | all | smoke | plan | estimate
set -euo pipefail
EXP=E055
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_STAGES=(control main)
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup
stage="${1:-}"
if rc_is_common "$stage"; then rc_common "$@"; exit 0; fi
case "$stage" in
  control) rc_runner control --no-control-gate; rc_common check-control ;;
  main) rc_runner main ;;
  summarize) rc_run summarize "$PYTHON" -m constellation_denoiser.summarize_E055 ;;
  all) rc_runner control --no-control-gate; rc_common check-control; rc_runner main
       rc_run summarize "$PYTHON" -m constellation_denoiser.summarize_E055 ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"
         rc_runner control --no-control-gate; rc_common check-control; rc_runner main
         rc_run summarize "$PYTHON" -m constellation_denoiser.summarize_E055 ;;
  *) sed -n '2,3p' "$0"; [[ -z "$stage" ]] && exit 0 || exit 2 ;;
esac
