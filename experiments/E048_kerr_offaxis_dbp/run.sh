#!/usr/bin/env bash
# E048: architecture A at larger memory (closed form) and DBP + off-axis triplets. Usage: bash experiments/E048_kerr_offaxis_dbp/run.sh <stage>
# Stages: fit | fit_summary | splits | control | select | main | summarize | estimate | smoke
set -euo pipefail
EXP=E048
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_STAGES=(control main)
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup
export TRIPLET_KERR_EXP=E048 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
stage="${1:-}"
if rc_is_common "$stage"; then export PLACEHOLDER_SELECTION=1; rc_common "$@"; exit 0; fi
case "$stage" in
  fit) rc_lock fit; rc_run fit "$PYTHON" -u -m constellation_denoiser.e046_fit all ;;
  fit_summary) rc_run fit "$PYTHON" -u -m constellation_denoiser.e046_fit summarize ;;
  splits) rc_lock splits; rc_run splits "$PYTHON" -u -m constellation_denoiser.e046_fit splits ;;
  control) rc_runner control --no-control-gate; rc_common check-control ;;
  select) rc_run select "$PYTHON" -m constellation_denoiser.runner --exp "$EXP" --stage select ;;
  main) rc_runner main ;;
  summarize) rc_run summarize "$PYTHON" -m constellation_denoiser.summarize_E048 ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"
         rc_run fit "$PYTHON" -u -m constellation_denoiser.e046_fit all
         rc_run fit "$PYTHON" -u -m constellation_denoiser.e046_fit summarize
         rc_runner control --no-control-gate; rc_common check-control
         rc_run select "$PYTHON" -m constellation_denoiser.runner --exp "$EXP" --stage select
         rc_runner main
         rc_run summarize "$PYTHON" -m constellation_denoiser.summarize_E048 ;;
  *) sed -n '2,3p' "$0"; [[ -z "$stage" ]] && exit 0 || exit 2 ;;
esac
