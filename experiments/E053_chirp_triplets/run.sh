#!/usr/bin/env bash
# E053: off-axis triplet table as a sum of chirps (H1) and recurrences (H2). Usage: bash experiments/E053_chirp_triplets/run.sh <stage>
# Stages: control | table-fit | fit | splits | summarize | smoke
set -euo pipefail
EXP=E053
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_STAGES=(control)
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
stage="${1:-}"
if rc_is_common "$stage"; then rc_common "$@"; exit 0; fi
case "$stage" in
  control) rc_runner control --no-control-gate; rc_common check-control ;;
  table-fit) rc_run table_fit "$PYTHON" -u -m constellation_denoiser.e053_table_fit ;;
  fit) rc_run fit "$PYTHON" -u -m constellation_denoiser.e053_data all ;;
  splits) rc_run splits "$PYTHON" -u -m constellation_denoiser.e053_data splits ;;
  summarize) rc_run summarize "$PYTHON" -m constellation_denoiser.summarize_E053 ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"
         rc_runner control --no-control-gate; rc_common check-control
         rc_run fit "$PYTHON" -u -m constellation_denoiser.e053_data all
         rc_run splits "$PYTHON" -u -m constellation_denoiser.e053_data splits
         rc_run summarize "$PYTHON" -m constellation_denoiser.summarize_E053 ;;
  *) sed -n '2,3p' "$0"; [[ -z "$stage" ]] && exit 0 || exit 2 ;;
esac
