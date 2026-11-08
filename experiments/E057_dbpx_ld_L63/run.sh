#!/usr/bin/env bash
# E057: DBP-X Ns = 20, direct_ld (joint LS start of the cascade) at L = 63. Usage: bash experiments/E057_dbpx_ld_L63/run.sh <stage>
# Stages: control | fit | summarize | smoke
set -euo pipefail
EXP=E057
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_STAGES=(control)
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
stage="${1:-}"
if rc_is_common "$stage"; then rc_common "$@"; exit 0; fi
case "$stage" in
  control) rc_runner control --no-control-gate; rc_common check-control ;;
  fit) rc_run fit "$PYTHON" -u -m constellation_denoiser.e057_ld63 fit ;;
  summarize) rc_run summarize "$PYTHON" -m constellation_denoiser.e057_ld63 summarize ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"
         rc_runner control --no-control-gate; rc_common check-control
         rc_run fit "$PYTHON" -u -m constellation_denoiser.e057_ld63 fit
         rc_run summarize "$PYTHON" -m constellation_denoiser.e057_ld63 summarize ;;
  *) sed -n '2,3p' "$0"; [[ -z "$stage" ]] && exit 0 || exit 2 ;;
esac
