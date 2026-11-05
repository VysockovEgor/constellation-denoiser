#!/usr/bin/env bash
# E050: DBP-X (whole-line dispersion added back, then Ns x [linear step, Kerr phase]). Usage: bash experiments/E050_dbpx_physical/run.sh <stage>
# Stages: unittest | start | control | train | ref | summarize | all | smoke | plan | estimate
set -euo pipefail
EXP=E050
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_STAGES=(control ref)
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup
stage="${1:-}"
dbpx() { rc_run "$1" "$PYTHON" -u -m constellation_denoiser.dbpx "$1"; }
summ() { rc_run summarize "$PYTHON" -m constellation_denoiser.summarize_E050; }
if rc_is_common "$stage"; then rc_common "$@"; exit 0; fi
case "$stage" in
  unittest) dbpx unittest ;;
  start) dbpx start ;;
  control) rc_runner control --no-control-gate; rc_common check-control ;;
  train) dbpx train ;;
  ref) rc_runner ref ;;
  summarize) summ ;;
  all) dbpx unittest; dbpx start; rc_runner control --no-control-gate; rc_common check-control
       dbpx train; rc_runner ref; summ ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"
         dbpx unittest; dbpx start; rc_runner control --no-control-gate; rc_common check-control
         dbpx train; rc_runner ref; summ ;;
  *) sed -n '2,3p' "$0"; [[ -z "$stage" ]] && exit 0 || exit 2 ;;
esac
