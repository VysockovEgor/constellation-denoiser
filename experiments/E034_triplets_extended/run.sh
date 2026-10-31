#!/usr/bin/env bash
# E034: triplets with larger M and pruning (closed form, CPU). Usage: bash experiments/E034_triplets_extended/run.sh <stage>
# Stages: main (fit + summarize) | fit | summarize | estimate | smoke
# Env: SMOKE=0 (SMOKE=1: 2/1/1 files, M = 5, one replica)
set -euo pipefail
EXP=E034
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup

stage="${1:-}"
case "$stage" in
  main|all) rc_lock main; rc_run main "$PYTHON" -u -m constellation_denoiser.summarize_E034 all ;;
  fit) rc_lock main; rc_run main "$PYTHON" -u -m constellation_denoiser.summarize_E034 fit ;;
  summarize) rc_run summarize "$PYTHON" -u -m constellation_denoiser.summarize_E034 summarize ;;
  estimate|plan) echo "Стадия main: без обучения по шагам (МНК в закрытой форме), оценка: ~1.5 ч (BLAS 16 потоков; M = 25, 31 и отбор K членов, 5 реплик)." ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"
         rc_run main "$PYTHON" -u -m constellation_denoiser.summarize_E034 all ;;
  *) sed -n '2,4p' "$0"; [[ -z "$stage" ]] && exit 0 || exit 2 ;;
esac
