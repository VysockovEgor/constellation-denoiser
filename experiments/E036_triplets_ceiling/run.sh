#!/usr/bin/env bash
# E036: ceiling of the triplets (M = 41, 51; pruning) (closed form, CPU). Usage: bash experiments/E036_triplets_ceiling/run.sh <stage>
# Stages: main (fit + summarize) | fit | summarize | estimate | smoke
# Env: SMOKE=0 (SMOKE=1: 2/1/1 files, M = 5, one replica)
set -euo pipefail
EXP=E036
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup

stage="${1:-}"
case "$stage" in
  main|all) rc_lock main; rc_run main "$PYTHON" -u -m constellation_denoiser.summarize_E036 all ;;
  fit) rc_lock main; rc_run main "$PYTHON" -u -m constellation_denoiser.summarize_E036 fit ;;
  summarize) rc_run summarize "$PYTHON" -u -m constellation_denoiser.summarize_E036 summarize ;;
  estimate|plan) echo "Стадия main: без обучения по шагам (МНК в закрытой форме), оценка: ~3 ч (BLAS 16 потоков; M = 41, 51, отбор из пула M = 51)." ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"
         rc_run main "$PYTHON" -u -m constellation_denoiser.summarize_E036 all ;;
  *) sed -n '2,4p' "$0"; [[ -z "$stage" ]] && exit 0 || exit 2 ;;
esac
