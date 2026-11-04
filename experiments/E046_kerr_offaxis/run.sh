#!/usr/bin/env bash
# E046: architecture A (Kerr with memory + off-axis triplets), closed form + L-BFGS. Usage: bash experiments/E046_kerr_offaxis/run.sh <stage>
# Stages: main (all fits) | splits (best per arch on the E023 splits) | summarize | estimate | smoke
set -euo pipefail
EXP=E046
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
case "${1:-}" in
  main) rc_lock main; rc_run main "$PYTHON" -u -m constellation_denoiser.e046_fit all ;;
  splits) rc_lock splits; rc_run splits "$PYTHON" -u -m constellation_denoiser.e046_fit splits ;;
  summarize) rc_run summarize "$PYTHON" -u -m constellation_denoiser.e046_fit summarize ;;
  estimate|plan) echo "Стадия main: 5 реплик x (3 M x (3 Nc x 6 конфигураций + 1) + 3 Kerr-only) = 300 подгонок; оценка: ~3 ч при JOBS 8 (подгонка 1-10 мин на 1 ядре)." ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"
         rc_run main "$PYTHON" -u -m constellation_denoiser.e046_fit all
         rc_run main "$PYTHON" -u -m constellation_denoiser.e046_fit summarize
         rc_run main "$PYTHON" -u -m constellation_denoiser.e046_fit splits
         rc_run main "$PYTHON" -u -m constellation_denoiser.e046_fit summarize ;;
  *) sed -n '2,3p' "$0"; [[ -z "${1:-}" ]] && exit 0 || exit 2 ;;
esac
