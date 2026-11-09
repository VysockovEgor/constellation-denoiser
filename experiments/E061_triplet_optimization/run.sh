#!/usr/bin/env bash
# E061: triplets T3 + FIR7 (M = 15) made cheaper (tying, pruning with refit). Usage: bash experiments/E061_triplet_optimization/run.sh <stage>
# Stages: main | plot | smoke
set -euo pipefail
EXP=E061
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
case "${1:-}" in
  main) rc_lock main; rc_run main "$PYTHON" -u -m constellation_denoiser.e061_triplet_opt ;;
  plot) PLOT_ONLY=1 rc_run plot "$PYTHON" -m constellation_denoiser.e061_triplet_opt ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"
         rc_run main "$PYTHON" -u -m constellation_denoiser.e061_triplet_opt ;;
  *) sed -n '2,3p' "$0"; [[ -z "${1:-}" ]] && exit 0 || exit 2 ;;
esac
