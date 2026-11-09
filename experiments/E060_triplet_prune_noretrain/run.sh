#!/usr/bin/env bash
# E060: pruning the ready triplets T (M = 15) without refitting. Usage: bash experiments/E060_triplet_prune_noretrain/run.sh <stage>
# Stages: main | smoke
set -euo pipefail
EXP=E060
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
case "${1:-}" in
  main) rc_lock main; rc_run main "$PYTHON" -u -m constellation_denoiser.e060_triplet_prune ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"
         rc_run main "$PYTHON" -u -m constellation_denoiser.e060_triplet_prune ;;
  *) sed -n '2,3p' "$0"; [[ -z "${1:-}" ]] && exit 0 || exit 2 ;;
esac
