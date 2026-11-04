#!/usr/bin/env bash
# E039: diagnostics of the triplets without training (local, closed form). Usage: bash experiments/E039_triplet_diagnostics/run.sh <stage>
# Stages: main | estimate | smoke
set -euo pipefail
EXP=E039
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup
case "${1:-}" in
  main|all) rc_lock main; rc_run main "$PYTHON" -u -m constellation_denoiser.e039_diagnostics all ;;
  estimate|plan) echo "Стадия main: без обучения, локально; оценка: ~0.5 ч (закрытая форма M = 31, 51 и предсказания на val/test)." ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"
         rc_run main "$PYTHON" -u -m constellation_denoiser.e039_diagnostics all ;;
  *) sed -n '2,3p' "$0"; [[ -z "${1:-}" ]] && exit 0 || exit 2 ;;
esac
