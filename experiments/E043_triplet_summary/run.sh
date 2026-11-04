#!/usr/bin/env bash
# E043: summary and fair comparison (no training). Usage: bash experiments/E043_triplet_summary/run.sh <stage>
# Stages: main | estimate | smoke
set -euo pipefail
EXP=E043
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup
case "${1:-}" in
  main|all|summarize) rc_run main "$PYTHON" -u -m constellation_denoiser.summarize_E043 ;;
  estimate|plan) echo "Стадия main: без обучения; оценка: ~0.05 ч." ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"; rc_run main "$PYTHON" -u -m constellation_denoiser.summarize_E043 ;;
  *) sed -n '2,3p' "$0"; [[ -z "${1:-}" ]] && exit 0 || exit 2 ;;
esac
