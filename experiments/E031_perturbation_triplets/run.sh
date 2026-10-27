#!/usr/bin/env bash
# E031: perturbation triplets in closed form (CPU, minutes). Usage: bash experiments/E031_perturbation_triplets/run.sh <stage>
# Stages: main (fit + summarize) | fit | summarize | estimate | smoke
# Env: SMOKE=0 (SMOKE=1: 2/1/1 files, M = 3, 5, one bootstrap replica)
set -euo pipefail
EXP=E031
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup

stage="${1:-}"
case "$stage" in
  main|all) rc_lock main; rc_run main "$PYTHON" -u -m constellation_denoiser.summarize_E031 all ;;
  fit) rc_lock main; rc_run main "$PYTHON" -u -m constellation_denoiser.summarize_E031 fit ;;
  summarize) rc_run summarize "$PYTHON" -u -m constellation_denoiser.summarize_E031 summarize ;;
  estimate|plan) echo "Стадия main: без обучения по шагам (МНК в закрытой форме), оценка: ~0.5 ч на 1 ядре (6 значений M x 5 рук x 5 реплик)." ;;
  smoke) export SMOKE=1; LOG_DIR="$EXP_DIR/results_smoke/logs"; mkdir -p "$LOG_DIR"
         rc_run main "$PYTHON" -u -m constellation_denoiser.summarize_E031 all ;;
  *) sed -n '2,4p' "$0"; [[ -z "$stage" ]] && exit 0 || exit 2 ;;
esac
