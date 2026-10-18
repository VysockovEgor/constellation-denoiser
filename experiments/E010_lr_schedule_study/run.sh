#!/usr/bin/env bash
# E010: lr schedules. Usage: bash experiments/E010_lr_schedule_study/run.sh <stage>
# Stages: probe | plan <stage|all> | list <stage> | baseline-check | stage1 | select | stage2 | stage3
#         | summarize | smoke
# Env: JOBS=auto THREADS=auto DEVICE=auto SEEDS=17,29,43 SMOKE=0 FORCE=0
set -euo pipefail
EXP=E010
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_STAGES=(baseline-check stage1 stage2 stage3)
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup

stage="${1:-}"
if [[ -z "$stage" || "$stage" == "--help" || "$stage" == "-h" ]]; then
  sed -n '2,5p' "$0"
  exit 0
fi

summarize() {
  rc_run summarize "$PYTHON" -m constellation_denoiser.merge_runs --exp "$EXP" --out lr_sweep.csv
  rc_run summarize "$PYTHON" -m constellation_denoiser.summarize_E010
}

if rc_is_common "$stage"; then
  rc_common "$@"
  exit 0
fi

case "$stage" in
  baseline-check)
    rc_runner baseline-check --no-control-gate
    rc_common check-control
    ;;
  stage1|stage2|stage3)
    rc_runner "$stage"
    ;;
  select)
    rc_run select "$PYTHON" -m constellation_denoiser.runner --exp "$EXP" --stage select
    ;;
  summarize)
    summarize
    ;;
  smoke)
    export SMOKE=1
    LOG_DIR="$EXP_DIR/results_smoke/logs"
    mkdir -p "$LOG_DIR"
    rc_runner baseline-check --no-control-gate
    rc_common check-control
    rc_runner stage1
    rc_run select "$PYTHON" -m constellation_denoiser.runner --exp "$EXP" --stage select
    rc_runner stage2
    rc_runner stage3
    summarize
    ;;
  *)
    echo "неизвестная стадия: $stage (см. bash $0 --help)" >&2
    exit 2
    ;;
esac
