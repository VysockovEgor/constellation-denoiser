#!/usr/bin/env bash
# E012: depth vs window. Usage: bash experiments/E012_depth_vs_window/run.sh <stage>
# Stages: probe | plan <stage|all> | list <stage> | control | const | best | summarize | smoke
# Env: JOBS=auto THREADS=auto DEVICE=auto SEEDS=17,29,43 SMOKE=0 FORCE=0
set -euo pipefail
EXP=E012
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_STAGES=(control const best)
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup

stage="${1:-}"
if [[ -z "$stage" || "$stage" == "--help" || "$stage" == "-h" ]]; then
  sed -n '2,5p' "$0"
  exit 0
fi

summarize() {
  rc_run summarize "$PYTHON" -m constellation_denoiser.summarize_E012
}

if rc_is_common "$stage"; then
  rc_common "$@"
  exit 0
fi

case "$stage" in
  control)
    rc_runner control --no-control-gate
    rc_common check-control
    ;;
  const|best)
    rc_runner "$stage"
    ;;
  summarize)
    summarize
    ;;
  smoke)
    export SMOKE=1
    LOG_DIR="$EXP_DIR/results_smoke/logs"
    mkdir -p "$LOG_DIR"
    rc_runner control --no-control-gate
    rc_common check-control
    rc_runner const
    rc_runner best
    summarize
    ;;
  *)
    echo "неизвестная стадия: $stage (см. bash $0 --help)" >&2
    exit 2
    ;;
esac
