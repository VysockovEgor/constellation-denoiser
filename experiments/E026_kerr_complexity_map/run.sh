#!/usr/bin/env bash
# E026: complexity map of the Kerr MLP. Usage: bash experiments/E026_kerr_complexity_map/run.sh <stage>
# Stages: probe | plan <stage|all> | estimate [stage] | list <stage> | control | grid | confirm | all | summarize | smoke
# Env: JOBS=auto THREADS=auto DEVICE=auto SEEDS=17,29,43,57,71 SMOKE=0 FORCE=0; plan/estimate/list of stages that
# depend on an earlier one use PLACEHOLDER_SELECTION=1 (first configurations) until that stage has results.
set -euo pipefail
EXP=E026
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_STAGES=(control grid confirm)
STAGES=(grid confirm)
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup

stage="${1:-}"
if [[ -z "$stage" || "$stage" == "--help" || "$stage" == "-h" ]]; then
  sed -n '2,6p' "$0"
  exit 0
fi

summarize() {
  rc_run summarize "$PYTHON" -m constellation_denoiser.summarize_E026
}

one() {   # one stage of the chain
  case "$1" in
    *) rc_runner "$1" ;;
  esac
}

if rc_is_common "$stage"; then
  export PLACEHOLDER_SELECTION="${PLACEHOLDER_SELECTION:-1}"
  rc_common "$@"
  exit 0
fi

case "$stage" in
  control)
    rc_runner control --no-control-gate
    rc_common check-control
    ;;
  all)
    for s in "${STAGES[@]}"; do one "$s"; done
    summarize
    ;;
  summarize) summarize ;;
  smoke)
    export SMOKE=1
    LOG_DIR="$EXP_DIR/results_smoke/logs"
    mkdir -p "$LOG_DIR"
    rc_runner control --no-control-gate
    rc_common check-control
    for s in "${STAGES[@]}"; do one "$s"; done
    summarize
    ;;
  *)
    if [[ " ${STAGES[*]} " == *" $stage "* ]]; then
      one "$stage"
    else
      echo "неизвестная стадия: $stage (см. bash $0 --help)" >&2
      exit 2
    fi
    ;;
esac
