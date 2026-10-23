#!/usr/bin/env bash
# E015: physics diagnostics (no training). Usage: bash experiments/E015_physics_diagnostics/run.sh <stage>
# Stages: diagnostics | smoke | plan | estimate | probe
# Env: SMOKE=0
set -euo pipefail
EXP=E015
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_STAGES=()
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup

stage="${1:-}"
if [[ -z "$stage" || "$stage" == "--help" || "$stage" == "-h" ]]; then
  sed -n '2,5p' "$0"
  exit 0
fi

case "$stage" in
  probe)
    rc_common probe
    ;;
  plan|estimate)
    echo "E015: без обучения; 1 прогон анализа (5 подгонок alpha по 30 train-файлам, бутстрап 200 по 45+45 файлам)."
    echo "Оценка: минуты (локальный smoke на 2/1/1 файлах ~10 с). Отдельный замер не нужен."
    ;;
  diagnostics)
    rc_lock diagnostics
    rc_run diagnostics "$PYTHON" -u -m constellation_denoiser.physics_diagnostics
    ;;
  smoke)
    export SMOKE=1
    LOG_DIR="$EXP_DIR/results_smoke/logs"
    mkdir -p "$LOG_DIR"
    rc_run diagnostics "$PYTHON" -u -m constellation_denoiser.physics_diagnostics
    ;;
  *)
    echo "неизвестная стадия: $stage (см. bash $0 --help)" >&2
    exit 2
    ;;
esac
