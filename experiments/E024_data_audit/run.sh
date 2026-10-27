#!/usr/bin/env bash
# E024: audit of data, RM counting and aggregation. Usage: bash experiments/E024_data_audit/run.sh <stage>
# Stages: probe | plan <stage|all> | estimate | list <stage> | control | main (audit; dedup only if leakage) | summarize | smoke
# Env: JOBS=auto THREADS=auto DEVICE=auto SEEDS=17,29,43,57,71 SMOKE=0 FORCE=0
set -euo pipefail
EXP=E024
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_STAGES=(control dedup)
source "$EXP_DIR/../../scripts/run_common.sh"
rc_setup

stage="${1:-}"
if [[ -z "$stage" || "$stage" == "--help" || "$stage" == "-h" ]]; then
  sed -n '2,5p' "$0"
  exit 0
fi

audit() {
  rc_run audit "$PYTHON" -m constellation_denoiser.audit_e024 leakage data budget noise
  rc_run audit "$PYTHON" -m constellation_denoiser.rm_audit
  rc_run audit "$PYTHON" -m constellation_denoiser.audit_e024 pareto
}

leak() {   # 0 if results/leakage.json says there is leakage
  local root="$EXP_DIR/results"
  [[ "$SMOKE" == "1" ]] && root="$EXP_DIR/results_smoke"
  "$PYTHON" -c "import json,sys; sys.exit(0 if not json.load(open('$root/leakage.json'))['no_leakage'] else 1)"
}

main() {
  audit
  if leak; then
    echo "E024: утечка найдена -> dedup-проверка (v4 h119, 1 seed, 60k); очередь остановлена (results/gate_status.json)"
    rc_runner dedup --no-control-gate
  fi
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
  main) main ;;
  dedup) rc_runner dedup --no-control-gate ;;
  summarize) rc_run audit "$PYTHON" -m constellation_denoiser.audit_e024 pareto ;;
  smoke)
    export SMOKE=1
    LOG_DIR="$EXP_DIR/results_smoke/logs"
    mkdir -p "$LOG_DIR"
    rc_runner control --no-control-gate
    rc_common check-control
    audit
    rc_runner dedup --no-control-gate
    ;;
  *)
    echo "неизвестная стадия: $stage (см. bash $0 --help)" >&2
    exit 2
    ;;
esac
