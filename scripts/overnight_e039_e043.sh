#!/usr/bin/env bash
# Queue E039-E043. Usage (repository root, inside tmux; normally `bash scripts/remote.sh go e039 <part>`):
#   p0  E039 diagnostics (closed form, no training; normally run locally)
#   p1  E040 cascade of triplet stages (control, main, confirm, summary -> results/gate_e040.json)
#   p2  E041 mixed stages - only if gate_e040.json has cascade_better: true or gate_e040_decision.json run_e041: true
#   p3  E042 parametric coefficients - only if gate_e039.json has passed: true and experiments/E042_* exists
#   p4  E043 summary (no training)
#   all p1 -> p2 -> p3 -> p4 (each gated as above)
# Same gates as overnight_e024_e033.sh: results/gate_status.json halt, memory >= 16 GB, controls, > 15 % failed runs.
set -uo pipefail
set +H
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
part="${1:-}"
case "$part" in p0|p1|p2|p3|p4|all) ;; *) sed -n '2,10p' "$0"; exit 2 ;; esac
mkdir -p logs
LOG="logs/queue_e039_${part}_$(date +%Y%m%d_%H%M%S).log"
export THREADS=1 DEVICE=cpu OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
export PYTHON PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
gate() { "$PYTHON" -m constellation_denoiser.gate "$@"; }
flag() { "$PYTHON" -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if d.get('$2') else 1)" 2>/dev/null; }

stage() {   # stage <folder> <stage> <kind>
  local folder="$1" name="$2" kind="$3" rc
  gate halted || return 1
  gate memory || return 1
  JOBS="$(gate jobs)"; export JOBS
  echo "=== $(date -Is) $folder: stage $name ($kind) JOBS=$JOBS ==="
  bash "experiments/$folder/run.sh" "$name"; rc=$?
  case "$kind" in
    control|critical) [[ "$rc" == 0 ]] || { gate set --reason "$folder $name: код $rc"; return 1; } ;;
    train|search)
      if [[ "$rc" != 0 && "$rc" != 1 ]]; then gate set --reason "$folder $name: код $rc"; return 1; fi
      local f=""; [[ "$kind" == search ]] && f="--search"
      gate stage --exp "${folder:0:4}" --stage "$name" $f || return 1 ;;
    other) [[ "$rc" == 0 ]] || echo "ОШИБКА: $folder $name (код $rc), очередь продолжается" ;;
  esac
}

experiment() { local folder="$1" item; shift
  for item in "$@"; do stage "$folder" "${item%%:*}" "${item##*:}" || { echo "ОШИБКА: $folder прерван на ${item%%:*}"; return 1; }; done; }

run_p0() { experiment E039_triplet_diagnostics main:critical; }
run_p1() { experiment E040_triplet_cascade control:control main:search confirm:train summarize:critical; }
run_p2() {
  if flag results/gate_e040.json cascade_better || flag results/gate_e040_decision.json run_e041; then
    experiment E041_mixed_stages control:control s1:search s2:search s3:train summarize:other
  else
    echo "E041 не запущен: каскад E040 не лучше одношаговых (results/gate_e040.json); нужно решение автора (results/gate_e040_decision.json)"
  fi
}
run_p3() {
  local folder; folder="$(ls -d experiments/E042_* 2>/dev/null | head -1)"
  if flag results/gate_e039.json passed && [[ -n "$folder" ]]; then
    experiment "$(basename "$folder")" control:control main:train summarize:other
  else
    echo "E042 не запущен: правило E039 п. 3(б) не выполнено или эксперимент не создан (results/gate_e039.json)"
  fi
}
run_p4() { experiment E043_triplet_summary main:other; }

{
  echo "=== $(date -Is) queue e039 $part: host=$(hostname) commit=$(git rev-parse --short HEAD) ==="
  case "$part" in
    p0) run_p0 ;; p1) run_p1 ;; p2) run_p2 ;; p3) run_p3 ;; p4) run_p4 ;;
    all) run_p1 && run_p2 && run_p3; run_p4 ;;
  esac
  echo "=== $(date -Is) queue e039 $part завершена; строк ОШИБКА: $(grep -c '^ОШИБКА' "$LOG" 2>/dev/null || echo 0) ==="
} 2>&1 | tee -a "$LOG"
