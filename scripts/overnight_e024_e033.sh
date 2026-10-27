#!/usr/bin/env bash
# Queue E024-E033. Usage (from the repository root, inside tmux; normally via `bash scripts/remote.sh go q <part>`):
#   p0   E024 (audit) + E031 (triplets, closed form)        p1   E025 (tuning; then starts E032 in tmux q_conv)
#   p2   E026 (Kerr map) + E028 (DBP depth)                  p3   E027 (activations) + E029 (front-end vs width)
#   p4   E030 (losses)                                       p5   E033 (final Pareto; waits for q_conv)
#   conv E032 (1M steps, JOBS 16)                            all  p0 -> p1 -> p2 -> p3 -> p4 -> p5 (+ conv in parallel)
# Before every stage: results/gate_status.json (halt: true stops the queue), free memory >= 16 GB (else halt after
# 30 min), JOBS = min(40, cores - foreign load - 2) (24 and 16 reserved while q_conv runs). After a training stage:
# halt if more than 15 % of its runs failed or diverged (tuning/map stages: failures only). A control outside its
# tolerance halts the queue. `all` first waits until the E020-E023 tmux sessions (e020_*) have ended.
set -uo pipefail
set +H
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
part="${1:-}"
case "$part" in p0|p1|p2|p3|p4|p5|conv|all) ;; *) sed -n '2,11p' "$0"; exit 2 ;; esac
mkdir -p logs logs/remote
LOG="logs/queue_${part}_$(date +%Y%m%d_%H%M%S).log"
export THREADS="${THREADS:-1}" DEVICE="${DEVICE:-cpu}"
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
export PYTHON PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
gate() { "$PYTHON" -m constellation_denoiser.gate "$@"; }

conv_running() { tmux ls -F '#{session_name}' 2>/dev/null | grep -qE '^q_conv_[0-9]{4}_[0-9]{4}$'; }

set_jobs() {
  if [[ "$part" == "conv" ]]; then
    JOBS="$(gate jobs --max 16)"
  elif conv_running; then
    JOBS="$(gate jobs --reserve 16 --max 24)"
  else
    JOBS="$(gate jobs)"
  fi
  export JOBS
}

halt() { gate set --reason "$1" --detail "${2:-}"; }

# stage <folder> <stage> <kind>: kind = control | train | search | other | critical
stage() {
  local folder="$1" name="$2" kind="$3" rc
  if ! gate halted; then return 1; fi
  if ! gate memory; then return 1; fi
  set_jobs
  echo "=== $(date -Is) $folder: stage $name ($kind) JOBS=$JOBS ==="
  bash "experiments/$folder/run.sh" "$name"
  rc=$?
  case "$kind" in
    control)
      [[ "$rc" == 0 ]] || { halt "$folder $name: контроль вне допуска или ошибка (код $rc)"; return 1; } ;;
    train|search)
      if [[ "$rc" != 0 && "$rc" != 1 ]]; then halt "$folder $name: код $rc"; return 1; fi
      local flag=""
      [[ "$kind" == search ]] && flag="--search"
      gate stage --exp "${folder:0:4}" --stage "$name" $flag || return 1 ;;
    critical)
      [[ "$rc" == 0 ]] || { halt "$folder $name: код $rc"; return 1; } ;;
    other)
      [[ "$rc" == 0 ]] || echo "ОШИБКА: $folder $name (код $rc), очередь продолжается" ;;
  esac
  return 0
}

experiment() {   # experiment <folder> <stage:kind>...
  local folder="$1" item
  shift
  for item in "$@"; do
    stage "$folder" "${item%%:*}" "${item##*:}" || { echo "ОШИБКА: $folder прерван на ${item%%:*}"; return 1; }
  done
}

start_conv() {
  if conv_running || [[ -f results/conv_started ]]; then return 0; fi
  local session="q_conv_$(date +%m%d_%H%M)"
  tmux new-session -d -s "$session" "cd $ROOT_DIR && bash scripts/overnight_e024_e033.sh conv 2>&1 | tee -a logs/remote/$session.log" \
    && echo "$session" >> .remote_sessions && date -Is > results/conv_started && echo "запущена сессия $session (E032)"
}

wait_sessions() {   # wait_sessions <regex> <what>
  while tmux ls -F '#{session_name}' 2>/dev/null | grep -qE "$1"; do
    echo "$(date -Is) жду окончания $2"
    sleep 600
  done
}

run_p0() {
  experiment E024_data_audit control:control main:critical || return 1
  experiment E031_perturbation_triplets main:other
}
run_p1() {
  experiment E025_family_tuning control:control s1:search check-code:control s2:search s3:train summarize:critical || return 1
  start_conv
}
run_p2() {
  experiment E026_kerr_complexity_map control:control grid:search confirm:train summarize:other || return 1
  experiment E028_dbp_depth_layerwise control:control a:train shared:train hybrid:train b:train a_long:train summarize:other
}
run_p3() {
  experiment E027_activation_ablation control:control s1:search s1_fixed:search s2:search s2_fixed:train s3:train \
    s3_fixed:train summarize:other || return 1
  experiment E029_frontend_vs_width control:control main:train summarize:other
}
run_p4() { experiment E030_loss_functions control:control s1:search s2:train summarize:other; }
run_p5() {
  wait_sessions '^q_conv_[0-9]{4}_[0-9]{4}$' "E032 (сессия q_conv)"
  experiment E033_pareto_final control:control fill:train summarize:other
}
run_conv() { experiment E032_convergence_1m control:control main:train summarize:other; }

{
  echo "=== $(date -Is) queue $part: THREADS=$THREADS DEVICE=$DEVICE host=$(hostname) commit=$(git rev-parse --short HEAD) ==="
  if [[ ! -f resources.json ]]; then
    bash experiments/E024_data_audit/run.sh probe || echo "ОШИБКА: probe"
  fi
  [[ "$part" == "all" ]] && wait_sessions '^e020_' "сессий E020-E023"
  case "$part" in
    p0) run_p0 ;; p1) run_p1 ;; p2) run_p2 ;; p3) run_p3 ;; p4) run_p4 ;; p5) run_p5 ;; conv) run_conv ;;
    all) run_p0 && run_p1 && run_p2 && run_p3 && run_p4 && run_p5 ;;
  esac
  echo "=== $(date -Is) queue $part завершена; остановка: $(gate halted >/dev/null && echo нет || echo ДА, см. results/gate_status.json);" \
       "строк ОШИБКА: $(grep -c '^ОШИБКА' "$LOG" 2>/dev/null || echo 0) ==="
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
