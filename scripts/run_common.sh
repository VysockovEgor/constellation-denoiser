# Shared helpers for experiments/E01x_*/run.sh. Source it, do not execute it.
#
# Expects EXP (e.g. E010) and EXP_DIR to be set by the caller. Provides:
#   rc_setup            cd to the repository root, choose python, export defaults
#   rc_run NAME CMD...  run CMD under nice, tee to the stage log, keep the exit code
#   rc_lock NAME        PID file for one stage; stale files are replaced, live ones refuse
#   rc_runner STAGE     train one stage through constellation_denoiser.runner
#   rc_is_common STAGE  true for probe / plan / list / check-control
#   rc_common STAGE     handle one of those stages (call outside `if` so `set -e` applies)

rc_setup() {
  ROOT_DIR="$(cd "$EXP_DIR/../.." && pwd)"
  cd "$ROOT_DIR"
  if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then PYTHON="$ROOT_DIR/.venv/bin/python"; else PYTHON="python3"; fi
  fi
  export PYTHON
  export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
  export JOBS="${JOBS:-auto}" THREADS="${THREADS:-auto}" DEVICE="${DEVICE:-auto}"
  export SEEDS="${SEEDS:-}" SMOKE="${SMOKE:-0}" FORCE="${FORCE:-0}" SYNTHETIC="${SYNTHETIC:-0}"
  export PYTHONUNBUFFERED=1
  if [[ "$SMOKE" == "1" ]]; then
    LOG_DIR="$EXP_DIR/results_smoke/logs"
  else
    LOG_DIR="$EXP_DIR/logs"
  fi
  mkdir -p "$LOG_DIR"
  RC_PIDS=()
  trap 'rc_cleanup' EXIT
  trap 'echo "прервано (Ctrl+C); длинные прогоны продолжатся с чекпоинта при повторном запуске" >&2; exit 130' INT TERM
}

rc_cleanup() {
  local pidfile
  for pidfile in "${RC_PIDS[@]}"; do
    rm -f "$pidfile"
  done
}

rc_lock() {
  local pidfile="$LOG_DIR/$1.pid"
  if [[ -f "$pidfile" ]]; then
    local old
    old="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null && [[ "$old" != "$$" ]]; then
      echo "стадия $1 уже запущена (PID $old, файл $pidfile); второй запуск отменён" >&2
      exit 5
    fi
    echo "удаляю устаревший PID-файл $pidfile (процесс ${old:-?} не существует)" >&2
  fi
  echo "$$" > "$pidfile"
  RC_PIDS+=("$pidfile")
}

rc_run() {
  local name="$1"
  shift
  local logfile="$LOG_DIR/$name.log"
  {
    echo "=== $(date -Is) $EXP $name: $* ==="
    echo "JOBS=$JOBS THREADS=$THREADS DEVICE=$DEVICE SEEDS=${SEEDS:-default} SMOKE=$SMOKE FORCE=$FORCE"
  } | tee -a "$logfile"
  set +e
  nice -n 10 "$@" 2>&1 | tee -a "$logfile"
  local status="${PIPESTATUS[0]}"
  set -e
  echo "=== $(date -Is) $EXP $name exit=$status ===" | tee -a "$logfile"
  return "$status"
}

rc_runner() {
  local stage="$1"
  shift
  rc_lock "$stage"
  rc_run "$stage" "$PYTHON" -u -m constellation_denoiser.runner --exp "$EXP" --stage "$stage" "$@"
}

rc_common() {
  case "$1" in
    probe)
      rc_lock probe
      rc_run probe "$PYTHON" -u scripts/probe_resources.py
      ;;
    plan)
      local stage="${2:-}"
      if [[ -z "$stage" ]]; then
        echo "использование: run.sh plan <стадия>|all" >&2
        exit 2
      fi
      if [[ "$stage" == "all" ]]; then
        local s
        for s in "${TRAIN_STAGES[@]}"; do
          "$PYTHON" -m constellation_denoiser.runner --exp "$EXP" --stage "$s" --plan || true
          echo
        done
      else
        "$PYTHON" -m constellation_denoiser.runner --exp "$EXP" --stage "$stage" --plan
      fi
      ;;
    list)
      "$PYTHON" -m constellation_denoiser.runner --exp "$EXP" --stage "${2:?стадия}" --list
      ;;
    check-control)
      rc_run check-control "$PYTHON" -m constellation_denoiser.runner --exp "$EXP" --stage control --check-control
      ;;
  esac
}

rc_is_common() {
  case "$1" in
    probe|plan|list|check-control) return 0 ;;
    *) return 1 ;;
  esac
}
