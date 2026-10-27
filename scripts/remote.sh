#!/usr/bin/env bash
# Two-command work cycle with the shared server (run from WSL, repository root).
#
#   bash scripts/remote.sh go part1 | part2 | all | e020 <part> | <E0NN> <stage>   push, update the server, start in tmux
#                              (part1/part2/all: chain E015-E019; e020 part1|part2|part3|all: chain E020-E023)
#   bash scripts/remote.sh estimate part1 | part2 | <E0NN> [stage]     hours from the server-side estimate stages
#   bash scripts/remote.sh sync                                        update the server checkout only (no job)
#   bash scripts/remote.sh status                                      sessions, processes, load, runs, logs, ОШИБКА
#   bash scripts/remote.sh fetch [--to <folder>] [--journal]           bring results back (+ make journal), commit, push
#   bash scripts/remote.sh stop <session>                              stop a session started by `go` (only those)
#   add --dry-run anywhere to print the commands instead of running them
#
# Test only: REMOTE_LOCAL_TEST=1 runs the "server" commands with local bash in REMOTE_DIR (a local clone).
# Config: scripts/remote.env (not in Git; see scripts/remote.env.example): SERVER, REMOTE_DIR, BRANCH.
# Safety: works only inside $REMOTE_DIR and with its own archives in ~ on the server; no sudo; kills only
# tmux sessions it created itself (recorded in $REMOTE_DIR/.remote_sessions); never kills processes by pattern.
set +H
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN=0
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then DRY_RUN=1; else ARGS+=("$arg"); fi
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

die() { echo "remote.sh: $*" >&2; exit 1; }

ENV_FILE="$ROOT_DIR/scripts/remote.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
elif [[ "$DRY_RUN" == "1" ]]; then
  SERVER="${SERVER:-user@server.example}"
  REMOTE_DIR="${REMOTE_DIR:-~/constellation-denoiser}"
else
  die "нет scripts/remote.env: скопируйте scripts/remote.env.example и заполните SERVER, REMOTE_DIR"
fi
: "${SERVER:?SERVER не задан в scripts/remote.env}"
: "${REMOTE_DIR:?REMOTE_DIR не задан в scripts/remote.env}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
[[ "$BRANCH" =~ ^[A-Za-z0-9._/-]+$ ]] || die "недопустимое имя ветки: $BRANCH"

REGISTRY=".remote_sessions"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-%C" -o ControlPersist=10m -o BatchMode=yes
          -o ServerAliveInterval=30)
SESSION_RE='^(part1|part2|all|e020_(part1|part2|part3|all)|E0[0-9][0-9])_[0-9]{4}_[0-9]{4}$'

# Server update = fast-forward that never loses server data (expects $BRANCH set before it):
#  stop if tracked files changed or the server branch has its own commits; untracked files the update
#  would overwrite must be byte-identical to origin (usually they are: `fetch` committed them from there).
SERVER_UPDATE=$(cat <<'UPDATE'
git fetch -q origin || exit 1
if ! git diff --quiet HEAD; then
  echo 'на сервере изменены отслеживаемые файлы (новые результаты?): сначала remote.sh fetch' >&2
  git diff --stat HEAD | tail -5 >&2; exit 1
fi
if git rev-parse -q --verify "$BRANCH" >/dev/null && [ "$(git rev-list --count "origin/$BRANCH..$BRANCH")" != 0 ]; then
  echo 'на сервере есть коммиты, которых нет в origin' >&2; exit 1
fi
bad=0
for f in $(git diff --name-only --diff-filter=A HEAD "origin/$BRANCH"); do
  if [ -e "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 && ! git show "origin/$BRANCH:$f" | cmp -s - "$f"; then
    echo "неотслеживаемый файл отличается от origin: $f" >&2; bad=1
  fi
done
[ "$bad" = 0 ] || { echo 'сначала remote.sh fetch' >&2; exit 1; }
git checkout -q -f -B "$BRANCH" "origin/$BRANCH" && git diff --quiet HEAD && git log --oneline -1
UPDATE
)

# remote "<command>": runs inside $REMOTE_DIR on the server (dry run: prints it)
remote() {
  local command="cd $REMOTE_DIR && $1"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] ssh %s "%s"\n' "$SERVER" "$command"
  elif [[ "${REMOTE_LOCAL_TEST:-0}" == "1" ]]; then   # test only: the "server" is a local clone
    bash -c "$command"
  else
    ssh "${SSH_OPTS[@]}" "$SERVER" "$command"
  fi
}

local_do() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

# "<part>" or "<E0NN> <stage>" -> server command and session stem
resolve_job() {
  case "${1:-}" in
    part1|part2|all)
      [[ $# -eq 1 ]] || die "после $1 аргументы не нужны"
      JOB_STEM="$1"
      JOB_CMD="bash scripts/overnight_e015_e018.sh $1"
      ;;
    e020)
      [[ $# -eq 2 && "$2" =~ ^(part1|part2|part3|all)$ ]] || die "использование: go e020 part1|part2|part3|all"
      JOB_STEM="e020_$2"
      JOB_CMD="bash scripts/overnight_e020_e023.sh $2"
      ;;
    E0[0-9][0-9])
      [[ $# -eq 2 ]] || die "использование: go <E0NN> <стадия>"
      [[ "$2" =~ ^[a-z0-9-]+$ ]] || die "недопустимая стадия: $2"
      local matches=(experiments/"$1"_*/run.sh)
      [[ -f "${matches[0]}" ]] || die "нет experiments/$1_*/run.sh"
      JOB_STEM="$1"
      JOB_CMD="bash ${matches[0]} $2"
      ;;
    *) die "ожидается part1, part2, all или <E0NN> <стадия>" ;;
  esac
}

env_prefix() {   # pass JOBS/THREADS/DEVICE/SEEDS/SHARE through when set locally
  local out="" name
  for name in JOBS THREADS DEVICE SEEDS SHARE POLITE_MAX_WAIT_S; do
    if [[ -n "${!name:-}" ]]; then
      [[ "${!name}" =~ ^[A-Za-z0-9.,_-]+$ ]] || die "недопустимое значение $name"
      out+="$name=${!name} "
    fi
  done
  printf '%s' "$out"
}

cmd_go() {
  resolve_job "$@"
  git diff --quiet && git diff --cached --quiet || die "есть незакоммиченные изменения в отслеживаемых файлах: сначала commit"
  local untracked
  untracked="$(git ls-files --others --exclude-standard | wc -l)"
  [[ "$untracked" == "0" ]] || echo "внимание: $untracked неотслеживаемых файлов локально (на сервер не попадут)"
  local_do git push -u origin "$BRANCH"
  remote "BRANCH=$BRANCH; $SERVER_UPDATE"
  local session="${JOB_STEM}_$(date +%m%d_%H%M)"
  local log="logs/remote/$session.log"
  local inner
  inner="cd $REMOTE_DIR && $(env_prefix)$JOB_CMD 2>&1 | tee -a $log"
  remote "if tmux ls -F '#{session_name}' 2>/dev/null | grep -qE '^${JOB_STEM}_[0-9]{4}_[0-9]{4}\$'; then \
echo 'работа ${JOB_STEM} уже идёт:' >&2; tmux ls -F '#{session_name}' | grep -E '^${JOB_STEM}_' >&2; exit 3; fi; \
mkdir -p logs/remote && tmux new-session -d -s $session $(printf '%q' "$inner") && echo $session >> $REGISTRY"
  echo "сессия: $session"
  echo "лог на сервере: $REMOTE_DIR/$log"
  echo "подключиться: ssh $SERVER -t tmux attach -t $session"
}

cmd_sync() {
  git diff --quiet && git diff --cached --quiet || die "есть незакоммиченные изменения в отслеживаемых файлах: сначала commit"
  local_do git push -u origin "$BRANCH"
  remote "BRANCH=$BRANCH; $SERVER_UPDATE"
  echo "сервер обновлён до origin/$BRANCH"
}

cmd_estimate() {
  local targets=()
  case "${1:-}" in
    part1) targets=("E016_phys_kerr_only all" "E017_phys_plus_mlp all") ;;
    part2) targets=("E018_dbp_depth all" "E019_confirm_long_runs all") ;;
    e020)
      case "${2:-}" in
        part1) targets=("E020_e017_anomaly all") ;;
        part2) targets=("E021_pareto all" "E022_baselines all") ;;
        part3) targets=("E023_split_robustness all") ;;
        all) targets=("E020_e017_anomaly all" "E021_pareto all" "E022_baselines all" "E023_split_robustness all") ;;
        *) die "использование: estimate e020 part1|part2|part3|all" ;;
      esac
      ;;
    E0[0-9][0-9])
      local stage="${2:-all}"
      [[ "$stage" =~ ^[a-z0-9-]+$ ]] || die "недопустимая стадия: $stage"
      local matches=(experiments/"$1"_*/run.sh)
      [[ -f "${matches[0]}" ]] || die "нет experiments/$1_*/run.sh"
      targets=("$(basename "$(dirname "${matches[0]}")") $stage")
      ;;
    *) die "ожидается part1, part2, e020 <часть> или <E0NN> [стадия]" ;;
  esac
  local script="" target
  for target in "${targets[@]}"; do
    local folder="${target%% *}" stage="${target#* }"
    local flag=""
    [[ "$folder" == E019_* ]] && flag="CONFIRM_ESTIMATE_FROM_SMOKE=\$( [ -f experiments/E018_dbp_depth/results/dbp_depth.csv ] && echo 0 || echo 1) "
    script+="$(env_prefix)${flag}bash experiments/$folder/run.sh estimate $stage; "
  done
  if [[ "$DRY_RUN" == "1" ]]; then
    remote "$script"
    return
  fi
  remote "$script" 2>&1 | tee /dev/stderr | awk '/оценка: ~/ { for (i = 1; i <= NF; i++) if ($i ~ /^~[0-9.]+$/) { sub("~", "", $i); s += $i; break } }
    END { printf "\nсуммарно при рекомендованных JOBS: ~%.1f ч (сумма стадий; стадии идут последовательно)\n", s }'
}

cmd_status() {
  remote "echo '== tmux'; tmux ls 2>/dev/null || echo 'нет сессий'; \
echo '== процессы проекта'; pgrep -fa '[c]onstellation_denoiser' || echo 'нет'; \
echo '== загрузка'; uptime; \
echo '== готовые прогоны'; for d in experiments/E*/results/runs; do [ -d \"\$d\" ] && printf '%6s  %s\n' \"\$(ls \"\$d\" | grep -c '\\.json\$')\" \"\$d\"; done; \
echo '== свежие логи (24 ч, последние 5 строк)'; \
for f in \$(find logs experiments -name '*.log' -mmin -1440 2>/dev/null | xargs -r ls -t | head -12); do echo \"-- \$f\"; tail -5 \"\$f\"; done; \
echo '== прогресс (последний итог по каждой стадии за 24 ч: готово / всего, ETA стадии)'; \
for f in \$(find experiments -path '*/logs/*.log' -mmin -1440 2>/dev/null | sort); do \
  run=\$(grep -E 'running [0-9]+ tasks' \"\$f\" | tail -1 | grep -oE 'running [0-9]+'); \
  ok=\$(grep -E '\\[ok [0-9]+/[0-9]+\\]' \"\$f\" | tail -1 | grep -oE '\\[ok [0-9]+/[0-9]+\\]|ETA ~[0-9]+ min'); \
  fin=\$(grep -E 'finished:|nothing to run' \"\$f\" | tail -1 | cut -c1-60); \
  [ -n \"\$run\$ok\" ] && echo \"\$f: \$run | \$(echo \$ok) | \$fin\"; done; \
echo '== текущий шаг цепочки'; ls -t logs/remote/*.log 2>/dev/null | head -1 | xargs -r grep -E '^=== .*: (control|main|summarize|diagnostics)' | tail -2; \
echo '== ОШИБКА'; grep -rn 'ОШИБКА' logs experiments/*/logs 2>/dev/null | tail -20 || true"
}

cmd_fetch() {
  local to="" journal=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --to) to="${2:?--to <папка>}"; shift 2 ;;
      --journal) journal=1; shift ;;
      *) die "использование: fetch [--to <папка>] [--journal]" ;;
    esac
  done
  local stamp archive local_dir
  stamp="$(date +%Y%m%d_%H%M%S)"
  archive="cdn_results_$stamp.tgz"
  local_dir="$ROOT_DIR/.remote_fetch"
  remote "tar czf ~/$archive --exclude='*/runs' --exclude='*/runs/*' --exclude='*.pt' --exclude='*.pid' \
--exclude='*/results_smoke*' --exclude='checkpoints' \$(ls -d experiments/E*/results experiments/E*/figures experiments/E*/logs results logs 2>/dev/null)"
  local_do mkdir -p "$local_dir"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] scp %s:~/%s %s/\n' "$SERVER" "$archive" "$local_dir"
  elif [[ "${REMOTE_LOCAL_TEST:-0}" == "1" ]]; then
    cp ~/"$archive" "$local_dir/"
  else
    scp "${SSH_OPTS[@]}" "$SERVER:~/$archive" "$local_dir/"
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] ssh %s "%s"\n' "$SERVER" "rm -f ~/$archive"
  elif [[ "${REMOTE_LOCAL_TEST:-0}" == "1" ]]; then
    rm -f ~/"$archive"
  else
    ssh "${SSH_OPTS[@]}" "$SERVER" "rm -f ~/$archive"      # only our own archive
  fi
  local_do tar xzf "$local_dir/$archive" -C "$ROOT_DIR"
  if [[ "$journal" == "1" ]]; then          # numbers of the journal pages come from the fetched files
    local_do make journal
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] git status; git add <файлы из архива ≤ 5 МБ, кроме игнорируемых>; git commit -m 'results: $(date +%F)'; git push"
  else
    git status --short
    local files=()
    while IFS= read -r path; do
      [[ -f "$path" ]] || continue
      [[ "$(stat -c%s "$path")" -le 5000000 ]] || { echo "пропускаю > 5 МБ: $path"; continue; }
      git check-ignore -q "$path" && continue
      files+=("$path")
    done < <(tar tzf "$local_dir/$archive" | grep -v '/$'; [[ "$journal" == "1" ]] && ls journal/_auto/*.md)
    if [[ ${#files[@]} -gt 0 ]]; then
      git add -- "${files[@]}"
    fi
    if git diff --cached --quiet; then
      echo "коммитить нечего: результаты совпадают с репозиторием"
    else
      git commit -m "results: $(date +%F)"
      git push -u origin "$(git rev-parse --abbrev-ref HEAD)"
    fi
  fi
  if [[ -n "$to" ]]; then
    local_do mkdir -p "$to"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[dry-run] cp <*.png и *_summary.csv, *_verdicts.csv, results/*.csv из архива> $to/"
    else
      tar tzf "$local_dir/$archive" | grep -E '\.png$|_summary\.csv$|_verdicts\.csv$|^results/[^/]+\.csv$' \
        | while IFS= read -r path; do [[ -f "$path" ]] && cp "$path" "$to/"; done
      echo "графики и сводки скопированы в $to"
    fi
  fi
}

cmd_stop() {
  local session="${1:-}"
  [[ "$session" =~ $SESSION_RE ]] || die "имя сессии не из remote.sh go: $session"
  remote "grep -qx '$session' $REGISTRY 2>/dev/null || { echo 'сессия $session не создавалась remote.sh, не трогаю' >&2; exit 4; }; \
tmux kill-session -t '$session' && grep -vx '$session' $REGISTRY > $REGISTRY.tmp; mv $REGISTRY.tmp $REGISTRY; echo 'остановлена: $session'"
}

case "${1:-}" in
  go) shift; cmd_go "$@" ;;
  sync) shift; cmd_sync ;;
  estimate) shift; cmd_estimate "$@" ;;
  status) shift; cmd_status ;;
  fetch) shift; cmd_fetch "$@" ;;
  stop) shift; cmd_stop "$@" ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
