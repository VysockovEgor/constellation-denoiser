#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Historical command: bash experiments/E009_kerr_ablation/scripts/launch_ablation_max.sh"
  exit 0
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/E009_kerr_ablation"
for path in "$EXP_DIR/results" "$EXP_DIR/figures" "$EXP_DIR/logs" "$ROOT_DIR/checkpoints/E009"; do
  if [[ -d "$path" ]] && find "$path" -type f -print -quit | grep -q .; then
    echo "E009 has archived outputs at $path; refusing to overwrite them." >&2
    exit 2
  fi
done
cd "$ROOT_DIR"
exec bash "$EXP_DIR/scripts/launch_ablation_max.sh"
