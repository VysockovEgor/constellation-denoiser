#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Historical command: bash experiments/E007_window_width_sweep/scripts/run_complex_mlp_sweep.sh"
  exit 0
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/E007_window_width_sweep"
for path in "$EXP_DIR/results" "$EXP_DIR/figures" "$EXP_DIR/logs" "$ROOT_DIR/checkpoints/E007"; do
  if [[ -d "$path" ]] && find "$path" -type f -print -quit | grep -q .; then
    echo "E007 has archived outputs at $path; refusing to overwrite them." >&2
    exit 2
  fi
done
cd "$ROOT_DIR"
exec bash "$EXP_DIR/scripts/run_complex_mlp_sweep.sh"
