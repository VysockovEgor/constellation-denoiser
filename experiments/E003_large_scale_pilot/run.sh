#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Historical command: PYTHONPATH=src python -m constellation_denoiser.large_scale_pilot --sample-files 300 --seeds 17 29 43 --epochs 40 --patience 10 --batch-size 1024 --threads 6"
  exit 0
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments/E003_large_scale_pilot"
for path in "$EXP_DIR/results" "$EXP_DIR/figures" "$EXP_DIR/logs" "$ROOT_DIR/checkpoints/E003"; do
  if [[ -d "$path" ]] && find "$path" -type f -print -quit | grep -q .; then
    echo "E003 has archived outputs at $path; refusing to overwrite them." >&2
    exit 2
  fi
done
cd "$ROOT_DIR"
exec env PYTHONPATH=src python -m constellation_denoiser.large_scale_pilot --sample-files 300 --seeds 17 29 43 --epochs 40 --patience 10 --batch-size 1024 --threads 6
