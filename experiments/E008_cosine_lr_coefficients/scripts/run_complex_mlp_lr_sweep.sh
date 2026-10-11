#!/usr/bin/env bash
set -u
if [[ "${1:-}" == "--help" ]]; then
  echo "Run the E008 cosine-learning-rate complex MLP sweep (20,000 steps, seeds 17 and 29)."
  exit 0
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p experiments/E008_cosine_lr_coefficients/logs
for config_index in 0 1 2; do
  PYTHONPATH=src python3 -u -m constellation_denoiser.complex_mlp_lr_sweep \
    --seed 17 --config-index "$config_index" --threads 6 --resume \
    >> experiments/E008_cosine_lr_coefficients/logs/complex_mlp_lr_seed17.log 2>&1 &
  pid17=$!
  PYTHONPATH=src python3 -u -m constellation_denoiser.complex_mlp_lr_sweep \
    --seed 29 --config-index "$config_index" --threads 6 --resume \
    >> experiments/E008_cosine_lr_coefficients/logs/complex_mlp_lr_seed29.log 2>&1 &
  pid29=$!
  wait "$pid17"
  status17=$?
  wait "$pid29"
  status29=$?
  if (( status17 != 0 || status29 != 0 )); then
    echo "Failed cosine configuration index $config_index (seed17=$status17, seed29=$status29)" >&2
    exit 1
  fi
done
