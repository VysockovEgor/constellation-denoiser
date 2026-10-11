#!/usr/bin/env bash
set -u
cd /home/user/projects/constellation-denoiser
mkdir -p reports/logs
for config_index in 0 1 2 3 4 5; do
  PYTHONPATH=src python3 -u -m constellation_denoiser.complex_mlp_sweep \
    --seed 17 --config-index "$config_index" --threads 6 --resume \
    >> reports/logs/complex_mlp_sweep_seed17.log 2>&1 &
  pid17=$!
  PYTHONPATH=src python3 -u -m constellation_denoiser.complex_mlp_sweep \
    --seed 29 --config-index "$config_index" --threads 6 --resume \
    >> reports/logs/complex_mlp_sweep_seed29.log 2>&1 &
  pid29=$!
  wait "$pid17"
  status17=$?
  wait "$pid29"
  status29=$?
  if (( status17 != 0 || status29 != 0 )); then
    echo "Failed configuration index $config_index (seed17=$status17, seed29=$status29)" >&2
    exit 1
  fi
done
