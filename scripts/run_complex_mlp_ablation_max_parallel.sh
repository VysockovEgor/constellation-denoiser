#!/usr/bin/env bash
set -uo pipefail
cd /home/user/projects/constellation-denoiser
mkdir -p reports/logs models/complex_mlp_ablation

PYTHON="${PYTHON:-/home/user/.virtualenvs/constellation-denoiser/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

NCPU="$(nproc)"
JOBS=8
THREADS=$(( NCPU / JOBS ))
(( THREADS < 1 )) && THREADS=1

{
  echo "=== ablation launch $(date -Is) ==="
  echo "python=$PYTHON"
  "$PYTHON" -c "import torch; print('torch', torch.__version__)" || exit 1
  echo "nproc=$NCPU jobs=$JOBS threads_per_job=$THREADS"
} >> reports/logs/complex_mlp_ablation_launch.log 2>&1

export PYTHONPATH=src
PIDS=()
for variant in a b c d; do
  for seed in 17 29; do
    log="reports/logs/complex_mlp_ablation_${variant}_seed${seed}.log"
    "$PYTHON" -u -m constellation_denoiser.complex_mlp_ablation_train \
      --variant "$variant" --seed "$seed" --threads "$THREADS" --resume \
      >> "$log" 2>&1 &
    PIDS+=($!)
    echo "started variant=$variant seed=$seed pid=$! threads=$THREADS" >> reports/logs/complex_mlp_ablation_launch.log
  done
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
    echo "job failed pid=$pid" >> reports/logs/complex_mlp_ablation_launch.log
  fi
done

if (( FAIL == 0 )); then
  "$PYTHON" -u -m constellation_denoiser.summarize_complex_mlp_ablation \
    >> reports/logs/complex_mlp_ablation_summarize.log 2>&1
  echo "=== summarize done $(date -Is) ===" >> reports/logs/complex_mlp_ablation_launch.log
else
  echo "=== training failed, skip summarize $(date -Is) ===" >> reports/logs/complex_mlp_ablation_launch.log
  exit 1
fi
