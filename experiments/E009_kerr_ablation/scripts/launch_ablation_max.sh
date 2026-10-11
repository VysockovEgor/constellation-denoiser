#!/usr/bin/env bash
set -uo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Run all E009 Kerr-ablation variants (a-d) for seeds 17 and 29."
  exit 0
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p experiments/E009_kerr_ablation/logs

PYTHON="${PYTHON:-python3}"

CORES="$(nproc)"
# Eight parallel runs (4 variants × 2 seeds); split CPU threads evenly.
JOBS=8
THREADS=$(( CORES / JOBS ))
if (( THREADS < 1 )); then
  THREADS=1
fi

{
  echo "=== launch $(date -Is) ==="
  echo "cores=$CORES jobs=$JOBS threads_per_job=$THREADS python=$PYTHON"
  "$PYTHON" -c "import torch; print('torch', torch.__version__)" || exit 1
} >> experiments/E009_kerr_ablation/logs/ablation_launch.log 2>&1

export PYTHONPATH=src
variants=(a b c d)
seeds=(17 29)
pids=()
names=()

for variant in "${variants[@]}"; do
  for seed in "${seeds[@]}"; do
    log="experiments/E009_kerr_ablation/logs/complex_mlp_ablation_${variant}_seed${seed}.log"
    echo "[start] variant=$variant seed=$seed threads=$THREADS -> $log" >> experiments/E009_kerr_ablation/logs/ablation_launch.log
    "$PYTHON" -u -m constellation_denoiser.complex_mlp_ablation_train \
      --variant "$variant" --seed "$seed" --threads "$THREADS" --resume \
      >> "$log" 2>&1 &
    pids+=("$!")
    names+=("${variant}_s${seed}")
  done
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[fail] ${names[$i]} pid=${pids[$i]}" >> experiments/E009_kerr_ablation/logs/ablation_launch.log
    fail=1
  else
    echo "[ok] ${names[$i]}" >> experiments/E009_kerr_ablation/logs/ablation_launch.log
  fi
done

if (( fail == 0 )); then
  echo "[summarize] $(date -Is)" >> experiments/E009_kerr_ablation/logs/ablation_launch.log
  "$PYTHON" -u -m constellation_denoiser.summarize_complex_mlp_ablation >> experiments/E009_kerr_ablation/logs/ablation_summarize.log 2>&1 \
    || fail=1
fi

echo "=== done $(date -Is) exit=$fail ===" >> experiments/E009_kerr_ablation/logs/ablation_launch.log
exit "$fail"
