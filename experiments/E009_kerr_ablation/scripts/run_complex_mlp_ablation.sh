#!/usr/bin/env bash
# Legacy wrapper: use scripts/launch_ablation_max.sh for full CPU utilization.
set -u
if [[ "${1:-}" == "--help" ]]; then
  echo "Run the E009 launch_ablation_max.sh workflow."
  exit 0
fi
exec "$(dirname "$0")/launch_ablation_max.sh"
