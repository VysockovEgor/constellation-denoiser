#!/usr/bin/env bash
# Legacy wrapper: use scripts/launch_ablation_max.sh for full CPU utilization.
set -u
exec "$(dirname "$0")/launch_ablation_max.sh"
