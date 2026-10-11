#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Historical command: unknown; see config.json and README.md."
  echo "Archived run: outputs are protected; create a new experiment directory to rerun."
  exit 0
fi
echo "E005 historical command and seed are not recorded; refusing an unverifiable rerun." >&2
exit 2
