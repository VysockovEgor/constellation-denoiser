"""Detect server resources, benchmark (JOBS, THREADS) and write resources.json.

Run from the repository root on the server:

    PYTHONPATH=src python scripts/probe_resources.py

Overrides: SHARE (default 0.5 of free cores), MAX_CORES, MAX_MEM_GB, MAX_JOBS,
PROBE_SECONDS (benchmark time budget, default 150).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from constellation_denoiser.resources import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
