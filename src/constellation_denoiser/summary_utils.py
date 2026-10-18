"""Shared group comparison for E012-E014 decision rules.

Seed spread = range (max - min) over seeds. A group "beats" another when the
difference of means exceeds the larger of the two ranges and is at least
``min_pp`` percentage points.
"""

from __future__ import annotations

import numpy as np


def spread(values) -> float:
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(max(values) - min(values)) if len(values) > 1 else float("nan")


def compare(a, b, min_pp: float = 0.5) -> dict:
    a = [v for v in a if v is not None and np.isfinite(v)]
    b = [v for v in b if v is not None and np.isfinite(v)]
    if not a or not b:
        return {"diff": float("nan"), "spread": float("nan"), "beats": False, "n_a": len(a), "n_b": len(b)}
    diff = float(np.mean(a) - np.mean(b))
    ref = max(spread(a), spread(b))
    ref_ok = np.isfinite(ref)
    return {"diff": diff, "spread": ref, "beats": bool(ref_ok and diff > ref and diff >= min_pp),
            "within_spread": bool(ref_ok and abs(diff) < ref), "n_a": len(a), "n_b": len(b)}
