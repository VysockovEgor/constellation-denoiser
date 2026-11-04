"""Points of the one-step closed-form triplets (E031, E034, E036) with three RM conventions (E040, E043)."""

from __future__ import annotations

import pandas as pd

from .rm_group_convention import rm_arxiv_2409_20023, rm_buffered, rm_buffered_bound
from .runner import exp_dir
from .triplets import triplet_terms


def one_step_points() -> pd.DataFrame:
    rows = []
    path = exp_dir("E031") / "results" / "triplet_summary.csv"
    if path.is_file():
        for _, r in pd.read_csv(path).iterrows():
            if not r.arm.startswith("T3"):
                continue
            fir = int(r.arm.replace("T3+FIR", "")) if "FIR" in r.arm else 0
            terms = triplet_terms(int(r.M), "T3")
            rows.append({"source": "E031", "label": f"{r.arm} M{int(r.M)}", "M": int(r.M), "rm_repo": r.rm_sym,
                         "rm_buffered_my_estimate": rm_buffered(terms, fir), "rm_arxiv_2409_20023": rm_arxiv_2409_20023(int(r.M), fir),
                         "val_mean": r.val_mean, "test_mean": r.test_mean, "test_min": r.test_min, "test_max": r.test_max,
                         "q2_mean": r.q2_mean, "q2_min": r.q2_mean, "q2_max": r.q2_mean, "n_seeds": r.n_replicas})
    for exp in ("E034", "E036"):
        path = exp_dir(exp) / "results" / "triplet_ext_summary.csv"
        if not path.is_file():
            continue
        for _, r in pd.read_csv(path).iterrows():
            M = int(r.variant.split("_M")[1].split("_")[0])
            full = r.variant.endswith("_full")
            rb = rm_buffered(triplet_terms(M, "T3"), 7) if full else rm_buffered_bound(int(r.K), M, 7)
            rows.append({"source": exp, "label": r.variant, "M": M, "rm_repo": r.rm, "rm_buffered_my_estimate": rb,
                         "rm_arxiv_2409_20023": rm_arxiv_2409_20023(M, 7), "val_mean": r.val_mean, "test_mean": r.test_mean,
                         "test_min": r.test_min, "test_max": r.test_max, "q2_mean": r.q2_mean, "q2_min": r.q2_min,
                         "q2_max": r.q2_max, "n_seeds": r.n_replicas})
    return pd.DataFrame(rows)
