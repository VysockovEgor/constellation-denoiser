"""Add Q2 (dB) columns to the E010-E014 per-run result tables; key points table.

    PYTHONPATH=src python scripts/recompute_q2.py

Reads experiments/E010...E014/results/runs/*.json when they exist; the server
results in Git contain only the merged per-run CSVs (lr_sweep.csv,
long_runs.csv, depth_window.csv, transformer_runs.csv, kan_hybrid.csv),
which carry the same fields (raw_ber, test_ber, test_reduction_valbest_select).
Writes <name>_q2.csv next to each table (originals untouched) and
results/q2_key_points.csv. Runs without a raw BER are skipped and listed;
nothing is reconstructed from a percentage without the raw BER.

Columns: q2_db_raw (raw test BER), q2_db_model / q2_gain_db (test BER at the
final step), valbest_q2_db_model / valbest_q2_gain_db (test at the
validation-best evaluation: BER = raw_ber * (1 - test_reduction_valbest_select/100),
the exact inverse of how that reduction was computed from the same raw BER).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from constellation_denoiser.q2 import q2_db  # noqa: E402

TABLES = {
    "E010": "E010_lr_schedule_study/results/lr_sweep.csv",
    "E011": "E011_long_runs_convergence/results/long_runs.csv",
    "E012": "E012_depth_vs_window/results/depth_window.csv",
    "E013": "E013_transformer_scaling/results/transformer_runs.csv",
    "E014": "E014_kerr_kan_hybrid/results/kan_hybrid.csv",
}
METRIC = "test_reduction_valbest_select"


def load(exp: str, relative: str) -> tuple[pd.DataFrame, str]:
    path = ROOT / "experiments" / relative
    runs_dir = path.parent / "runs"
    if runs_dir.is_dir() and any(runs_dir.glob("*.json")):
        from constellation_denoiser.merge_runs import load_runs

        return load_runs(path.parent), str(runs_dir)
    return pd.read_csv(path), str(path)


def add_q2(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["q2_db_raw"] = frame.raw_ber.map(q2_db)
    frame["q2_db_model"] = frame.test_ber.map(q2_db)
    frame["q2_gain_db"] = frame.q2_db_model - frame.q2_db_raw
    valbest_ber = frame.raw_ber * (1 - frame[METRIC] / 100)
    frame["valbest_q2_db_model"] = valbest_ber.map(q2_db)
    frame["valbest_q2_gain_db"] = frame.valbest_q2_db_model - frame.q2_db_raw
    return frame


def key_row(source: str, label: str, part: pd.DataFrame, note: str) -> dict:
    return {"source": source, "model": label, "note": note, "n_seeds": len(part),
            "seeds": " ".join(str(int(s)) for s in sorted(part.seed)),
            "macs_per_symbol_merged": int(part.macs_per_symbol_merged.iloc[0]),
            "test_reduction_valbest_mean": part[METRIC].mean(), "test_reduction_valbest_min": part[METRIC].min(),
            "test_reduction_valbest_max": part[METRIC].max(), "q2_db_raw": part.q2_db_raw.mean(),
            "valbest_q2_gain_db_mean": part.valbest_q2_gain_db.mean(),
            "valbest_q2_gain_db_min": part.valbest_q2_gain_db.min(),
            "valbest_q2_gain_db_max": part.valbest_q2_gain_db.max(), "final_q2_gain_db_mean": part.q2_gain_db.mean()}


def best_by_validation(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    means = frame.groupby(keys).best_val_reduction.mean()
    best = means.idxmax()
    best = best if isinstance(best, tuple) else (best,)
    mask = np.logical_and.reduce([frame[k] == v for k, v in zip(keys, best)])
    return frame[mask]


def main() -> int:
    frames, skipped = {}, []
    for exp, relative in TABLES.items():
        path = ROOT / "experiments" / relative
        if not path.is_file() and not (path.parent / "runs").is_dir():
            skipped.append(f"{exp}: нет {relative}")
            continue
        frame, source = load(exp, relative)
        missing = frame.raw_ber.isna() | frame.test_ber.isna()
        for run_id in frame.run_id[missing]:
            skipped.append(f"{exp}: {run_id} без raw_ber/test_ber")
        frame = add_q2(frame[~missing])
        out = path.with_name(path.stem + "_q2.csv")
        frame.to_csv(out, index=False)
        frames[exp] = frame
        print(f"{exp}: {len(frame)} прогонов из {source} -> {out.relative_to(ROOT)}")

    rows = []
    if "E012" in frames:
        e012 = frames["E012"]
        base = e012[(e012.lr_branch == "const_1e-3") & (e012.window == 31) & (e012.budget == 20000)]
        for variant, label in (("v1_two_kerr_both", "complex_both"), ("v2_two_kerr_last", "complex_a"),
                               ("v3_one_hidden", "one_hidden")):
            part = base[base.variant == variant]
            if len(part):
                rows.append(key_row("E012", label, part, "window 31, hidden 48, N=30, 20k, const 1e-3"))
    if "E011" in frames:
        e011 = frames["E011"]
        for variant in ("complex_both", "complex_a", "complex_one_hidden", "real_tanh"):
            for n_train in (30, 210):
                part = e011[(e011.variant == variant) & (e011.lr_name == "const_1e-3")
                            & (e011.n_train_nominal == n_train)]
                if len(part):
                    label = "one_hidden" if variant == "complex_one_hidden" else variant
                    rows.append(key_row("E011", label, part, f"window 31, N={n_train}, {int(part.steps.max())} steps, "
                                        "const 1e-3; real_tanh hidden 96"))
    if "E013" in frames:
        e013 = frames["E013"]
        trans = e013[e013.model == "transformer"]
        if len(trans):
            part = best_by_validation(trans, ["variant", "window", "n_train_nominal", "budget"])
            rows.append(key_row("E013", "best transformer (by validation)", part,
                                f"{part.variant.iloc[0]}, window {int(part.window.iloc[0])}, "
                                f"N={int(part.n_train_nominal.iloc[0])}, {int(part.budget.iloc[0])} steps"))
    if "E014" in frames:
        e014 = frames["E014"]
        hybrids = e014[e014.variant.isin(["k2_parallel_joint", "k3_parallel_staged", "k4_cascade_staged"])].copy()
        if len(hybrids):
            hybrids["grid"] = hybrids.get("spec_grid")
            part = best_by_validation(hybrids, ["variant", "grid"])
            rows.append(key_row("E014", "best Kerr + KAN hybrid (by validation)", part,
                                f"{part.variant.iloc[0]}, G={int(part.grid.iloc[0])}, 40k steps; "
                                "multiplies: sparse convention (dense in kan_hybrid_q2.csv)"))
    out = ROOT / "results" / "q2_key_points.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"ключевые точки: {out.relative_to(ROOT)} ({len(rows)} строк)")
    (ROOT / "results" / "q2_skipped.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2) + "\n",
                                                      encoding="utf-8")
    print(f"пропущено: {len(skipped)}" + (": " + "; ".join(skipped[:5]) if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
