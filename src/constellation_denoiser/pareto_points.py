"""Points 'configuration -> mean and min-max over seeds' collected from the per-run tables of E010-E022.

Used by summarize_E021 (Pareto plot, level table, members for E023) and summarize_E022/E023.
A point: family, source experiment, variant, lr branch, step budget, MACs (after merging; KAN sparse),
number of seeds, mean validation, test mean/min/max and Q2 gain mean/min/max, all at the
validation-best evaluation. Only N = 30 training files (the common setting) is used.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .paths import EXPERIMENTS_DIR
from .q2 import q2_db

TEST, VAL = "test_reduction_valbest_select", "best_val_reduction"
OURS = ("complex Kerr MLP", "Phys + MLP", "PhysKerr", "DBP")
BASELINES = ("линейный FIR", "DBP (заданная нелинейность)", "CNN (приближение)")

# (experiment folder, per-run table relative to results/ or results_smoke/, use smoke results when SMOKE=1)
SOURCES = [
    ("E010_lr_schedule_study", "lr_sweep_q2.csv", False), ("E011_long_runs_convergence", "long_runs_q2.csv", False),
    ("E012_depth_vs_window", "depth_window_q2.csv", False), ("E013_transformer_scaling", "transformer_runs_q2.csv", False),
    ("E014_kerr_kan_hybrid", "kan_hybrid_q2.csv", False), ("E016_phys_kerr_only", "phys_kerr.csv", False),
    ("E017_phys_plus_mlp", "phys_plus_mlp.csv", False), ("E018_dbp_depth", "dbp_depth.csv", False),
    ("E020_e017_anomaly", "runs_table.csv", True), ("E021_pareto", "runs_table.csv", True),
    ("E022_baselines", "runs_table.csv", True),
]


def family_of_row(row) -> str:
    model = row.get("model")
    variant = str(row.get("variant", ""))
    if model in ("complex_both", "complex_a", "complex_one_hidden"):
        return "complex Kerr MLP"
    if model == "phys_kerr":
        return "линейный FIR" if float(row.get("spec_n_alpha", 1) or 0) == 0 else "PhysKerr"
    if model == "phys_plus_mlp":
        return "Phys + MLP"
    if model == "phys_dbp":
        return "DBP (заданная нелинейность)" if variant.startswith("dbp_fixed") else "DBP"
    if model == "real_tanh":
        return "real tanh MLP"
    if model == "transformer":
        return "transformer"
    if model in ("kan", "kerr_kan"):
        return "KAN"
    if model == "cnn":
        return "CNN (приближение)"
    return str(model)


def _q2_column(frame: pd.DataFrame) -> pd.Series:
    for col in ("valbest_q2_gain_db", "valbest_select_q2_gain_db"):
        if col in frame:
            return frame[col]
    return pd.Series([q2_db(r * (1 - t / 100)) - q2_db(r) for r, t in zip(frame.raw_ber, frame[TEST])], index=frame.index)


def load_per_run(smoke: bool) -> pd.DataFrame:
    frames = []
    for folder, name, smoke_aware in SOURCES:
        path = EXPERIMENTS_DIR / folder / ("results_smoke" if smoke and smoke_aware else "results") / name
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if "n_train_nominal" in frame:
            frame = frame[frame.n_train_nominal.fillna(30).astype(int) == 30]
        frame = frame.assign(source=folder[:4], q2=_q2_column(frame))
        frames.append(frame)
    confirm = EXPERIMENTS_DIR / "E019_confirm_long_runs" / "results" / "confirm.csv"
    if confirm.is_file():
        c = pd.read_csv(confirm)
        member_family = {"E016": "PhysKerr", "E017": "Phys + MLP", "E018": "DBP", "control": "complex Kerr MLP"}
        choice_path = confirm.with_name("confirm_choice.json")
        specs = {}
        if choice_path.is_file():
            for m in json.loads(choice_path.read_text(encoding="utf-8")):
                name = f"{m['experiment']}_{m['spec'].get('name', m['variant'])}"
                specs[name] = {k: v for k, v in m["spec"].items() if k != "name"}
        spec_frame = pd.DataFrame([{f"spec_{k}": v for k, v in specs.get(m, {}).items()} for m in c.member])
        frames.append(pd.concat([pd.DataFrame({
            "source": "E019", "variant": c.member, "lr_name": c.lr_name, "budget": c.nominal_horizon, "seed": c.seed,
            VAL: c.best_val_reduction, TEST: c.test_reduction_valbest, "macs_per_symbol_merged": c.macs_per_symbol_merged,
            "q2": c.valbest_q2_gain_db, "family": [member_family[m.split("_")[0]] for m in c.member],
            "model": "confirm", "diverged": False}), spec_frame], axis=1))
    runs = pd.concat(frames, ignore_index=True, sort=False)
    if "family" not in runs:
        runs["family"] = None
    runs["family"] = [f if isinstance(f, str) and f else family_of_row(r) for f, (_, r) in zip(runs.family, runs.iterrows())]
    runs = runs[~runs.get("diverged", pd.Series(False, index=runs.index)).fillna(False).astype(bool)]
    return runs


def points(smoke: bool = False) -> pd.DataFrame:
    runs = load_per_run(smoke)
    keys = ["family", "source", "variant", "lr_name", "budget"]
    grouped = runs.groupby(keys, dropna=False)
    out = grouped.agg(macs=("macs_per_symbol_merged", "first"), n_seeds=("seed", "nunique"), val_mean=(VAL, "mean"),
                      test_mean=(TEST, "mean"), test_min=(TEST, "min"), test_max=(TEST, "max"),
                      q2_mean=("q2", "mean"), q2_min=("q2", "min"), q2_max=("q2", "max")).reset_index()
    first = grouped.first().reset_index()
    spec_cols = [c for c in first.columns if c.startswith("spec_") or c.startswith("lr_") and c != "lr_name"]
    out = out.merge(first[keys + spec_cols], on=keys, how="left")
    out["label"] = out.source + " · " + out.variant.astype(str) + " · " + out.lr_name.astype(str) + " · " + \
        (out.budget // 1000).astype(int).astype(str) + "k"
    return out


def level_of(macs: float, levels: list[int], factor: float) -> int | None:
    for level in levels:
        if level / factor <= macs <= level * factor:
            return level
    return None


def best_per_level(pts: pd.DataFrame, levels: list[int], factor: float) -> pd.DataFrame:
    pts = pts.assign(level=[level_of(m, levels, factor) for m in pts.macs])
    pts = pts.dropna(subset=["level"])
    idx = pts.groupby(["level", "family"]).val_mean.idxmax()
    return pts.loc[idx].sort_values(["level", "test_mean"], ascending=[True, False])


def spec_from_point(row) -> dict:
    spec = {}
    for key, value in row.items():
        if key.startswith("spec_") and value == value and value is not None:
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, float) and value.is_integer() and key != "spec_sym":
                value = int(value)
            if key == "spec_sym":
                value = bool(value) if not isinstance(value, str) else value == "True"
            spec[key[5:]] = value
    return spec


def lr_from_point(row) -> dict | None:
    from dataclasses import fields

    from .lr_schedules import LRConfig

    allowed = {f.name for f in fields(LRConfig)}
    cfg = {k[3:]: (v.item() if hasattr(v, "item") else v) for k, v in row.items()
           if k.startswith("lr_") and k[3:] in allowed and v == v and v is not None}
    if not cfg:
        name = str(row.get("lr_name", ""))
        if name == "const_1e-3":
            return {"peak_lr": 0.001, "schedule": "const", "warmup": 0, "c_mult": 1.0}
        if name.startswith("best_e010_cm"):        # E019 branches: E010 stage3 choice with this c_mult
            choice = EXPERIMENTS_DIR / "E010_lr_schedule_study" / "results" / "stage3_choice.json"
            if choice.is_file():
                cfg = dict(json.loads(choice.read_text(encoding="utf-8"))["a"]["lr_config"])
                cfg.update(c_mult=float(name.rsplit("cm", 1)[1]), horizon=None)
                return cfg
        return None
    if "horizon" in cfg:
        cfg["horizon"] = None
    return cfg


def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=lambda v: v.item() if hasattr(v, "item") else str(v))


def nan_safe(value) -> float:
    return float(value) if value is not None and np.isfinite(value) else float("nan")
