"""Aggregator v2 (E024 part 6, E033): Pareto points 'RM per symbol -> quality' from per-run tables.

Differences from pareto_points (E021), which merged configurations that share a variant name (E012:
all windows of one variant in one point, so v4 w31 fell out) and only knew four levels:
- a point is one configuration: source, variant, every spec_* and lr_* field, loss, step budget;
  it needs >= MIN_SEEDS seeds;
- the family comes from an explicit table results/family_map.csv (source, model, variant -> family;
  variant "*" covers every variant of that source and model), never from substrings;
- level L holds the points with |ln(RM / L)| <= ln(1.25) (+-25 % on a log scale);
- the best point of a family at a level and the Pareto envelope are chosen by validation
  (mean best_val_reduction); test and Q2 gain at the validation-best evaluation are reported.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .paths import EXPERIMENTS_DIR, PROJECT_ROOT
from .q2 import q2_db

TEST, VAL = "test_reduction_valbest_select", "best_val_reduction"
FAMILY_MAP = PROJECT_ROOT / "results" / "family_map.csv"
MIN_SEEDS = 3
LEVEL_FACTOR = 1.25
LEVELS = [300, 600, 1000, 2000, 4000, 6500, 10000, 16000, 25000, 40000]

# per-run tables: (folder, file under results/); E019 is read from confirm.csv
TABLES = [
    ("E010_lr_schedule_study", "lr_sweep_q2.csv"), ("E011_long_runs_convergence", "long_runs_q2.csv"),
    ("E012_depth_vs_window", "depth_window_q2.csv"), ("E013_transformer_scaling", "transformer_runs_q2.csv"),
    ("E014_kerr_kan_hybrid", "kan_hybrid_q2.csv"), ("E016_phys_kerr_only", "phys_kerr.csv"),
    ("E017_phys_plus_mlp", "phys_plus_mlp.csv"), ("E018_dbp_depth", "dbp_depth.csv"),
    ("E020_e017_anomaly", "runs_table.csv"), ("E021_pareto", "runs_table.csv"), ("E022_baselines", "runs_table.csv"),
    ("E023_split_robustness", "runs_table.csv"),
    ("E025_family_tuning", "runs_table.csv"), ("E026_kerr_complexity_map", "runs_table.csv"),
    ("E027_activation_ablation", "runs_table.csv"), ("E028_dbp_depth_layerwise", "runs_table.csv"),
    ("E029_frontend_vs_width", "runs_table.csv"), ("E030_loss_functions", "runs_table.csv"),
    ("E031_perturbation_triplets", "runs_table.csv"), ("E032_convergence_1m", "runs_table.csv"),
    ("E033_pareto_final", "runs_table.csv"),
]
E019_FAMILY = {"E016": "PhysKerr", "E017": "Phys + MLP", "E018": "DBP", "control": "Kerr MLP"}
FAMILY_ORDER = ["Kerr MLP", "complex MLP, other activation", "Phys + MLP", "PhysKerr", "DBP", "DBP + MLP",
                "PB-NLC (triplets)", "real MLP", "KAN", "transformer", "linear FIR", "DBP (fixed nonlinearity)", "CNN"]


def load_tables(smoke: bool = False) -> pd.DataFrame:
    frames = []
    for folder, name in TABLES:
        path = EXPERIMENTS_DIR / folder / ("results_smoke" if smoke and folder >= "E024" else "results") / name
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if "stage" in frame:
            frame = frame[frame.stage != "control"]
        if "control_key" in frame:
            frame = frame[frame.control_key.isna()]
        if "n_train_nominal" in frame:
            frame = frame[frame.n_train_nominal.fillna(30).astype(int) == 30]
        if folder.startswith("E023"):          # other splits: not comparable with the common test set
            continue
        frames.append(frame.assign(source=folder[:4]))
    confirm = EXPERIMENTS_DIR / "E019_confirm_long_runs" / "results" / "confirm.csv"
    if confirm.is_file():
        c = pd.read_csv(confirm)
        choice = confirm.with_name("confirm_choice.json")
        specs = {}
        if choice.is_file():
            for m in json.loads(choice.read_text(encoding="utf-8")):
                specs[f"{m['experiment']}_{m['spec'].get('name', m['variant'])}"] = m["spec"]
        model = {"control_complex_a": "complex_a", "control_complex_both": "complex_both"}
        spec_rows = [{f"spec_{k}": v for k, v in specs.get(m, {}).items() if k != "name"} for m in c.member]
        frames.append(pd.concat([pd.DataFrame({
            "source": "E019", "variant": c.member, "model": [model.get(m, specs.get(m, {}).get("model", "confirm"))
                                                             for m in c.member],
            "lr_name": c.lr_name, "budget": c.nominal_horizon, "seed": c.seed, VAL: c.best_val_reduction,
            TEST: c.test_reduction_valbest, "macs_per_symbol_merged": c.macs_per_symbol_merged,
            "valbest_q2_gain_db": c.valbest_q2_gain_db, "diverged": False, "raw_ber": np.nan}), pd.DataFrame(spec_rows)], axis=1))
    runs = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if runs.empty:
        return runs
    runs = runs[~runs.get("diverged", pd.Series(False, index=runs.index)).fillna(False).astype(bool)]
    q2 = runs.get("valbest_q2_gain_db")
    if "valbest_select_q2_gain_db" in runs:
        q2 = runs.valbest_select_q2_gain_db.where(runs.valbest_select_q2_gain_db.notna(), q2)
    raw = runs.raw_ber.fillna(runs.raw_ber.dropna().median() if runs.raw_ber.notna().any() else 0.009551859551859552)
    computed = pd.Series([q2_db(r * (1 - t / 100)) - q2_db(r) for r, t in zip(raw, runs[TEST])], index=runs.index)
    runs["q2"] = q2.where(q2.notna(), computed) if q2 is not None else computed
    return runs.reset_index(drop=True)


# --------------------------------------------------------------------------- explicit family table

def default_family(source: str, model: str, variant: str) -> str:
    """Used once to write results/family_map.csv; afterwards the table is the source of truth."""
    if source == "E019":
        return E019_FAMILY.get(variant.split("_")[0], "unmapped")
    if model in ("complex_both", "complex_a", "complex_one_hidden"):
        return "Kerr MLP"
    if model == "act_mlp":
        return "Kerr MLP" if variant.startswith("kerr_c_train") else "complex MLP, other activation"
    if model == "phys_kerr":
        return "linear FIR" if variant.startswith("fir_") else "PhysKerr"
    if model == "phys_plus_mlp":
        return "Phys + MLP"
    if model in ("phys_dbp", "dbp_shared", "dbp_noise"):
        return "DBP (fixed nonlinearity)" if variant.startswith("dbp_fixed") else "DBP"
    if model == "dbp_mlp":
        return "DBP + MLP"
    if model in ("real_tanh", "real_mlp"):
        return "real MLP"
    if model in ("kan", "kerr_kan"):
        return "KAN"
    if model == "transformer":
        return "transformer"
    if model == "cnn":
        return "CNN"
    if model == "triplets":
        return "PB-NLC (triplets)"
    return "unmapped"


WILDCARD_SOURCES = {"E025": ["complex_one_hidden", "complex_both", "complex_a", "phys_kerr", "phys_dbp", "phys_plus_mlp",
                             "real_mlp", "kerr_kan", "transformer"],
                    "E026": ["complex_one_hidden", "complex_both"], "E028": ["phys_dbp", "dbp_shared", "dbp_noise", "dbp_mlp"],
                    "E029": ["complex_one_hidden", "phys_plus_mlp"],
                    "E030": ["complex_one_hidden", "phys_dbp", "phys_plus_mlp"], "E031": ["triplets"],
                    "E032": ["complex_one_hidden", "complex_both", "phys_plus_mlp", "phys_dbp", "real_mlp"],
                    "E033": ["complex_one_hidden", "complex_both", "phys_plus_mlp", "phys_dbp", "real_mlp"]}


def write_family_map(path=FAMILY_MAP) -> pd.DataFrame:
    """Explicit rows for every (source, model, variant) of E010-E023 and wildcard rows for E025+."""
    runs = load_tables()
    rows = []
    if not runs.empty:
        combos = runs[["source", "model", "variant"]].astype(str).drop_duplicates()
        rows = [{"source": s, "model": m, "variant": v, "family": default_family(s, m, v)} for s, m, v in combos.values]
    for source, models in WILDCARD_SOURCES.items():
        rows += [{"source": source, "model": m, "variant": "*", "family": default_family(source, m, "")} for m in models]
    config = json.loads((EXPERIMENTS_DIR / "E027_activation_ablation" / "config.json").read_text(encoding="utf-8")) \
        if (EXPERIMENTS_DIR / "E027_activation_ablation" / "config.json").is_file() else {"activations": [], "stages": {}}
    widths = sorted({config.get("stages", {}).get("s1", {}).get("hidden", 48)} |
                    set(config.get("stages", {}).get("s2", {}).get("hidden", [])))
    for act in config.get("activations", []):
        for hidden in widths:
            v = f"{act}_h{hidden}"
            rows.append({"source": "E027", "model": "act_mlp", "variant": v, "family": default_family("E027", "act_mlp", v)})
    frame = pd.DataFrame(rows).drop_duplicates(["source", "model", "variant"]).sort_values(["source", "model", "variant"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def family_of(runs: pd.DataFrame, path=FAMILY_MAP) -> pd.Series:
    table = pd.read_csv(path) if path.is_file() else write_family_map(path)
    exact = {(s, m, v): f for s, m, v, f in table[["source", "model", "variant", "family"]].astype(str).values}
    out = []
    for s, m, v in runs[["source", "model", "variant"]].astype(str).values:
        out.append(exact.get((s, m, v)) or exact.get((s, m, "*")) or "unmapped")
    return pd.Series(out, index=runs.index)


# --------------------------------------------------------------------------- points

def signature(row: pd.Series, prefix: str, skip: tuple[str, ...]) -> str:
    items = [f"{k[len(prefix):]}={row[k]}" for k in sorted(row.index) if k.startswith(prefix) and k not in skip
             and not (isinstance(row[k], float) and math.isnan(row[k]))]
    return ";".join(items)


def points(runs: pd.DataFrame, min_seeds: int = MIN_SEEDS) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    runs = runs.copy()
    runs["family"] = family_of(runs)
    runs["spec_sig"] = [signature(r, "spec_", ("spec_name",)) for _, r in runs.iterrows()]
    runs["lr_sig"] = [signature(r, "lr_", ("lr_name", "lr_branch", "lr_horizon")) or str(r.get("lr_name"))
                      for _, r in runs.iterrows()]
    if "loss" not in runs:
        runs["loss"] = "mse"
    runs["loss"] = runs["loss"].fillna("mse")
    keys = ["family", "source", "variant", "spec_sig", "lr_sig", "loss", "budget"]
    grouped = runs.groupby(keys, dropna=False)
    out = grouped.agg(rm=("macs_per_symbol_merged", "first"), n_seeds=("seed", "nunique"), val_mean=(VAL, "mean"),
                      val_min=(VAL, "min"), val_max=(VAL, "max"), test_mean=(TEST, "mean"), test_min=(TEST, "min"),
                      test_max=(TEST, "max"), q2_mean=("q2", "mean"), q2_min=("q2", "min"), q2_max=("q2", "max"),
                      lr_name=("lr_name", "first"), model=("model", "first")).reset_index()
    out = out[out.n_seeds >= min_seeds]
    out["label"] = out.source + " · " + out.variant.astype(str) + " · " + out.lr_name.astype(str) + " · " + \
        (out.budget.fillna(0) // 1000).astype(int).astype(str) + "k"
    return out.reset_index(drop=True)


def level_of(rm: float, levels=LEVELS, factor: float = LEVEL_FACTOR) -> int | None:
    for level in levels:
        if rm > 0 and abs(math.log(rm / level)) <= math.log(factor):
            return level
    return None


def best_per_level(pts: pd.DataFrame, levels=LEVELS, factor: float = LEVEL_FACTOR) -> pd.DataFrame:
    if pts.empty:
        return pts
    pts = pts.assign(level=[level_of(m, levels, factor) for m in pts.rm]).dropna(subset=["level"])
    if pts.empty:
        return pts
    idx = pts.groupby(["level", "family"]).val_mean.idxmax()
    return pts.loc[idx].sort_values(["level", "val_mean"], ascending=[True, False]).reset_index(drop=True)


def envelope(pts: pd.DataFrame) -> pd.DataFrame:
    """Pareto envelope chosen by validation: no other point with lower RM has a higher mean validation."""
    if pts.empty:
        return pts
    order = pts.sort_values(["rm", "val_mean"], ascending=[True, False])
    keep, best = [], -math.inf
    for index, row in order.iterrows():
        if row.val_mean > best:
            keep.append(index)
            best = row.val_mean
    return order.loc[keep]


# --------------------------------------------------------------------------- E024: before -> after

def e024_report(config: dict, smoke: bool, out, figures) -> None:
    runs = load_tables()
    if not FAMILY_MAP.is_file():
        write_family_map()
    pts = points(runs)
    pts.to_csv(out / "pareto_v2_points.csv", index=False)
    best = best_per_level(pts)
    best.to_csv(out / "pareto_v2_level_best.csv", index=False)
    old_path = EXPERIMENTS_DIR / "E021_pareto" / "results" / "level_best.csv"
    rows = []
    if old_path.is_file():
        old = pd.read_csv(old_path)
        for level, part in old.groupby("level"):
            new = pts.assign(level=[level_of(m, [level], LEVEL_FACTOR) for m in pts.rm]).dropna(subset=["level"])
            top_new = new.loc[new.val_mean.idxmax()] if len(new) else None
            top_old = part.loc[part.val_mean.idxmax()]
            rows.append({"level": int(level), "old_best": top_old.label, "old_family": top_old.family,
                         "old_rm": top_old.macs, "old_val": top_old.val_mean, "old_test": top_old.test_mean,
                         "new_best": None if top_new is None else top_new.label,
                         "new_family": None if top_new is None else top_new.family,
                         "new_rm": None if top_new is None else top_new.rm,
                         "new_val": None if top_new is None else top_new.val_mean,
                         "new_test": None if top_new is None else top_new.test_mean,
                         "new_n_seeds": None if top_new is None else top_new.n_seeds})
    before_after = pd.DataFrame(rows)
    before_after.to_csv(out / "e021_before_after.csv", index=False)
    lines = ["| уровень RM | было (E021): точка, тест % | стало (агрегатор v2): точка, тест % |", "|---|---|---|"]
    for _, r in before_after.iterrows():
        lines.append(f"| {r.level} | {r.old_best} ({r.old_family}), {r.old_test:.2f} | "
                     f"{r.new_best} ({r.new_family}), {r.new_test:.2f} |")
    (out / "e021_before_after.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_pareto(pts, envelope(pts), figures / "E024_pareto_v2.png", "test", "Preliminary Pareto (aggregator v2, all points"
                " with >= 3 seeds; envelope by validation)")


def plot_pareto(pts: pd.DataFrame, env: pd.DataFrame, path, metric: str, title: str, hd_fec: float | None = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from . import plot_style

    plot_style.apply()
    col = {"test": ("test_mean", "test_min", "test_max", "test BER reduction, %"),
           "q2": ("q2_mean", "q2_min", "q2_max", "Q2 gain, dB")}[metric]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    markers = "osD^vP*Xhp<>8"
    handles = []
    families = [f for f in FAMILY_ORDER if f in set(pts.family)] + sorted(set(pts.family) - set(FAMILY_ORDER))
    shown = pts[pts[col[0]] > (-10 if metric == "test" else -0.3)]
    for index, family in enumerate(families):
        part = shown[shown.family == family]
        if part.empty:
            continue
        c, m = plot_style.SERIES[index % len(plot_style.SERIES)], markers[index % len(markers)]
        ax.errorbar(part.rm.to_numpy(float), part[col[0]].to_numpy(float),
                    yerr=[np.clip((part[col[0]] - part[col[1]]).to_numpy(float), 0, None),
                          np.clip((part[col[2]] - part[col[0]]).to_numpy(float), 0, None)],
                    fmt=m, color=c, markersize=5, alpha=0.75, capsize=1.5, linewidth=0.8)
        handles.append(Line2D([], [], color=c, marker=m, linestyle="", label=family))
    if len(env):
        ax.plot(env.rm.to_numpy(float), env[col[0]].to_numpy(float), "-", color=plot_style.TEXT, linewidth=1.2,
                drawstyle="steps-post")
        handles.append(Line2D([], [], color=plot_style.TEXT, label="Pareto envelope (chosen by validation)"))
    if hd_fec is not None:
        ax.axhline(hd_fec, color=plot_style.TEXT, linestyle="--", linewidth=1)
        handles.append(Line2D([], [], color=plot_style.TEXT, linestyle="--", label="HD-FEC 3.8e-3"))
    ax.set_xscale("log")
    ax.set_xlabel("real multiplies per symbol (RM, log scale)")
    ax.set_ylabel(col[3] + " (min-max over seeds)")
    ax.set_title(title, fontsize=10)
    ax.legend(handles=handles, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- E033

def confirmed(pts_runs: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Confirmed runs: >= 200k steps (closed-form E031 exempt), sources listed in config, recipe stored."""
    rule = config["confirmed"]
    runs = pts_runs[pts_runs.source.isin(rule["sources"])]
    long = runs.budget.fillna(0) >= rule["min_steps"]
    closed = runs.source.isin(rule.get("closed_form_sources", []))
    if "recipe_source" in runs:
        recipe_ok = runs.recipe_source.notna() | ~runs.source.isin(rule.get("recipe_required", []))
    else:
        recipe_ok = ~runs.source.isin(rule.get("recipe_required", []))
    return runs[(long | closed) & recipe_ok]


def missing_levels(config: dict, smoke: bool) -> list[dict]:
    """Up to fill.max_runs / fill.seeds configurations: Kerr one_hidden w31 at RM levels without a confirmed point."""
    runs = load_tables(smoke)
    pts = points(confirmed(runs, config), min_seeds=config["confirmed"]["min_seeds"]) if not runs.empty else pd.DataFrame()
    have = set()
    if not pts.empty:
        have = {level_of(m) for m in pts.rm}
    items = []
    cap = config["fill"]["max_runs"] // config["fill"]["seeds"]
    for level in LEVELS:
        if level in have:
            continue
        hidden = max(1, round(level / (4 * 31 + 11)))
        spec = {"model": "complex_one_hidden", "window": 31, "hidden": hidden}
        key = min(config["fill"]["recipes"], key=lambda k: abs(math.log(hidden / config["fill"]["recipes"][k])))
        items.append({"name": f"fill_oh_w31_h{hidden}", "family": "Kerr MLP", "level": level, "spec": spec, "recipe": key})
    return items[:cap]
