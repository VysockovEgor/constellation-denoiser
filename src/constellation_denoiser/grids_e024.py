"""Task lists for E024-E033, registered into grids.BUILDERS / SPECIAL_STAGES / POST_HOOKS by grids.py.

Common choices (see DECISIONS.md): N = 30 train files, validation 45, test 45 as in E004-E023;
evaluation (validation, and test at the same points for reporting) every max(500, steps / 40) steps;
selection between stages only by mean ``best_val_reduction`` over seeds, diverged configurations excluded.
Stages that depend on an earlier stage stop with a message until it is complete; with
PLACEHOLDER_SELECTION=1 (estimate only) they take the first configurations instead.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from .grids import (Options, control_tasks, e018_phases, load_config, lr_from, make_task, phys_spec, read_runs,
                    scale_phases)
from .lr_schedules import LRConfig
from .recipes import load_recipe

# --------------------------------------------------------------------------- helpers


def eval_every_for(steps: int) -> int:
    return max(500, (steps // 40) // 500 * 500)


def placeholder() -> bool:
    return os.environ.get("PLACEHOLDER_SELECTION", "0") == "1"


def first_seeds(config: dict, options: Options, n: int) -> list[int]:
    seeds = (options.seeds or config["seeds"])[:n]
    return seeds[:1] if options.smoke else seeds


def results_dir(directory: Path, smoke: bool) -> Path:
    return directory / ("results_smoke" if smoke else "results")


def save_json(directory: Path, options: Options, name: str, payload) -> None:
    if placeholder():
        return
    from .runner import atomic_write_json

    atomic_write_json(results_dir(directory, options.smoke) / name, payload)


def short(value: float) -> str:
    return f"{value:.3g}".replace("e-0", "e-")


def lr_label(lr: dict) -> str:
    cfg = LRConfig.from_dict(lr)
    parts = [{"const": "const", "cosine_to_zero": "cos", "cosine_to_floor": "floor"}[cfg.schedule], short(cfg.peak_lr)]
    if cfg.warmup:
        parts.append(f"wu{cfg.warmup}")
    parts.append(f"cm{cfg.c_mult:g}")
    if cfg.weight_decay:
        parts.append(f"wd{cfg.weight_decay:g}")
    if cfg.optimizer != "adam":
        parts.append(cfg.optimizer)
    return "_".join(parts)


def run(exp: str, stage: str, directory: Path, config: dict, options: Options, *, name: str, spec: dict, lr: dict,
        steps: int, seed: int, lr_name: str | None = None, phases: list[dict] | None = None, meta: dict | None = None,
        eval_every: int | None = None, files: dict | None = None, **extra) -> dict:
    lr = lr_from(lr, horizon=None)
    lr_name = lr_name or lr_label(lr)
    every = eval_every or eval_every_for(steps)
    task = make_task(exp, stage, directory, options, spec=dict(spec, name=name), seed=seed,
                     n_train=config["n_train_files"], phases=scale_phases(phases or [{"steps": 1}], steps), lr=lr,
                     lr_name=lr_name, eval_every=every, select_every=every, variant=name,
                     run_id=f"{name}_{lr_name}_{steps // 1000}k_seed{seed}", meta=dict(meta or {}),
                     ckpt_every=config.get("ckpt_every", 10_000), **extra)
    if files is not None:
        task["files"] = files
    return task


def group_val(runs: list[dict], key: str, budget: int | None = None) -> dict:
    """{meta[key]: (mean best_val_reduction, n, any diverged)} over runs of one budget."""
    groups: dict = {}
    for r in runs:
        if r.get(key) is None or (budget is not None and int(r["budget"]) != budget):
            continue
        groups.setdefault(r[key], []).append(r)
    return {k: (float(np.mean([m["best_val_reduction"] for m in v])), len(v), any(m.get("diverged") for m in v))
            for k, v in groups.items()}


def require(expected: list[dict], runs: list[dict], what: str) -> bool:
    """True if every expected task has a result; otherwise stop (or continue in placeholder mode)."""
    have = {r["run_id"] for r in runs}
    missing = [t["run_id"] for t in expected if t["run_id"] not in have]
    if not missing:
        return True
    if placeholder():
        return False
    raise SystemExit(f"{what}: нет {len(missing)} из {len(expected)} прогонов (например {missing[0]}); "
                     "сначала завершите предыдущую стадию")


def best_keys(stats: dict, n: int, fallback: list) -> list:
    ok = [(k, v[0]) for k, v in stats.items() if not v[2] and math.isfinite(v[0])]
    ranked = [k for k, _ in sorted(ok, key=lambda kv: kv[1], reverse=True)]
    return ranked[:n] if ranked else fallback[:n]


def recipe_lr(key: str, options: Options, scale: float = 1.0) -> tuple[dict, dict, str]:
    recipe = load_recipe(key, options.smoke)
    lr = dict(recipe["lr"])
    lr["peak_lr"] = lr["peak_lr"] * scale
    return lr, recipe["spec_overrides"], recipe["source"]


# --------------------------------------------------------------------------- E024 (dedup retrain if leakage)

def e024_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E024"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage != "dedup":
        raise SystemExit(f"E024: неизвестная стадия {stage!r}")
    item = config["dedup_retrain"]
    lr, _, source = recipe_lr("F3", options)
    lr = lr_from(config["lr_policies"][item["lr"]]) if item.get("lr") in config.get("lr_policies", {}) else lr
    return [run(exp, stage, directory, config, options, name=item["name"], spec=item["spec"], lr=lr,
                steps=item["steps"], seed=seed, post_hook="dedup_eval", meta={"recipe_source": source})
            for seed in first_seeds(config, options, 1)]


def dedup_eval_hook(model, task, train, val, test) -> dict:
    """BER on the test windows whose centred K-symbol label sequence never occurs in the train files."""
    import torch

    from .audit_e024 import label_windows, labels_of
    from .experiment import ber_from_iq
    from .window_data import predict_iq

    window = int(task["spec"]["window"])
    _, config = load_config("E024")
    k = int(config["leakage"]["rule_k"])
    seen = set()
    for row in labels_of(train.tx_c):
        seen.update(label_windows(row, k))
    keep = []
    edge = window // 2
    for row in labels_of(test.tx_c):
        keys = label_windows(row, k)       # window j covers symbols j .. j+k-1; centre j + k//2
        mask = np.ones(len(row), dtype=bool)
        for j, key in enumerate(keys):
            if key in seen:
                mask[j + k // 2] = False
        keep.append(mask[edge: len(row) - edge])
    keep = np.concatenate(keep)
    model = model.eval()
    with torch.inference_mode():
        pred = predict_iq(model, test, window)
    _, tx_c = test.centers(window)
    rx_c, _ = test.centers(window)
    raw_all = ber_from_iq(np.column_stack((rx_c.real, rx_c.imag)), tx_c)
    raw_dedup = ber_from_iq(np.column_stack((rx_c.real[keep], rx_c.imag[keep])), tx_c[keep])
    ber_all, ber_dedup = ber_from_iq(pred, tx_c), ber_from_iq(pred[keep], tx_c[keep])
    return {"dedup_k": k, "dedup_kept_fraction": float(keep.mean()),
            "final_reduction_all": 100 * (raw_all - ber_all) / raw_all,
            "final_reduction_dedup": 100 * (raw_dedup - ber_dedup) / raw_dedup if raw_dedup > 0 else float("nan")}


# --------------------------------------------------------------------------- E025 (equal tuning budget)

def _categorical(u: float, values: list):
    return values[min(int(u * len(values)), len(values) - 1)]


def sobol_points(n: int, dims: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc

    m = max(0, math.ceil(math.log2(max(n, 1))))
    return qmc.Sobol(d=dims, scramble=True, seed=seed).random_base2(m)[:n]


def e025_configs(key: str, pair: dict, config: dict) -> list[dict]:
    """Sobol configurations of one pair plus ref_current and ref_e010best (ids F2_s00 ..., F2_ref_current ...)."""
    space = config["spaces"][pair["space"]]
    dims = list(space["dims"])
    points = sobol_points(pair.get("n_sobol", config["n_sobol"]), len(dims), config["sobol_seed"])
    lo, hi = math.log10(space["peak_lr"][0]), math.log10(space["peak_lr"][1])
    items = []
    for index, u in enumerate(points):
        values = dict(zip(dims, u))
        lr = dict(space.get("base_lr", {}))
        lr["peak_lr"] = float(f"{10 ** (lo + values['peak_lr'] * (hi - lo)):.3g}")
        schedule = _categorical(values["schedule"], space["schedule"])
        lr["schedule"], lr["floor_frac"] = ("cosine_to_floor", space["floor_frac"]) if schedule == "cosine_to_floor" \
            else (schedule, 0.1)
        if "warmup" in values:
            lr["warmup"] = _categorical(values["warmup"], space["warmup"])
        if "c_mult" in values:
            lr["c_mult"] = float(_categorical(values["c_mult"], space["c_mult"]))
        wd = float(_categorical(values["weight_decay"], space["weight_decay"])) if "weight_decay" in values else 0.0
        lr["weight_decay"] = wd
        if wd > 0 or space.get("base_lr", {}).get("optimizer") == "adamw":
            lr["optimizer"] = "adamw"
        overrides = {}
        if "activation" in values:
            overrides["activation"] = _categorical(values["activation"], space["activation"])
        if "init_scale" in values:
            overrides["init_scale"] = float(_categorical(values["init_scale"], space["init_scale"]))
        items.append({"id": f"{key}_s{index:02d}", "kind": "sobol", "lr": LRConfig.from_dict(lr).to_dict(),
                      "spec_overrides": overrides})
    for ref in ("ref_current", "ref_e010best"):
        lr = dict(config["refs"][ref]) if ref == "ref_e010best" else dict(pair["ref_current"])
        items.append({"id": f"{key}_{ref}", "kind": ref, "lr": LRConfig.from_dict(lr).to_dict(),
                      "spec_overrides": dict(pair.get("ref_spec_overrides", {}))})
    return items


def e025_pair_tasks(stage: str, directory: Path, config: dict, options: Options, key: str, items: list[dict],
                    n_seeds: int, steps: int) -> list[dict]:
    pair = config["pairs"][key]
    tasks = []
    for item in items:
        spec = dict(pair["spec"], **item["spec_overrides"])
        for seed in first_seeds(config, options, n_seeds):
            tasks.append(run("E025", stage, directory, config, options, name=item["id"], spec=spec, lr=item["lr"],
                             lr_name="r", steps=steps, seed=seed, phases=pair.get("phases"),
                             meta={"family": pair["family"], "e025_pair": key, "e025_config": item["id"],
                                   "e025_kind": item["kind"], "spec_overrides": item["spec_overrides"]}))
    return tasks


def e025_stage_plan(key: str, config: dict) -> dict:
    pair, stages = config["pairs"][key], config["stages"]
    s3 = dict(stages["s3"], **pair.get("s3", {}))
    return {"s1": stages["s1"], "s2": stages["s2"], "s3": s3}


def e025_selected(directory: Path, config: dict, options: Options, key: str, upto: str) -> list[dict]:
    """Configurations of pair ``key`` that enter stage ``upto`` (s2 or s3)."""
    items = e025_configs(key, config["pairs"][key], config)
    if options.smoke:
        items = items[:2] + items[-2:]
    by_id = {i["id"]: i for i in items}
    plan = e025_stage_plan(key, config)
    runs = [r for r in read_runs(directory, options.smoke) if r.get("e025_pair") == key]
    ref = f"{key}_ref_current"
    s1_tasks = e025_pair_tasks("s1", directory, config, options, key, items, plan["s1"]["seeds"], plan["s1"]["steps"])
    complete = require(s1_tasks, runs, f"E025 {key} s1")
    s1 = group_val(runs, "e025_config", plan["s1"]["steps"] if not options.smoke else None) if complete else {}
    chosen = best_keys(s1, plan["s2"]["top"], [i["id"] for i in items])
    s2_ids = chosen + ([ref] if ref not in chosen and plan["s2"].get("add_ref", True) else [])
    if upto == "s2":
        return [by_id[i] for i in s2_ids]
    s2_tasks = e025_pair_tasks("s2", directory, config, options, key, [by_id[i] for i in s2_ids], plan["s2"]["seeds"],
                               plan["s2"]["steps"])
    complete = complete and require(s2_tasks, runs, f"E025 {key} s2")
    s2 = {k: v for k, v in (group_val(runs, "e025_config", plan["s2"]["steps"] if not options.smoke else None)
                            if complete else {}).items() if k in s2_ids}
    chosen = best_keys(s2, plan["s3"]["top"], s2_ids)
    s3_ids = chosen + ([ref] if ref not in chosen and plan["s3"].get("add_ref", True) else [])
    return [by_id[i] for i in s3_ids]


def e025_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E025"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage not in ("s1", "s2", "s3"):
        raise SystemExit(f"E025: неизвестная стадия {stage!r}")
    tasks, selection = [], {}
    for key in config["pairs"]:
        plan = e025_stage_plan(key, config)[stage]
        if stage == "s1":
            items = e025_configs(key, config["pairs"][key], config)
            if options.smoke:        # smoke: two Sobol points and the two references per pair
                items = items[:2] + items[-2:]
        else:
            items = e025_selected(directory, config, options, key, stage)
            selection[key] = [i["id"] for i in items]
        tasks += e025_pair_tasks(stage, directory, config, options, key, items, plan["seeds"], plan["steps"])
    if stage == "s1":            # code check of the anchor 66.0 (E019 control complex_a, cm30, 200k), in parallel
        check = config["code_check"]
        for seed in first_seeds(config, options, len(check["seeds"])):
            tasks.append(run(exp, stage, directory, config, options, name=check["name"], spec=check["spec"],
                             lr=check["lr"], lr_name="cos_3e-3_cm30", steps=check["steps"], seed=seed,
                             control_key="code_check", meta={"family": "code check"}))
    else:
        save_json(directory, options, f"{stage}_selection.json", selection)
    return tasks


def e025_check_code(options: Options) -> int:
    """Mean test of the code-check runs against config['code_check'] (target, tol); halt the queue if outside."""
    from .gate import set_halt
    from .runner import atomic_write_json, log

    directory, config = load_config("E025")
    check = config["code_check"]
    runs = [r for r in read_runs(directory, options.smoke) if r.get("control_key") == "code_check"]
    values = [r["test_reduction_valbest_select"] for r in runs if r.get("test_reduction_valbest_select") is not None]
    mean = float(np.mean(values)) if values else float("nan")
    ok = len(values) == len(first_seeds(config, options, len(check["seeds"]))) and abs(mean - check["target"]) <= check["tol"]
    atomic_write_json(results_dir(directory, options.smoke) / "code_check.json",
                      {"values": values, "mean": mean, "target": check["target"], "tol": check["tol"], "ok": bool(ok),
                       "source": check["source"]})
    log(f"code check (complex_a cm30 200k): {mean:.2f}% по {len(values)} seed; цель {check['target']}±{check['tol']} -> "
        f"{'OK' if ok else 'НЕ ПРОШЁЛ'}")
    if not ok and not options.smoke:
        set_halt("E025 code check вне допуска", f"{mean:.2f} vs {check['target']}±{check['tol']}")
        return 3
    return 0


# --------------------------------------------------------------------------- E026 (window x width map)

def e026_grid(config: dict) -> list[tuple[str, dict, str]]:
    from .model_zoo import cost

    out = []
    g = config["grid"]
    for w in g["windows"]:
        for h in g["hidden"]:
            spec = {"model": "complex_one_hidden", "window": w, "hidden": h}
            if g["rm_min"] <= cost(spec).macs_merged <= g["rm_max"]:
                out.append((f"oh_w{w}_h{h}", spec, "F2"))
    b = config["both"]
    for w in b["windows"]:
        for h in b["hidden"]:
            out.append((f"both_w{w}_h{h}", {"model": "complex_both", "window": w, "hidden": h}, "F4"))
    return out


def e026_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E026"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage not in ("grid", "confirm"):
        raise SystemExit(f"E026: неизвестная стадия {stage!r}")
    grid = e026_grid(config)
    if options.smoke:                # smoke: every 6th grid point
        grid = grid[::6]

    def tasks_for(items, n_seeds, steps, stage_name):
        out = []
        for name, spec, key in items:
            lr, overrides, source = recipe_lr(key, options)
            for seed in first_seeds(config, options, n_seeds):
                out.append(run(exp, stage_name, directory, config, options, name=name, spec=dict(spec, **overrides),
                               lr=lr, steps=steps, seed=seed, meta={"recipe": key, "recipe_source": source,
                                                                   "rm": None}))
        return out

    first = tasks_for(grid, config["grid"]["seeds"], config["grid"]["steps"], "grid")
    if stage == "grid":
        return first
    from .model_zoo import cost

    runs = read_runs(directory, options.smoke)
    complete = require(first, runs, "E026 grid")
    stats = group_val(runs, "variant", None if options.smoke else config["grid"]["steps"]) if complete else {}
    points = [(name, cost(spec).macs_merged, stats.get(name, (float("nan"), 0, True))) for name, spec, _ in grid]
    front = pareto_front([(n, m, v[0]) for n, m, v in points if not v[2] and math.isfinite(v[0])])
    if not front:
        front = [(n, m, 0.0) for n, m, _ in points]
    chosen = thin_by_log(front, config["confirm"]["n_points"])
    save_json(directory, options, "confirm_selection.json", [{"name": n, "rm": m, "val_mean": v} for n, m, v in chosen])
    names = {n for n, _, _ in chosen}
    return tasks_for([g for g in grid if g[0] in names], config["confirm"]["seeds"], config["confirm"]["steps"], "confirm")


def pareto_front(points: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """Points not dominated in (lower RM, higher value), sorted by RM."""
    front, best = [], -math.inf
    for name, rm, value in sorted(points, key=lambda p: (p[1], -p[2])):
        if value > best:
            front.append((name, rm, value))
            best = value
    return front


def thin_by_log(front: list, n: int) -> list:
    if len(front) <= n:
        return front
    logs = np.log([max(p[1], 1) for p in front])
    targets = np.linspace(logs.min(), logs.max(), n)
    picked = sorted({int(np.argmin(np.abs(logs - t))) for t in targets})
    return [front[i] for i in picked]


# --------------------------------------------------------------------------- E027 (activation ablation)

def e027_lr_items(config: dict, activation: str) -> list[tuple[str, dict]]:
    items = []
    c_mults = config["kerr_c_mult"] if activation in config["c_mult_activations"] else [1]
    for item in config["lr_grid"]:
        for c_mult in c_mults:
            lr = lr_from(config["base_lr"], **{k: v for k, v in item.items() if k != "name"}, c_mult=c_mult)
            items.append((f"{item['name']}_cm{c_mult:g}", lr))
    return items


def final_c(directory: Path, smoke: bool, run_id: str) -> float:
    import pandas as pd

    path = results_dir(directory, smoke) / "curves" / f"{run_id}.csv"
    if not path.is_file():
        return float("nan")
    c = pd.read_csv(path).c1.dropna()
    return float(c.iloc[-1]) if len(c) else float("nan")


def e027_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E027"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    stages = ("s1", "s1_fixed", "s2", "s2_fixed", "s3", "s3_fixed")
    if stage not in stages:
        raise SystemExit(f"E027: неизвестная стадия {stage!r}")
    acts = config["activations"]
    fixed = "kerr_c_fixed"
    base = stage.split("_")[0]
    runs = read_runs(directory, options.smoke) if stage != "s1" else []

    def spec(act, hidden, c=None):
        s = {"model": "act_mlp", "window": config["window"], "hidden": hidden, "activation": act}
        if act == fixed:
            s["c_fixed"] = c
        return s

    def make(act, lr_id, lr, hidden, n_seeds, steps, stage_name, c=None):
        name = f"{act}_h{hidden}"
        return [run(exp, stage_name, directory, config, options, name=name, spec=spec(act, hidden, c), lr=lr,
                    lr_name=lr_id, steps=steps, seed=seed,
                    meta={"activation": act, "lr_id": lr_id, "e027_stage": base, "c_fixed": c})
                for seed in first_seeds(config, options, n_seeds)]

    def c_median(hidden, steps, lr_id=None):
        """Median final c of kerr_c_train runs at this width and step budget (same lr name without cm if given)."""
        values = [final_c(directory, options.smoke, r["run_id"]) for r in runs
                  if r.get("activation") == "kerr_c_train" and int(r["spec"]["hidden"]) == hidden
                  and (options.smoke or int(r["budget"]) == steps)
                  and (lr_id is None or r["lr_id"].rsplit("_cm", 1)[0] == lr_id.rsplit("_cm", 1)[0])]
        values = [v for v in values if math.isfinite(v)]
        if values:
            return float(np.median(values))
        if placeholder() or options.smoke:
            return float(config["c_fixed_fallback"])
        raise SystemExit(f"E027 {stage}: нет обученных c у kerr_c_train (h={hidden}, {steps} шагов)")

    s1 = config["stages"]["s1"]
    if stage in ("s1", "s1_fixed"):
        tasks = []
        for act in acts:
            if (act == fixed) != (stage == "s1_fixed"):
                continue
            for lr_id, lr in e027_lr_items(config, act):
                c = c_median(s1["hidden"], s1["steps"], lr_id) if act == fixed else None
                tasks += make(act, lr_id, lr, s1["hidden"], s1["seeds"], s1["steps"], stage, c)
        return tasks

    def top_recipes(act, n):
        lr_items = dict(e027_lr_items(config, act))
        mine = [r for r in runs if r.get("activation") == act and r.get("e027_stage") == "s1"]
        have = {r["lr_id"] for r in mine}
        if not placeholder() and not options.smoke and set(lr_items) - have:
            raise SystemExit(f"E027 {stage}: s1 не завершена для {act}")
        stats = group_val(mine, "lr_id")
        return [(k, lr_items[k]) for k in best_keys(stats, n, list(lr_items))]

    s2 = config["stages"]["s2"]
    if base == "s2":
        tasks = []
        for act in acts:
            if (act == fixed) != (stage == "s2_fixed"):
                continue
            for lr_id, lr in top_recipes(act, s2["top"]):
                for hidden in s2["hidden"]:
                    c = c_median(hidden, s2["steps"]) if act == fixed else None
                    tasks += make(act, lr_id, lr, hidden, s2["seeds"], s2["steps"], stage, c)
        return tasks
    s3 = config["stages"]["s3"]
    tasks = []
    for act in acts:
        if (act == fixed) != (stage == "s3_fixed"):
            continue
        candidates = top_recipes(act, s2["top"])
        mine = [r for r in runs if r.get("activation") == act and r.get("e027_stage") == "s2"
                and int(r["spec"]["hidden"]) == s3["hidden"]]
        stats = group_val(mine, "lr_id")
        best = best_keys({k: v for k, v in stats.items() if k in dict(candidates)}, 1, [k for k, _ in candidates])[0]
        c = c_median(s3["hidden"], s3["steps"]) if act == fixed else None
        tasks += make(act, best, dict(candidates)[best], s3["hidden"], s3["seeds"], s3["steps"], stage, c)
    return tasks


# --------------------------------------------------------------------------- E028 (DBP depth, layer-wise diagnosis)

def dbp_spec(n_steps: int, fir_len: int, n_alpha: int, model: str = "phys_dbp", **extra) -> dict:
    return phys_spec(31, fir_len, n_alpha, True, model=model, n_steps=n_steps,
                     name=f"dbp_Ns{n_steps}_L{fir_len}_R{n_alpha}", **extra)


def e028_grid(config: dict) -> list[tuple[str, dict]]:
    g, out = config["grid_a"], []
    for n in g["n_steps"]:
        for fir in g["fir_len"]:
            for r in g["n_alpha"]:
                out.append((f"dbp_Ns{n}_L{fir}_R{r}", dbp_spec(n, fir, r)))
    for n in g["extra_n_steps"]:
        name = f"dbp_Ns{n}_L{g['extra_fir_len']}_R{g['extra_n_alpha']}"
        if name not in dict(out):
            out.append((name, dbp_spec(n, g["extra_fir_len"], g["extra_n_alpha"])))
    return out


def e028_b_variants(config: dict, options: Options) -> list[tuple[str, dict, list[dict], dict]]:
    """(name, spec, phases, lr) of the layer-wise diagnosis; all at Ns = 8, L = 31, R = 7, same total steps."""
    b = config["b"]
    ns, per, fine, div = b["n_steps"], b["steps_per_layer"], b["finetune_steps"], b["finetune_lr_divisor"]
    base = lr_from(config["e018_lr"])
    spec = dbp_spec(ns, b["fir_len"], b["n_alpha"])
    noisy = dbp_spec(ns, b["fir_len"], b["n_alpha"], model="dbp_noise", init_noise=b["init_noise"])
    fine_lr = lr_from(base, peak_lr=base["peak_lr"] / div)
    finetune = [{"steps": fine, "trainable": "all", "reset_optimizer": True, "lr": fine_lr}]

    def cumulative(lr=None):
        return [dict({"steps": per, "trainable": [f"steps.{j}." for j in range(k + 1)], "reset_optimizer": True},
                     **({"lr": lr} if lr else {})) for k in range(ns)]

    rewarm = lr_from(base, schedule="cosine_to_zero", warmup=b["rewarm_warmup"])
    layerwise_rewarm = [dict(p, lr=rewarm) for p in e018_phases("layerwise", ns, per, fine, base, div)[:-1]]
    half = (ns * per + fine) // 2
    pre = [{"steps": half // ns, "trainable": [f"steps.{k}."], "reset_optimizer": True} for k in range(ns)]
    joint_after = [{"steps": ns * per + fine - (half // ns) * ns, "trainable": "all", "reset_optimizer": True}]
    recipe, _, _ = recipe_lr("F6", options)
    recipe = lr_from(recipe, horizon=None)
    return [
        ("b_a_e018_layerwise", spec, e018_phases("layerwise", ns, per, fine, base, div), base),
        ("b_b_identity_all_trainable", spec, cumulative() + finetune, base),
        ("b_c_small_random_all_trainable", noisy, cumulative() + finetune, base),
        ("b_d_lr_rewarm_per_stage", spec, layerwise_rewarm + finetune, base),
        ("b_e_layerwise_then_joint", spec, pre + joint_after, base),
        ("b_f_joint_from_scratch", spec, [{"steps": ns * per + fine}], base),
        ("b_g_identity_all_trainable_recipe", spec,
         cumulative(recipe) + [dict(finetune[0], lr=lr_from(recipe, peak_lr=recipe["peak_lr"] / div))], recipe),
    ]


def e028_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E028"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage not in ("a", "a_long", "shared", "hybrid", "b"):
        raise SystemExit(f"E028: неизвестная стадия {stage!r}")

    def dbp_recipe(n_steps):
        return recipe_lr("F6" if n_steps <= 8 else "F7", options)

    def tasks_for(items, n_seeds, steps, stage_name, key=None):
        out = []
        for name, spec in items:
            lr, _, source = recipe_lr(key, options) if key else dbp_recipe(int(spec["n_steps"]))
            for seed in first_seeds(config, options, n_seeds):
                out.append(run(exp, stage_name, directory, config, options, name=name, spec=spec, lr=lr, steps=steps,
                               seed=seed, meta={"recipe_source": source, "e028_stage": stage_name}))
        return out

    a = config["grid_a"]
    grid = e028_grid(config)
    first = tasks_for(grid, a["seeds"], a["steps"], "a")
    if stage == "a":
        return first
    if stage == "a_long":
        runs = read_runs(directory, options.smoke)
        complete = require(first, runs, "E028 a")
        stats = group_val([r for r in runs if r.get("e028_stage") == "a"], "variant",
                          None if options.smoke else a["steps"]) if complete else {}
        chosen = best_keys(stats, config["a_long"]["top"], [n for n, _ in grid])
        save_json(directory, options, "a_long_selection.json", chosen)
        return tasks_for([g for g in grid if g[0] in chosen], config["a_long"]["seeds"], config["a_long"]["steps"],
                         "a_long")
    if stage == "shared":
        s = config["shared"]
        items = [(f"dbps_Ns{n}_L{s['fir_len']}_R{s['n_alpha']}",
                  dbp_spec(n, s["fir_len"], s["n_alpha"], model="dbp_shared")) for n in s["n_steps"]]
        return tasks_for(items, s["seeds"], s["steps"], "shared")
    if stage == "hybrid":
        h = config["hybrid"]
        items = [(f"dbpmlp_Ns{n}_h{hid}", dbp_spec(n, h["fir_len"], h["n_alpha"], model="dbp_mlp", mlp_hidden=hid))
                 for n in h["n_steps"] for hid in h["mlp_hidden"]]
        return tasks_for(items, h["seeds"], h["steps"], "hybrid", key="F8")
    b = config["b"]
    out = []
    for name, spec, phases, lr in e028_b_variants(config, options):
        total = sum(int(p["steps"]) for p in phases)
        for seed in first_seeds(config, options, b["seeds"]):
            task = make_task(exp, "b", directory, options, spec=dict(spec, name=name), seed=seed,
                             n_train=config["n_train_files"], phases=phases, lr=lr, lr_name=lr_label(lr),
                             eval_every=b["eval_every"], select_every=b["eval_every"], variant=name,
                             run_id=f"{name}_{total // 1000}k_seed{seed}", meta={"e028_stage": "b"},
                             force_checkpoints=True, ckpt_every=config.get("ckpt_every", 10_000))
            out.append(task)
    return out


# --------------------------------------------------------------------------- E029 (physics front-end vs width)

def nearest_recipe(hidden: int, table: dict) -> str:
    return min(table, key=lambda key: (abs(math.log(hidden) - math.log(table[key])), table[key]))


def e029_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E029"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage != "main":
        raise SystemExit(f"E029: неизвестная стадия {stage!r}")
    items = []
    for hidden in config["hidden"]:
        items.append((f"pure_h{hidden}", {"model": "complex_one_hidden", "window": 31, "hidden": hidden},
                      nearest_recipe(hidden, config["pure_recipes"]), "pure", hidden))
        fe = config["front_end"]
        items.append((f"hyb_R{fe['n_alpha']}_L{fe['fir_len']}_h{hidden}",
                      phys_spec(31, fe["fir_len"], fe["n_alpha"], True, model="phys_plus_mlp", mlp_hidden=hidden),
                      nearest_recipe(hidden, config["hybrid_recipes"]), "hybrid", hidden))
    for fe in config["front_end_variants"]:
        items.append((f"hyb_R{fe['n_alpha']}_L{fe['fir_len']}_h{config['variant_hidden']}",
                      phys_spec(31, fe["fir_len"], fe["n_alpha"], True, model="phys_plus_mlp",
                                mlp_hidden=config["variant_hidden"]), "F9", "hybrid", config["variant_hidden"]))
    tasks, seen = [], set()
    for name, spec, key, arm, hidden in items:
        if name in seen:
            continue
        seen.add(name)
        lr, _, source = recipe_lr(key, options)
        for seed in first_seeds(config, options, config["n_seeds"]):
            tasks.append(run(exp, stage, directory, config, options, name=name, spec=spec, lr=lr, steps=config["steps"],
                             seed=seed, meta={"arm": arm, "width": hidden, "recipe": key, "recipe_source": source}))
    return tasks


# --------------------------------------------------------------------------- E030 (loss functions)

def e030_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E030"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage not in ("s1", "s2"):
        raise SystemExit(f"E030: неизвестная стадия {stage!r}")
    s1 = config["stages"]["s1"]

    def make(model_key, loss, scale, n_seeds, steps, stage_name):
        item = config["models"][model_key]
        lr, _, source = recipe_lr(item["recipe"], options, scale)
        name = f"{model_key}_{loss.replace('+', 'p')}"
        return [run(exp, stage_name, directory, config, options, name=name, spec=item["spec"], lr=lr,
                    lr_name=f"x{scale:g}", steps=steps, seed=seed, loss=loss,
                    meta={"model_key": model_key, "loss_name": loss, "lr_scale": scale, "recipe_source": source})
                for seed in first_seeds(config, options, n_seeds)]

    first = [t for m in config["models"] for loss in config["losses"] for scale in s1["lr_scales"]
             for t in make(m, loss, scale, s1["seeds"], s1["steps"], "s1")]
    if stage == "s1":
        return first
    runs = read_runs(directory, options.smoke)
    complete = require(first, runs, "E030 s1")
    s2, tasks, selection = config["stages"]["s2"], [], {}
    for model_key in config["models"]:
        for loss in config["losses"]:
            mine = [r for r in runs if r.get("model_key") == model_key and r.get("loss_name") == loss
                    and r["stage"] == "s1"]
            stats = group_val(mine, "lr_scale") if complete else {}
            scale = best_keys(stats, 1, [1.0])[0]
            selection[f"{model_key}_{loss}"] = scale
            tasks += make(model_key, loss, float(scale), s2["seeds"], s2["steps"], "s2")
    save_json(directory, options, "s2_selection.json", selection)
    return tasks


# --------------------------------------------------------------------------- E032 (1M steps)

def e032_members(config: dict, options: Options) -> list[dict]:
    from .recipes import load_all

    members = [dict(m) for m in config["members"]]
    recipes = load_all(options.smoke)
    real = config["real_candidates"]
    vals = {k: recipes.get(k, {}).get("val_mean", -math.inf) for k in real}
    real_key = max(vals, key=vals.get) if any(math.isfinite(v) for v in vals.values()) else real[-1]
    members.append({"name": f"real_{real_key}", "recipe": real_key, "spec": config["real_specs"][real_key]})
    dbp = dict(config["dbp_default"])
    summary = Path(load_config("E028")[0]) / "results" / "a_summary.csv"
    if summary.is_file():
        import pandas as pd

        frame = pd.read_csv(summary)
        frame = frame[frame.stage.isin(["a", "a_long"])]
        if len(frame):
            best = frame.loc[frame.val_mean.idxmax()]
            dbp = {"name": f"dbp_e028_{best.variant}", "recipe": "F6" if int(best.n_steps) <= 8 else "F7",
                   "spec": dbp_spec(int(best.n_steps), int(best.fir_len), int(best.n_alpha))}
    members.append(dbp)
    return members


def e032_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E032"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage != "main":
        raise SystemExit(f"E032: неизвестная стадия {stage!r}")
    tasks = []
    for item in e032_members(config, options):
        lr, overrides, source = recipe_lr(item["recipe"], options)
        for seed in first_seeds(config, options, config["n_seeds"]):
            tasks.append(run(exp, stage, directory, config, options, name=item["name"], spec=dict(item["spec"], **overrides),
                             lr=lr, steps=config["steps"], seed=seed, eval_every=config["eval_every"],
                             meta={"recipe": item["recipe"], "recipe_source": source}))
    return tasks


# --------------------------------------------------------------------------- E033 (fill missing Pareto levels)

def e033_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E033"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage != "fill":
        raise SystemExit(f"E033: неизвестная стадия {stage!r}")
    from .pareto_v2 import missing_levels

    items = missing_levels(config, options.smoke)
    save_json(directory, options, "fill_selection.json", items)
    tasks = []
    for item in items:
        lr, overrides, source = recipe_lr(item["recipe"], options)
        for seed in first_seeds(config, options, config["fill"]["seeds"]):
            tasks.append(run(exp, stage, directory, config, options, name=item["name"], spec=dict(item["spec"], **overrides),
                             lr=lr, steps=config["fill"]["steps"], seed=seed,
                             meta={"family": item["family"], "level": item["level"], "recipe": item["recipe"],
                                   "recipe_source": source}))
    return tasks


BUILDERS = {"E024": e024_tasks, "E025": e025_tasks, "E026": e026_tasks, "E027": e027_tasks, "E028": e028_tasks,
            "E029": e029_tasks, "E030": e030_tasks, "E032": e032_tasks, "E033": e033_tasks}
SPECIAL_STAGES = {("E025", "check-code"): e025_check_code}
POST_HOOKS = {"dedup_eval": dedup_eval_hook}


# --------------------------------------------------------------------------- E035 (triplets by gradient, + Kerr MLP)

def e035_warm_path(directory: Path, smoke: bool, M: int) -> str:
    from .paths import PROJECT_ROOT

    return str((results_dir(directory, smoke) / f"warm_M{M}.npz").relative_to(PROJECT_ROOT))


def e035_fit(options: Options) -> int:
    """Closed-form T3 + FIR coefficients on all train files (lambda by validation) -> results/warm_M<M>.npz."""
    from .runner import atomic_write_json, default_files, log
    from .triplets_ext import closed_form_coefficients
    from .window_data import load_fileset, resolve_paths

    directory, config = load_config("E035")
    files = default_files(config["n_train_files"], options.smoke)
    train, val = (load_fileset(resolve_paths(files[k])) for k in ("train", "validation"))
    out = results_dir(directory, options.smoke)
    out.mkdir(parents=True, exist_ok=True)
    info = {}
    for M in sorted({c["M"] for c in config["configs"].values()}):
        coef, lam = closed_form_coefficients(train, val, M, config["fir_len"], config["lambda_grid"], 4096)
        np.savez(out / f"warm_M{M}.npz", coef=coef)
        info[f"M{M}"] = {"lambda": lam, "n_coefficients": int(len(coef))}
        log(f"E035 warm start M={M}: {len(coef)} coefficients, lambda {lam:g}")
    atomic_write_json(out / "warm_info.json", info)
    return 0


def e035_spec(directory: Path, config: dict, options: Options, item: dict) -> dict:
    spec = {"model": "triplet_net", "window": item["window"], "M": item["M"], "fir_len": config["fir_len"],
            "mlp_hidden": item.get("mlp_hidden", 0)}
    if item.get("warm"):
        spec["warm"] = e035_warm_path(directory, options.smoke, item["M"])
    return spec


def e035_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E035"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage not in ("s1", "s2"):
        raise SystemExit(f"E035: неизвестная стадия {stage!r}")
    s1 = config["stages"]["s1"]

    def make(name, item, lr_item, n_seeds, steps, stage_name):
        lr = lr_from(config["base_lr"], **{k: v for k, v in lr_item.items() if k != "name"})
        return [run(exp, stage_name, directory, config, options, name=name, spec=e035_spec(directory, config, options, item),
                    lr=lr, lr_name=lr_item["name"], steps=steps, seed=seed,
                    meta={"e035_config": name, "lr_id": lr_item["name"], "recipe_source": "E035 s1 (validation)"})
                for seed in first_seeds(config, options, n_seeds)]

    first = [t for name, item in config["configs"].items() for lr_item in config["lr_grid"]
             for t in make(name, item, lr_item, s1["seeds"], s1["steps"], "s1")]
    if stage == "s1":
        return first
    runs = read_runs(directory, options.smoke)
    complete = require(first, runs, "E035 s1")
    s2, tasks, selection = config["stages"]["s2"], [], {}
    by_lr = {i["name"]: i for i in config["lr_grid"]}
    for name, item in config["configs"].items():
        stats = group_val([r for r in runs if r.get("e035_config") == name and r["stage"] == "s1"], "lr_id") if complete else {}
        best = best_keys(stats, 1, [config["lr_grid"][0]["name"]])[0]
        selection[name] = best
        tasks += make(name, item, by_lr[best], s2["seeds"], s2["steps"], "s2")
    save_json(directory, options, "s2_selection.json", selection)
    return tasks


BUILDERS["E035"] = e035_tasks
SPECIAL_STAGES[("E035", "fit")] = e035_fit


def _register() -> None:
    from . import grids

    grids.BUILDERS.update(BUILDERS)
    grids.SPECIAL_STAGES.update(SPECIAL_STAGES)
    grids.POST_HOOKS.update(POST_HOOKS)


_register()
