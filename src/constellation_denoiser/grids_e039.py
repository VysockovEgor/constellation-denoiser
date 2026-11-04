"""Task lists for E040-E042 (registered into grids.BUILDERS by grids.py, like grids_e024)."""

from __future__ import annotations

import json
import math

import numpy as np

from .cascade_e040 import cascade_rm, window_for
from .grids import Options, control_tasks, load_config, read_runs
from . import grids_e024 as g24   # module reference: grids_e024 may still be initializing when this is imported


def k_th_of(rule: str, M: int) -> int:
    return {"M2/8": M * M // 8, "M2/4": M * M // 4, "M2/2": M * M // 2, "M2": M * M}[rule]


def cascade_spec(stages: list[dict], shared: bool = False) -> dict:
    return {"model": "cascade", "window": window_for(stages), "stages": stages, "shared": shared}


def in_level(rm: float, level: float, tol: float) -> bool:
    return abs(rm / level - 1) <= tol


# --------------------------------------------------------------------------- E040

def e040_grid(config: dict) -> list[tuple[str, dict]]:
    g, out = config["grid"], []
    for ns in g["n_stages"]:
        for M in g["M"]:
            for rule in g["K_th"]:
                for L in g["L"]:
                    stage = {"type": "T", "M": M, "K_th": k_th_of(rule, M), "L": L}
                    stages = [dict(stage) for _ in range(ns)]
                    rm, w = cascade_rm(stages), window_for(stages)
                    if not (g["rm_min"] <= rm <= g["rm_max"]) or w > g["max_window"]:
                        continue
                    name = f"c_Ns{ns}_M{M}_K{stage['K_th']}_L{L}"
                    out.append((name, cascade_spec(stages)))
                    if ns > 1:
                        out.append((name + "_shared", cascade_spec(stages, shared=True)))
    names = {n for n, _ in out}
    lm = config.get("level_matched")
    for level in (config["levels"] if lm else []):     # points with an integer K_th that puts RM within the level window
        for ns in lm["n_stages"]:
            for M in lm["M"]:
                for L in lm["L"]:
                    best = None
                    for k in range(1, M * M + 1):
                        stages = [{"type": "T", "M": M, "K_th": k, "L": L} for _ in range(ns)]
                        rm = cascade_rm(stages)
                        if in_level(rm, level, config["level_tol"]) and window_for(stages) <= g["max_window"]:
                            if best is None or abs(rm - level) < abs(cascade_rm(best) - level):
                                best = stages
                    if best is not None:
                        name = f"c_Ns{ns}_M{M}_K{best[0]['K_th']}_L{L}"
                        if name not in names:
                            names.add(name)
                            out.append((name, cascade_spec(best)))
                            if ns > 1:
                                out.append((name + "_shared", cascade_spec(best, shared=True)))
    return out


def e040_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E040"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage not in ("main", "confirm"):
        raise SystemExit(f"E040: неизвестная стадия {stage!r}")
    grid = e040_grid(config)
    if options.smoke:
        grid = [g for g in grid if g[1]["window"] <= 41][:6]

    def make(items, n_seeds, steps, stage_name):
        return [g24.run(exp, stage_name, directory, config, options, name=name, spec=spec, lr=config["lr"], lr_name="cos1e-3",
                    steps=steps, seed=seed, meta={"rm_repo": cascade_rm(spec["stages"]), "shared": spec["shared"],
                                                  "n_stages": len(spec["stages"]), "recipe_source": "E040 fixed lr"})
                for name, spec in items for seed in g24.first_seeds(config, options, n_seeds)]

    first = make(grid, config["main"]["seeds"], config["main"]["steps"], "main")
    if stage == "main":
        return first
    runs = read_runs(directory, options.smoke)
    complete = g24.require(first, runs, "E040 main")
    stats = g24.group_val([r for r in runs if r["stage"] == "main"], "variant") if complete else {}
    by_name = dict(grid)
    chosen, selection = [], {}
    for level in config["levels"]:
        near = {k: v for k, v in stats.items() if in_level(cascade_rm(by_name[k]["stages"]), level, config["level_tol"])}
        if not near and g24.placeholder():
            near = {n: (0.0, 1, False) for n, s in grid if in_level(cascade_rm(s["stages"]), level, 0.3)}
        top = g24.best_keys(near, config["confirm"]["top_per_level"], [])
        selection[str(level)] = top
        for name in top:
            twin = name[:-7] if name.endswith("_shared") else name + "_shared"
            for item in (name, twin):
                if item in by_name and item not in chosen:
                    chosen.append(item)
    g24.save_json(directory, options, "confirm_selection.json", selection)
    return make([(n, by_name[n]) for n in chosen], config["confirm"]["seeds"], config["confirm"]["steps"], "confirm")


# --------------------------------------------------------------------------- E041

def e041_sample(config: dict, arm: str, level: int) -> list[tuple[str, dict]]:
    """Up to n_configs distinct random cascades of one arm with RM within level +- tol (rejection sampling)."""
    space = config["spaces"][arm]
    rng = np.random.default_rng(config["sample_seed"] + level + (0 if arm == "mixed" else 1))
    seen, out = set(), []
    for _ in range(config["max_draws"]):
        ns = int(rng.integers(space["n_stages"][0], space["n_stages"][1] + 1))
        if arm == "uniform":
            M = int(rng.choice(space["M"]))
            k_th = int(rng.integers(1, M * M + 1)) if space["K_th"] == "any" else k_th_of(str(rng.choice(space["K_th"])), M)
            st = {"type": "T", "M": M, "K_th": k_th, "L": int(rng.choice(space["L"]))}
            stages = [dict(st) for _ in range(ns)]
        else:
            stages = []
            for _ in range(ns):
                kind = str(rng.choice(space["types"]))
                s = {"type": kind}
                if kind != "skip":
                    s["L"] = int(rng.choice(space["L"]))
                if kind in ("T", "KT"):
                    s["M"] = int(rng.choice(space["M"]))
                    s["K_th"] = k_th_of(str(rng.choice(space["K_th"])), s["M"])
                if kind in ("K", "KT"):
                    s["R"] = int(rng.choice(space["R"]))
                stages.append(s)
            if all(s["type"] == "skip" for s in stages):
                continue
        key = json.dumps(stages, sort_keys=True)
        if key in seen:
            continue
        rm = cascade_rm(stages)
        if not in_level(rm, level, config["level_tol"]) or window_for([s for s in stages if s["type"] != "skip"]) > config["max_window"]:
            continue
        seen.add(key)
        out.append((f"{arm[0]}{level // 1000}k_{len(out):02d}", cascade_spec(stages)))
        if len(out) >= config["n_configs"]:
            break
    return out


def e041_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E041"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage not in ("s1", "s2", "s3"):
        raise SystemExit(f"E041: неизвестная стадия {stage!r}")
    plan = config["stages"]

    def make(items, n_seeds, steps, stage_name, arm, level):
        return [g24.run(exp, stage_name, directory, config, options, name=name, spec=spec, lr=config["lr"], lr_name="cos1e-3",
                    steps=steps, seed=seed, meta={"arm": arm, "level": level, "rm_repo": cascade_rm(spec["stages"]),
                                                  "e041_config": name, "recipe_source": "E041 fixed lr"})
                for name, spec in items for seed in g24.first_seeds(config, options, n_seeds)]

    tasks, selection = [], {}
    runs = read_runs(directory, options.smoke) if stage != "s1" else []
    for level in config["levels"]:
        for arm in config["arms"]:
            items = e041_sample(config, arm, level)
            if options.smoke:
                items = [i for i in items if i[1]["window"] <= 61][:2]
            first = make(items, plan["s1"]["seeds"], plan["s1"]["steps"], "s1", arm, level)
            if stage == "s1":
                tasks += first
                continue
            by = dict(items)
            mine = [r for r in runs if r.get("arm") == arm and r.get("level") == level]
            ok = g24.require(first, mine, f"E041 s1 {arm} {level}")
            s1 = g24.group_val([r for r in mine if r["stage"] == "s1"], "e041_config") if ok else {}
            top4 = g24.best_keys(s1, plan["s2"]["top"], list(by))
            if stage == "s2":
                selection[f"{arm}_{level}"] = top4
                tasks += make([(n, by[n]) for n in top4], plan["s2"]["seeds"], plan["s2"]["steps"], "s2", arm, level)
                continue
            second = make([(n, by[n]) for n in top4], plan["s2"]["seeds"], plan["s2"]["steps"], "s2", arm, level)
            ok = ok and g24.require(second, mine, f"E041 s2 {arm} {level}")
            s2 = {k: v for k, v in (g24.group_val([r for r in mine if r["stage"] == "s2"], "e041_config") if ok else {}).items()
                  if k in top4}
            top2 = g24.best_keys(s2, plan["s3"]["top"], top4)
            selection[f"{arm}_{level}"] = top2
            tasks += make([(n, by[n]) for n in top2], plan["s3"]["seeds"], plan["s3"]["steps"], "s3", arm, level)
    if stage != "s1":
        g24.save_json(directory, options, f"{stage}_selection.json", selection)
    return tasks


BUILDERS = {"E040": e040_tasks, "E041": e041_tasks}


def _register() -> None:
    from . import grids

    grids.BUILDERS.update(BUILDERS)


_register()
