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


# --------------------------------------------------------------------------- E047 (cascade variants, one factor at a time)

def e047_init(options: Options) -> int:
    """Ridge solution of the first stage (T terms on x, target s - x, all train files, lambda by validation)."""
    from .cascade_e040 import stage_terms
    from .paths import PROJECT_ROOT
    from .runner import atomic_write_json, default_files, log
    from .triplets import EDGE, FileGram, ber_columns, predict, solve
    from .window_data import load_fileset, resolve_paths

    directory, config = load_config("E047")
    st = config["base_stage"]
    terms = stage_terms(st["M"], st["K_th"])
    files = default_files(30, options.smoke)
    train, val = (load_fileset(resolve_paths(files[k])) for k in ("train", "validation"))
    gram = rhs = None
    for f in range(train.n_files):
        g = FileGram(train.rx_c[f], train.tx_c[f], terms, 0, False, 4096)
        gram = g.gram if gram is None else gram + g.gram
        rhs = g.rhs if rhs is None else rhs + g.rhs
    lams = config["lambda_grid"]
    coefs = np.stack([solve(gram, rhs, lam) for lam in lams], axis=1)
    val_tx = np.concatenate([val.tx_c[f][EDGE:-EDGE] for f in range(val.n_files)])
    pv = np.concatenate([predict(val.rx_c[f], coefs, terms, 0, False, 4096) for f in range(val.n_files)])
    best = int(np.argmin(ber_columns(pv, val_tx)))
    out = g24.results_dir(directory, options.smoke)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "init_stage0.npz", weights=coefs[:, best])
    atomic_write_json(out / "init_stage0.json", {"lambda": lams[best], "n_terms": len(terms)})
    log(f"E047: ridge init of stage 0, {len(terms)} terms, lambda {lams[best]:g}")
    return 0


def e047_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E047"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage != "main":
        raise SystemExit(f"E047: неизвестная стадия {stage!r}")
    from .paths import PROJECT_ROOT

    stages = [dict(config["base_stage"]) for _ in range(config["n_stages"])]
    init = str((g24.results_dir(directory, options.smoke) / "init_stage0.npz").relative_to(PROJECT_ROOT))
    steps = config["steps"]
    half = steps // 2
    variants = {
        "v1_independent_joint": ({}, None),
        "v2_shared_joint": ({"shared": True}, None),
        "v3_stagewise_freeze": ({}, [{"steps": half, "trainable": ["stages.0."]},
                                     {"steps": steps - half, "trainable": ["stages.1."], "reset_optimizer": True}]),
        "v4_ridge_init_stage0": ({"init_first": init}, None),
        "v5_free_no_symmetry": ({"free": True}, None),
    }
    tasks = []
    for name, (extra, phases) in variants.items():
        spec = dict(cascade_spec(stages), **extra)
        for seed in g24.first_seeds(config, options, config["n_seeds"]):
            tasks.append(g24.run(exp, stage, directory, config, options, name=name, spec=spec, lr=config["lr"],
                                 lr_name="cos1e-3", steps=steps, seed=seed, phases=phases,
                                 meta={"variant_factor": name, "rm_repo": cascade_rm(stages)}))
    return tasks


BUILDERS["E047"] = e047_tasks


# --------------------------------------------------------------------------- E048 (DBP + off-axis triplets)

def e048_select(options: Options) -> int:
    """Off-axis T3 terms (m != 0, n != 0) of the closed-form M fit, strongest first -> results/terms_offaxis.npz."""
    from .runner import atomic_write_json, default_files, log
    from .triplets_ext import rank_terms
    from .window_data import load_fileset, resolve_paths

    directory, config = load_config("E048")
    files = default_files(30, options.smoke)
    train, val = (load_fileset(resolve_paths(files[k])) for k in ("train", "validation"))
    ranked = rank_terms(train, val, config["select_M"], config["select_fir_len"], config["lambda_grid"], 4096)
    off = [t for t in ranked if t[0] != 0 and t[1] != 0][: max(config["dbp_k"])]
    out = g24.results_dir(directory, options.smoke)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "terms_offaxis.npz", terms=np.array(off))
    atomic_write_json(out / "terms_offaxis.json", {"M": config["select_M"], "top": [list(t) for t in off]})
    log(f"E048: {len(off)} внеосевых членов сохранены")
    return 0


def e048_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E048"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage != "main":
        raise SystemExit(f"E048: неизвестная стадия {stage!r}")
    from .paths import PROJECT_ROOT

    terms = str((g24.results_dir(directory, options.smoke) / "terms_offaxis.npz").relative_to(PROJECT_ROOT))
    tasks = []
    for ns, key in config["dbp"].items():
        lr, _, source = g24.recipe_lr(key, options)
        for k in [0] + list(config["dbp_k"]):
            spec = dict(config["dbp_spec"], model="dbp_triplets", window=31, n_steps=int(ns), k=k, terms=terms if k else None)
            for seed in g24.first_seeds(config, options, config["n_seeds"]):
                tasks.append(g24.run(exp, stage, directory, config, options, name=f"dbpoff_Ns{ns}_K{k}", spec=spec, lr=lr,
                                     steps=config["steps"], seed=seed,
                                     meta={"n_steps": int(ns), "k": k, "recipe": key, "recipe_source": source}))
    return tasks


BUILDERS["E048"] = e048_tasks


# --------------------------------------------------------------------------- E049 (DBP: physical vs identity initialization)

def e049_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E049"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage != "main":
        raise SystemExit(f"E049: неизвестная стадия {stage!r}")
    calib = g24.results_dir(directory, False) / "calib_summary.json"
    if calib.is_file():
        sign = json.loads(calib.read_text())["sign_of_learned_c"] or config["sign_fallback"]
    elif options.smoke or g24.placeholder():
        sign = config["sign_fallback"]
    else:
        raise SystemExit("E049: нет results/calib_summary.json: сначала стадия calib")
    alpha = config["alpha_centre_total"]
    tasks = []
    for item in config["configs"]:
        base = {"window": 31, "fir_len": item["fir_len"], "n_alpha": item["n_alpha"], "sym": True, "n_steps": item["n_steps"]}
        lr, _, source = g24.recipe_lr(item["recipe"], options)
        arms = {"delta": dict(base, model="phys_dbp"),
                "phys_fir": dict(base, model="dbp_phys", c_sign=sign, fir_init="physical"),
                "phys_fir_alpha": dict(base, model="dbp_phys", c_sign=sign, fir_init="physical", alpha_centre=alpha)}
        for arm, spec in arms.items():
            name = f"dbp_Ns{item['n_steps']}_L{item['fir_len']}_R{item['n_alpha']}_{arm}"
            for seed in g24.first_seeds(config, options, config["n_seeds"]):
                tasks.append(g24.run(exp, stage, directory, config, options, name=name, spec=spec, lr=lr,
                                     steps=config["steps"], seed=seed,
                                     meta={"arm": arm, "n_steps": item["n_steps"], "c_sign": sign, "recipe_source": source}))
    return tasks


BUILDERS["E049"] = e049_tasks


# --------------------------------------------------------------------------- E050 (reference: windowed DBP of E028, 10 seeds)

def e050_tasks(stage: str, options: Options) -> list[dict]:
    """Stage ref: the E028 DBP (window 31, FIR delta, alpha 0, recipe F7) at the Ns of config["ref"], 10 seeds.
    DBP-X itself is trained by constellation_denoiser.dbpx (not by the runner)."""
    exp = "E050"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage != "ref":
        raise SystemExit(f"E050: неизвестная стадия {stage!r}")
    ref = config["ref"]
    lr, _, source = g24.recipe_lr(ref["recipe"], options)
    tasks = []
    for ns in ref["n_steps"]:
        spec = g24.dbp_spec(ns, ref["fir_len"], ref["n_alpha"])
        for seed in g24.first_seeds(config, options, ref["n_seeds"]):
            tasks.append(g24.run(exp, stage, directory, config, options, name=spec["name"], spec=spec, lr=lr,
                                 steps=ref["steps"], seed=seed, meta={"n_steps": ns, "recipe_source": source}))
    return tasks


BUILDERS["E050"] = e050_tasks


def e051_tasks(stage: str, options: Options) -> list[dict]:
    """E051 is closed form (e046_fit); the runner only trains the control."""
    exp = "E051"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    raise SystemExit(f"E051: неизвестная стадия {stage!r} (подгонки: e046_fit)")


BUILDERS["E051"] = e051_tasks


def e053_tasks(stage: str, options: Options) -> list[dict]:
    """E053 is closed form (e053_data); the runner only trains the control."""
    exp = "E053"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    raise SystemExit(f"E053: неизвестная стадия {stage!r} (подгонки: e053_data)")


BUILDERS["E053"] = e053_tasks


def e052_tasks(stage: str, options: Options) -> list[dict]:
    """E052 is closed form (e052_dbpx_triplets); the runner only trains the control."""
    exp = "E052"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    raise SystemExit(f"E052: неизвестная стадия {stage!r}")


BUILDERS["E052"] = e052_tasks


def e054_tasks(stage: str, options: Options) -> list[dict]:
    """E054 trains DBP-X in e054_prune; the runner only trains the control."""
    exp = "E054"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    raise SystemExit(f"E054: неизвестная стадия {stage!r}")


BUILDERS["E054"] = e054_tasks


def e055_tasks(stage: str, options: Options) -> list[dict]:
    """Small real tanh MLP (one hidden layer) with the E025 F10 recipe, 5 seeds per (window, hidden)."""
    exp = "E055"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage != "main":
        raise SystemExit(f"E055: неизвестная стадия {stage!r}")
    tasks = []
    for item in config["grid"]:
        w, h = item["window"], item["hidden"]
        spec = {"model": "real_mlp", "window": w, "hidden": h, "layers": 1, "activation": "tanh", "init_scale": 1.0}
        for seed in g24.first_seeds(config, options, len(config["seeds"])):
            tasks.append(g24.run(exp, stage, directory, config, options, name=f"real_W{w}_h{h}", spec=spec,
                                 lr=config["lr"], steps=config["steps"], seed=seed,
                                 meta={"params": item["params"], "window": w, "hidden": h}))
    return tasks


BUILDERS["E055"] = e055_tasks


def e056_tasks(stage: str, options: Options) -> list[dict]:
    """E056 is closed form (e046_fit); the runner only trains the control."""
    exp = "E056"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    raise SystemExit(f"E056: неизвестная стадия {stage!r}")


BUILDERS["E056"] = e056_tasks


def e057_tasks(stage: str, options: Options) -> list[dict]:
    """E057 trains DBP-X in e057_ld63 (e054_prune.run); the runner only trains the control."""
    exp = "E057"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    raise SystemExit(f"E057: неизвестная стадия {stage!r}")


BUILDERS["E057"] = e057_tasks


def e059_tasks(stage: str, options: Options) -> list[dict]:
    """E059 trains DBP-X in e059_kerr_width (dbpx.train_one); the runner only trains the control."""
    exp = "E059"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    raise SystemExit(f"E059: неизвестная стадия {stage!r}")


BUILDERS["E059"] = e059_tasks


def _register() -> None:
    from . import grids

    grids.BUILDERS.update(BUILDERS)
    grids.SPECIAL_STAGES[("E047", "init")] = e047_init
    grids.SPECIAL_STAGES[("E048", "select")] = e048_select


_register()
