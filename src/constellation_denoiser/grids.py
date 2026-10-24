"""Task lists for E010-E014 built from each experiment's config.json.

A task is a plain dict consumed by ``runner.run_task``. Run ids do not contain
the stage name, so a configuration shared by two stages (e.g. the const 1e-3
control in E010 baseline-check and stage1) is trained once and reused.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .lr_schedules import LRConfig
from .paths import CHECKPOINTS_DIR

SMOKE_STEPS = 300
SMOKE_EVAL = 100


@dataclass
class Options:
    smoke: bool = False
    force: bool = False
    synthetic: bool = False
    seeds: list[int] | None = None
    max_steps: int | None = None
    steps: int | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Options":
        def flag(name):
            return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes")

        seeds = os.environ.get("SEEDS", "").replace(" ", ",").strip(",")
        return cls(
            smoke=flag("SMOKE"), force=flag("FORCE"), synthetic=flag("SYNTHETIC"),
            seeds=[int(s) for s in seeds.split(",") if s] or None,
            max_steps=int(os.environ["MAX_STEPS"]) if os.environ.get("MAX_STEPS") else None,
            steps=int(os.environ["STEPS"]) if os.environ.get("STEPS") else None,
        )


# --------------------------------------------------------------------------- helpers

def load_config(exp: str) -> tuple[Path, dict]:
    from .runner import exp_dir

    directory = exp_dir(exp)
    return directory, json.loads((directory / "config.json").read_text(encoding="utf-8"))


def seeds_for(config: dict, options: Options, default_key: str = "seeds") -> list[int]:
    seeds = options.seeds or config[default_key]
    return seeds[:1] if options.smoke else seeds


def lr_tag(lr: dict) -> str:
    return LRConfig.from_dict(lr).tag()


def spec_tag(spec: dict) -> str:
    keys = [k for k in ("hidden", "d_model", "layers", "grid", "kan_hidden") if k in spec]
    return "_".join([spec.get("name", spec["model"]), f"w{spec['window']}"] + [f"{k[0]}{spec[k]}" for k in keys])


def make_task(exp: str, stage: str, directory: Path, options: Options, *, spec: dict, seed: int,
              n_train: int, phases: list[dict], lr: dict, lr_name: str, eval_every: int,
              test_every: int | None = None, select_every: int = 1000, run_id: str | None = None,
              **extra) -> dict:
    from .runner import default_files

    total = sum(int(p["steps"]) for p in phases)
    if run_id is None:
        run_id = f"{spec_tag(spec)}_N{n_train}_{lr_name}_{total // 1000}k_seed{seed}"
    if os.environ.get("TEST_EVERY"):
        # test is report-only; it must still hit every selection point (steps divisible by select_every)
        override = int(os.environ["TEST_EVERY"])
        if select_every % override:
            raise SystemExit(f"TEST_EVERY={override} must divide select_every={select_every}")
        test_every = max(override, eval_every)
    if options.smoke:
        share = max(1, SMOKE_STEPS // len(phases))
        phases = [dict(p, steps=min(int(p["steps"]), share)) for p in phases]
        eval_every, test_every, select_every = SMOKE_EVAL, SMOKE_EVAL, SMOKE_EVAL
    root = directory / ("results_smoke" if options.smoke else "results")
    ckpt_dir = (root / "checkpoints") if options.smoke else (CHECKPOINTS_DIR / exp)
    task = {
        "experiment": exp, "stage": stage, "run_id": run_id, "spec": spec, "seed": seed,
        "files": default_files(n_train, options.smoke), "phases": phases, "lr": lr, "lr_name": lr_name,
        "eval_every": eval_every, "test_every": test_every or eval_every, "select_every": select_every,
        "out_json": str(root / "runs" / f"{run_id}.json"), "out_curve": str(root / "curves" / f"{run_id}.csv"),
        "ckpt_dir": str(ckpt_dir), "smoke": options.smoke, "synthetic": options.synthetic, "budget": total,
        "n_train_nominal": n_train,
    }
    task.update(extra)
    return task


def read_runs(directory: Path, smoke: bool) -> list[dict]:
    runs_dir = directory / ("results_smoke" if smoke else "results") / "runs"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(runs_dir.glob("*.json"))]


def lr_from(base: dict, **changes) -> dict:
    merged = dict(base)
    merged.update(changes)
    return LRConfig.from_dict(merged).to_dict()


def mean_final_val(runs: list[dict]) -> dict[tuple, dict]:
    """Group runs by (variant, lr_name, budget): mean final val reduction, divergence."""
    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        groups.setdefault((run["variant"], run["lr_name"], run["budget"]), []).append(run)
    out = {}
    for key, members in groups.items():
        values = [m["val_reduction"] for m in members if m.get("val_reduction") is not None]
        out[key] = {"mean_val_reduction": float(np.mean(values)) if values else float("nan"),
                    "n": len(members), "diverged": any(m.get("diverged") for m in members),
                    "lr_config": members[0]["lr_config"]}
    return out


# --------------------------------------------------------------------------- E010

def e010_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E010"
    directory, config = load_config(exp)
    steps = config["steps"]
    eval_every, test_every, select_every = config["eval_every"], config["test_every"], config["select_every"]
    n_train = config["n_train_files"]
    seeds = seeds_for(config, options)
    base_lr = config["base_lr"]
    tasks = []

    def add(arch: str, lr: dict, budget: int = steps, control_key: str | None = None, seed_list=None):
        lr = lr_from(lr, horizon=None)
        name = lr_tag(lr)
        for seed in seed_list or seeds:
            tasks.append(make_task(
                exp, stage, directory, options, spec=dict(config["models"][arch], name=arch), seed=seed,
                n_train=n_train, phases=[{"steps": budget}], lr=lr, lr_name=name, eval_every=eval_every,
                test_every=test_every, select_every=select_every, variant=arch,
                control_key=control_key if budget == steps else None,
            ))

    if stage == "baseline-check":
        for arch in config["models"]:
            add(arch, base_lr, control_key=f"{arch}_const_1e-3_20k")
        add("both", lr_from(base_lr, schedule="cosine_to_zero"), control_key="both_cosine_to_zero_1e-3_20k")
        return tasks

    if stage == "stage1":
        grid = config["stages"]["stage1"]
        for arch in config["models"]:
            for peak in grid["peak_lr"]:
                for schedule in grid["schedule"]:
                    add(arch, lr_from(base_lr, peak_lr=peak, schedule=schedule))
        return tasks

    if stage == "stage2":
        selection = e010_selection_path(directory, options.smoke)
        if not selection.is_file():
            raise SystemExit(f"{selection} не найден: сначала выполните стадию select")
        chosen = json.loads(selection.read_text(encoding="utf-8"))
        grid = config["stages"]["stage2"]
        for arch, configs in chosen["top"].items():
            for item in configs:
                for warmup in grid["warmup"]:
                    for c_mult in grid["c_mult"]:
                        add(arch, lr_from(item["lr_config"], warmup=warmup, c_mult=c_mult))
        return tasks

    if stage == "stage3":
        runs = [r for r in read_runs(directory, options.smoke) if r["budget"] == steps]
        groups = mean_final_val(runs)
        choice = {}
        for arch in config["models"]:
            candidates = {k: v for k, v in groups.items() if k[0] == arch and not v["diverged"]}
            if not candidates:
                raise SystemExit(f"нет результатов stage1/stage2 для {arch}: стадия stage3 невозможна")
            best_key = max(candidates, key=lambda k: candidates[k]["mean_val_reduction"])
            choice[arch] = {"lr_name": best_key[1], **candidates[best_key]}
        from .runner import atomic_write_json

        atomic_write_json(e010_selection_path(directory, options.smoke).with_name("stage3_choice.json"), choice)
        for budget in config["stages"]["stage3"]["budgets"]:
            for arch in config["models"]:
                add(arch, choice[arch]["lr_config"], budget=budget)
                add(arch, base_lr, budget=budget)
        return tasks
    raise SystemExit(f"E010: неизвестная стадия {stage!r}")


def e010_selection_path(directory: Path, smoke: bool) -> Path:
    return directory / ("results_smoke" if smoke else "results") / "stage1_selection.json"


def e010_select(options: Options) -> int:
    """Pick the 2 best stage1 configurations per architecture by mean final validation reduction."""
    from .runner import atomic_write_json, log

    directory, config = load_config("E010")
    expected = {t["run_id"] for t in e010_tasks("stage1", options)}
    runs = [r for r in read_runs(directory, options.smoke) if r["run_id"] in expected]
    missing = len(expected) - len(runs)
    if missing:
        log(f"select: не хватает {missing} прогонов stage1 из {len(expected)}; выбор по неполным данным запрещён")
        return 2
    groups = mean_final_val(runs)
    top = {}
    for arch in config["models"]:
        candidates = sorted(((k, v) for k, v in groups.items() if k[0] == arch and not v["diverged"]),
                            key=lambda kv: kv[1]["mean_val_reduction"], reverse=True)
        top[arch] = [{"lr_name": k[1], **v} for k, v in candidates[: config["stages"]["select"]["top_k"]]]
        for item in top[arch]:
            log(f"select {arch}: {item['lr_name']} mean val reduction {item['mean_val_reduction']:.2f}%")
    payload = {"criterion": "mean over seeds of final validation BER reduction, diverged configs excluded",
               "top": top, "best": {arch: items[0] for arch, items in top.items() if items},
               "smoke": options.smoke}
    atomic_write_json(e010_selection_path(directory, options.smoke), payload)
    return 0


# --------------------------------------------------------------------------- best_e010 branch (E011, E012)

def best_e010_lr(smoke: bool) -> dict[str, dict] | None:
    """Best stage1 lr config per E010 architecture ('both', 'a'), or None if E010 select has not run."""
    from .runner import exp_dir

    path = e010_selection_path(exp_dir("E010"), smoke)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {arch: item["lr_config"] for arch, item in payload.get("best", {}).items()}


def lr_branches(config: dict, arch: str | None, smoke: bool) -> list[tuple[str, dict]]:
    """[('const_1e-3', lr), ('best_e010-<tag>', lr)]; the second only if E010 selected one for ``arch``."""
    branches = [("const_1e-3", lr_from(config["base_lr"]))]
    best = best_e010_lr(smoke) if arch else None
    if best and arch in best:
        lr = lr_from(best[arch], horizon=None)
        branches.append((f"best_e010-{lr_tag(lr)}", lr))
    return branches


def control_tasks(exp: str, directory: Path, config: dict, options: Options) -> list[dict]:
    """20k-step controls reproducing E007/E009 numbers (eval every 1000 as in E007-E009)."""
    tasks = []
    for key, item in config["control"]["runs"].items():
        rule = config["control"]["targets"][key]
        seeds = item.get("seeds", rule["seeds"])      # control runs may cover more seeds than the tolerance
        for seed in ([seeds[0]] if options.smoke else seeds):
            tasks.append(make_task(
                exp, "control", directory, options, spec=dict(item["spec"]), seed=seed,
                n_train=item["n_train_files"], phases=[{"steps": item["steps"]}], lr=lr_from(config["base_lr"]),
                lr_name="const_1e-3", eval_every=1000, select_every=1000, control_key=key,
                variant=item["spec"].get("name"), run_id=f"control_{spec_tag(item['spec'])}_N{item['n_train_files']}"
                f"_{item['steps'] // 1000}k_seed{seed}",
            ))
    return tasks


# --------------------------------------------------------------------------- E011

def e011_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E011"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage not in ("const", "best"):
        raise SystemExit(f"E011: неизвестная стадия {stage!r}")
    steps = options.max_steps or config["max_steps"]
    every = config["eval_every"]
    tasks = []
    for name, model in config["models"].items():
        spec = dict(model["spec"], name=name)
        for lr_name, lr in lr_branches(config, model.get("e010_arch"), options.smoke):
            if (stage == "const") != (lr_name == "const_1e-3"):
                continue
            for n_train in config["n_train_files"]:
                for seed in seeds_for(config, options):
                    tasks.append(make_task(
                        exp, stage, directory, options, spec=spec, seed=seed, n_train=n_train,
                        phases=[{"steps": steps}], lr=lr, lr_name=lr_name, eval_every=every,
                        select_every=every, variant=name, ckpt_every=config["ckpt_every"],
                    ))
    if stage == "best" and not tasks:
        from .runner import log

        log("E011 best: нет experiments/E010_lr_schedule_study/results/stage1_selection.json — ветка пропущена")
    return tasks


# --------------------------------------------------------------------------- E012

def e012_variant_spec(variant: dict, window: int) -> dict:
    from .model_zoo import complex_two_hidden_macs, width_matching_macs

    spec = {"model": variant["model"], "window": window, "hidden": variant.get("hidden", 48)}
    if variant.get("match_macs_of_two_hidden_kerr_both"):
        spec["hidden"] = width_matching_macs(window, complex_two_hidden_macs(window, 48))
    return spec


def e012_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E012"
    directory, config = load_config(exp)
    if stage not in ("control", "const", "best"):
        raise SystemExit(f"E012: неизвестная стадия {stage!r}")
    controls = config["control"]["targets"]
    tasks = []
    for name, variant in config["variants"].items():
        for window in config["windows"]:
            spec = dict(e012_variant_spec(variant, window), name=name)
            for lr_name, lr in lr_branches(config, variant.get("e010_arch"), options.smoke):
                branch = "const" if lr_name == "const_1e-3" else "best"
                for budget, every in zip(config["steps"], config["eval_every"]):
                    for seed in seeds_for(config, options):
                        key = next((k for k, rule in controls.items() if rule["variant"] == name
                                    and rule["window"] == window and rule["steps"] == budget
                                    and branch == "const" and seed in rule["seeds"]), None)
                        if stage == "control" and key is None:
                            continue
                        if stage in ("const", "best") and branch != stage:
                            continue
                        tasks.append(make_task(
                            exp, stage, directory, options, spec=spec, seed=seed,
                            n_train=config["n_train_files"], phases=[{"steps": budget}], lr=lr,
                            lr_name=lr_name, eval_every=every, select_every=every, variant=name,
                            control_key=key, ckpt_every=config["ckpt_every"],
                            post_hook="merge_check" if variant.get("merge_check") else None,
                        ))
    if stage == "control" and options.smoke:
        tasks = [t for t in tasks if t["seed"] == tasks[0]["seed"]]
    if stage == "best" and not tasks:
        from .runner import log

        log("E012 best: нет experiments/E010_lr_schedule_study/results/stage1_selection.json — ветка пропущена")
    return tasks


def merge_check_hook(model, task, train, val, test) -> dict:
    """Merged inference graph of variant a must match the trained model on 1000 windows (float32)."""
    import torch

    from .complex_mlp_ablation import MergedAblationMLP

    window = int(task["spec"]["window"])
    model = model.cpu().eval()
    merged = MergedAblationMLP(model).eval()
    n = min(1000, train.n_windows(window))
    x, _ = train.gather(torch.arange(n), window)
    with torch.inference_mode():
        diff = float((model(x.cpu()) - merged(x.cpu())).abs().max().item())
    return {"merge_max_abs_diff": diff, "merge_windows_checked": n, "merge_ok": bool(diff < 1e-4)}


# --------------------------------------------------------------------------- E013

def e013_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E013"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage not in config["stages"]:
        raise SystemExit(f"E013: неизвестная стадия {stage!r}")
    grid = config["stages"][stage]
    lr = lr_from(config["transformer_lr"])
    tasks = []
    for window in grid["windows"]:
        for d_model, layers in grid["d_model_layers"]:
            spec = {"model": "transformer", "window": window, "d_model": d_model, "layers": layers,
                    "heads": config["heads"], "ffn_mult": config["ffn_mult"], "name": "transformer"}
            for n_train in grid["n_train_files"]:
                for seed in seeds_for(config, options):
                    tasks.append(make_task(
                        exp, stage, directory, options, spec=spec, seed=seed, n_train=n_train,
                        phases=[{"steps": grid["steps"]}], lr=lr, lr_name="adamw_wu1000_cosine",
                        eval_every=grid["eval_every"], select_every=grid["eval_every"],
                        variant=f"transformer_d{d_model}_L{layers}", ckpt_every=config["ckpt_every"],
                    ))
    return tasks


# --------------------------------------------------------------------------- E014

def e014_phases(kind: str, steps: int) -> list[dict]:
    half = steps // 2
    if kind == "single":
        return [{"steps": steps}]
    if kind == "continued":   # same Kerr MLP, fresh Adam for the second half (equal-steps control of k3/k4)
        return [{"steps": half}, {"steps": steps - half, "trainable": "all", "reset_optimizer": True}]
    if kind == "staged":      # Kerr alone, then Kerr frozen and only the KAN trained on the residual
        return [{"steps": half, "trainable": ["kerr."]},
                {"steps": steps - half, "trainable": ["kan."], "reset_optimizer": True}]
    raise ValueError(kind)


def e014_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E014"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage != "main":
        raise SystemExit(f"E014: неизвестная стадия {stage!r}")
    steps = min(options.steps or config["steps"], config["max_steps"])
    lr = lr_from(config["base_lr"])
    tasks = []
    for name, model in config["models"].items():
        grids_ = config["grids"] if model.get("uses_grid") else [None]
        for grid in grids_:
            spec = dict(model["spec"], window=config["window"], name=name)
            if grid is not None:
                spec["grid"] = grid
            for seed in seeds_for(config, options):
                tasks.append(make_task(
                    exp, stage, directory, options, spec=spec, seed=seed, n_train=config["n_train_files"],
                    phases=e014_phases(model["phases"], steps), lr=lr, lr_name="const_1e-3",
                    eval_every=config["eval_every"], select_every=config["eval_every"], variant=name,
                    post_hook="kan_costs", force_checkpoints=True, ckpt_every=config["ckpt_every"],
                ))
    return tasks


def kan_costs_hook(model, task, train, val, test) -> dict:
    from .kan import kan_cost_details

    spec = task["spec"]
    if spec["model"] in ("kan", "kerr_kan"):
        return kan_cost_details(spec)
    from .model_zoo import cost

    macs = cost(spec).macs
    return {"kan_widths": [], "kan_params": 0, "kan_macs_dense": 0, "kan_macs_sparse": 0, "kerr_macs": macs,
            "macs_dense_total": macs, "macs_sparse_total": macs}


# --------------------------------------------------------------------------- E016-E019 helpers

def e010_stage3_lr(smoke: bool, arch: str = "a") -> dict | None:
    """lr config chosen by E010 stage3 (validation) for ``arch``; smoke prefers results_smoke if present."""
    from .runner import exp_dir

    directory = exp_dir("E010")
    candidates = [directory / "results_smoke" / "stage3_choice.json"] if smoke else []
    candidates.append(directory / "results" / "stage3_choice.json")
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))[arch]["lr_config"]
    return None


def phys_lr_branches(config: dict, smoke: bool) -> list[tuple[str, dict]]:
    """const_1e-3 plus best_e010 with c_mult 1 and 30 (the latter scales the lr of alpha / Kerr c)."""
    branches = [("const_1e-3", lr_from(config["base_lr"]))]
    best = e010_stage3_lr(smoke)
    if best is not None:
        for c_mult in config.get("best_e010_c_mult", [1, 30]):
            branches.append((f"best_e010_cm{c_mult}", lr_from(best, c_mult=c_mult, horizon=None)))
    return branches


def phys_spec(window: int, fir_len: int, n_alpha: int, sym: bool, **extra) -> dict:
    kind = extra.pop("model", "phys_kerr")
    name = extra.pop("name", None) or (f"fir_L{fir_len}_{'sym' if sym else 'nosym'}" if n_alpha == 0 else
                                       f"phys_R{n_alpha}_L{fir_len}_{'sym' if sym else 'nosym'}")
    return {"model": kind, "window": window, "fir_len": fir_len, "n_alpha": n_alpha, "sym": sym, "name": name,
            **extra}


# --------------------------------------------------------------------------- E016

def e016_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E016"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage != "main":
        raise SystemExit(f"E016: неизвестная стадия {stage!r}")
    grid = config["grid"]
    specs, seen = [], set()
    for n_alpha in grid["n_alpha"]:
        for fir_len in grid["fir_len"]:
            for sym in grid["sym"]:
                # with one alpha and no FIR the symmetric and general models are identical: train once
                key = (n_alpha, fir_len, sym if (n_alpha > 1 or fir_len > 1) else False)
                if key not in seen:
                    seen.add(key)
                    specs.append(phys_spec(config["window"], fir_len=key[1], n_alpha=key[0], sym=key[2]))
    for fir in config["fir_only"]:
        specs.append(phys_spec(config["window"], fir["fir_len"], 0, fir["sym"]))
    tasks = []
    for spec in specs:
        for lr_name, lr in phys_lr_branches(config, options.smoke):
            for seed in seeds_for(config, options):
                tasks.append(make_task(
                    exp, stage, directory, options, spec=spec, seed=seed, n_train=config["n_train_files"],
                    phases=[{"steps": config["steps"]}], lr=lr, lr_name=lr_name, eval_every=config["eval_every"],
                    select_every=config["eval_every"], variant=spec["name"],
                ))
    return tasks


# --------------------------------------------------------------------------- E017

def e016_results_csv(smoke: bool) -> Path | None:
    from .runner import exp_dir

    directory = exp_dir("E016")
    for path in ([directory / "results_smoke" / "phys_kerr.csv"] if smoke else []) + [directory / "results" /
                                                                                      "phys_kerr.csv"]:
        if path.is_file():
            return path
    return None


def e016_choice(config: dict, smoke: bool) -> dict:
    """PhysKerr configuration (and its lr branch) with the best mean validation reduction in E016."""
    import pandas as pd

    path = e016_results_csv(smoke)
    if path is not None:
        runs = pd.read_csv(path)
        runs = runs[(runs.model == "phys_kerr") & (runs.spec_n_alpha > 0) & (~runs.diverged.astype(bool))]
        if not runs.empty:
            mean = runs.groupby(["variant", "lr_name"]).agg(
                val=("best_val_reduction", "mean"), n_alpha=("spec_n_alpha", "first"),
                fir_len=("spec_fir_len", "first"), sym=("spec_sym", "first")).reset_index()
            best = mean.loc[mean.val.idxmax()]
            return {"n_alpha": int(best.n_alpha), "fir_len": int(best.fir_len), "sym": bool(best.sym),
                    "lr_name": str(best.lr_name), "mean_best_val_reduction": float(best.val), "source": str(path)}
    fallback = config["phys_fallback"]
    return {**fallback, "lr_name": None, "source": "fallback (E016 results not found)"}


def e017_lr(config: dict, choice: dict, smoke: bool) -> tuple[str, dict]:
    branches = dict(phys_lr_branches(config, smoke))
    if choice.get("lr_name") in branches:          # the branch under which the chosen PhysKerr was best
        return choice["lr_name"], branches[choice["lr_name"]]
    best = e010_stage3_lr(smoke)
    if best is not None:                           # plan default: best_e010 as chosen by E010 stage3
        lr = lr_from(best, horizon=None)
        return f"best_e010_cm{lr['c_mult']:g}", lr
    return "const_1e-3", lr_from(config["base_lr"])


def e017_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E017"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage != "main":
        raise SystemExit(f"E017: неизвестная стадия {stage!r}")
    choice = e016_choice(config, options.smoke)
    lr_name, lr = e017_lr(config, choice, options.smoke)
    from .runner import atomic_write_json

    atomic_write_json(directory / ("results_smoke" if options.smoke else "results") / "e016_choice.json",
                      {**choice, "lr_name_used": lr_name, "lr_config_used": lr})
    tag = f"R{choice['n_alpha']}_L{choice['fir_len']}_{'sym' if choice['sym'] else 'nosym'}"
    steps = config["steps_per_stage"]
    tasks = []
    for hidden in config["mlp_hidden"]:
        for mode in config["modes"]:
            spec = phys_spec(config["window"], choice["fir_len"], choice["n_alpha"], choice["sym"],
                             model="phys_plus_mlp", mlp_hidden=hidden, name=f"hyb_{tag}_h{hidden}_{mode}")
            if mode == "frozen":
                phases = [{"steps": steps, "trainable": ["phys."]},
                          {"steps": steps, "trainable": ["mlp."], "reset_optimizer": True}]
            else:
                phases = [{"steps": 2 * steps}]
            for seed in seeds_for(config, options):
                tasks.append(make_task(
                    exp, stage, directory, options, spec=spec, seed=seed, n_train=config["n_train_files"],
                    phases=phases, lr=lr, lr_name=lr_name, eval_every=config["eval_every"],
                    select_every=config["eval_every"], variant=f"h{hidden}_{mode}", mode=mode,
                    force_checkpoints=True, ckpt_every=config["ckpt_every"],
                ))
    return tasks


# --------------------------------------------------------------------------- E018

def e018_phases(mode: str, n_steps: int, per_step: int, finetune: int, base_lr: dict, finetune_div: float) -> list[dict]:
    if mode == "layerwise":
        phases = [{"steps": per_step, "trainable": [f"steps.{k}."], "reset_optimizer": True} for k in range(n_steps)]
        lr = lr_from(base_lr, peak_lr=base_lr["peak_lr"] / finetune_div)
        return phases + [{"steps": finetune, "trainable": "all", "reset_optimizer": True, "lr": lr}]
    if mode == "joint_from_scratch":
        return [{"steps": n_steps * per_step + finetune}]
    raise ValueError(mode)


def e018_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E018"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage != "main":
        raise SystemExit(f"E018: неизвестная стадия {stage!r}")
    grid = config["grid"]
    lr = lr_from(config["base_lr"])
    tasks = []
    for n_steps in grid["n_steps"]:
        for fir_len in grid["fir_len"]:
            for n_alpha in grid["n_alpha"]:
                for mode in config["modes"]:
                    spec = phys_spec(config["window"], fir_len, n_alpha, True, model="phys_dbp", n_steps=n_steps,
                                     name=f"dbp_Ns{n_steps}_L{fir_len}_R{n_alpha}_{mode}")
                    phases = e018_phases(mode, n_steps, config["steps_per_layer"], config["finetune_steps"],
                                         config["base_lr"], config["finetune_lr_divisor"])
                    for seed in seeds_for(config, options):
                        tasks.append(make_task(
                            exp, stage, directory, options, spec=spec, seed=seed, n_train=config["n_train_files"],
                            phases=phases, lr=lr, lr_name="const_1e-3", eval_every=config["eval_every"],
                            select_every=config["eval_every"], variant=spec["name"],
                            force_checkpoints=True, ckpt_every=config["ckpt_every"],
                        ))
    return tasks


# --------------------------------------------------------------------------- E019 (confirm)

def results_csv(exp: str, name: str, smoke: bool) -> Path | None:
    from .runner import exp_dir

    directory = exp_dir(exp)
    for path in ([directory / "results_smoke" / name] if smoke else []) + [directory / "results" / name]:
        if path.is_file():
            return path
    return None


def spec_from_row(row) -> dict:
    spec = {}
    for key, value in row.items():
        if key.startswith("spec_") and value == value and value is not None:   # skip NaN
            name = key[5:]
            if hasattr(value, "item"):              # numpy scalar -> Python (JSON-serializable)
                value = value.item()
            if isinstance(value, float) and value.is_integer() and name not in ("sym",):
                value = int(value)
            if name == "sym":
                value = bool(value) if not isinstance(value, str) else value == "True"
            spec[name] = value
    return spec


def family_best(exp: str, csv_name: str, model: str, smoke: bool) -> dict | None:
    """Validation-best configuration (mean best_val_reduction over seeds) of one family, with its lr branch."""
    import pandas as pd

    path = results_csv(exp, csv_name, smoke)
    if path is None:
        return None
    runs = pd.read_csv(path)
    runs = runs[(runs.model == model) & (~runs.diverged.astype(bool))]
    if model == "phys_kerr":
        runs = runs[runs.spec_n_alpha > 0]
    if runs.empty:
        return None
    means = runs.groupby(["variant", "lr_name"]).best_val_reduction.mean()
    variant, lr_name = means.idxmax()
    row = runs[(runs.variant == variant) & (runs.lr_name == lr_name)].iloc[0]
    branch_vals = runs[runs.variant == variant].groupby("lr_name").best_val_reduction.mean().to_dict()
    return {"experiment": exp, "variant": variant, "lr_name": lr_name, "spec": spec_from_row(row),
            "mean_best_val_reduction": float(means.max()), "branch_validation": branch_vals, "source": str(path)}


def scale_phases(phases: list[dict], total: int) -> list[dict]:
    """Same phase structure, lengths scaled to ``total`` steps (the last phase takes the remainder)."""
    original = sum(int(p["steps"]) for p in phases)
    scaled, used = [], 0
    for index, phase in enumerate(phases):
        steps = total - used if index == len(phases) - 1 else int(round(int(phase["steps"]) * total / original))
        scaled.append(dict(phase, steps=steps))
        used += steps
    return scaled


def family_phases(item: dict, config_by_exp: dict) -> list[dict]:
    spec, variant = item["spec"], item["variant"]
    if item["experiment"] == "E017" and variant.endswith("frozen"):
        return [{"steps": 1, "trainable": ["phys."]}, {"steps": 1, "trainable": ["mlp."], "reset_optimizer": True}]
    if item["experiment"] == "E018" and variant.endswith("layerwise"):
        e018 = config_by_exp["E018"]
        return e018_phases("layerwise", int(spec["n_steps"]), e018["steps_per_layer"], e018["finetune_steps"],
                           e018["base_lr"], e018["finetune_lr_divisor"])
    return [{"steps": 1}]


def e019_tasks(stage: str, options: Options) -> list[dict]:
    exp = "E019"
    directory, config = load_config(exp)
    if stage == "control":
        return control_tasks(exp, directory, config, options)
    if stage != "main":
        raise SystemExit(f"E019: неизвестная стадия {stage!r}")
    from .runner import atomic_write_json, log

    configs = {e: load_config(e)[1] for e in ("E016", "E017", "E018")}
    # estimate only: pick members from smoke results while keeping full lengths (no effect on real runs)
    from_smoke = options.smoke or os.environ.get("CONFIRM_ESTIMATE_FROM_SMOKE") == "1"
    members = []
    for exp_name, csv_name, model in config["families"]:
        item = family_best(exp_name, csv_name, model, from_smoke)
        if item is None:
            log(f"E019: нет результатов {exp_name} ({csv_name}) — семейство пропущено")
            continue
        members.append({**item, "kind": "phys"})
    for name, spec in config["controls"].items():
        members.append({"experiment": "control", "variant": name, "spec": dict(spec, name=name), "kind": spec["model"]})
    if not (from_smoke and not options.smoke):
        atomic_write_json(directory / ("results_smoke" if options.smoke else "results") / "confirm_choice.json",
                          members)

    stage3 = {arch: e010_stage3_lr(options.smoke, arch) for arch in ("a", "both")}
    tasks = []
    for item in members:
        spec = dict(item["spec"], name=f"{item['experiment']}_{item['spec'].get('name', item['variant'])}")
        phase_template = family_phases(item, configs) if item["kind"] == "phys" else [{"steps": 1}]
        if item["kind"] == "phys":
            vals = item.get("branch_validation", {})
            c_mult = 30 if vals.get("best_e010_cm30", float("-inf")) > vals.get("best_e010_cm1", float("-inf")) else 1
            best = stage3["a"]
            best_lr = None if best is None else lr_from(best, c_mult=c_mult, horizon=None)
        else:
            arch = "both" if item["kind"] == "complex_both" else "a"
            best_lr = None if stage3[arch] is None else lr_from(stage3[arch], horizon=None)
        branches = [("const_1e-3", lr_from(config["base_lr"]), config["const_steps"])]
        if best_lr is not None:
            branches += [(f"best_e010_cm{best_lr['c_mult']:g}", best_lr, config["best_steps"])]
        for lr_name, lr, lengths in branches:
            for total in lengths:
                phases = scale_phases(phase_template, total)
                if lr_name != "const_1e-3":
                    phases = [dict(p, lr=lr_from(lr, peak_lr=lr["peak_lr"] / configs["E018"]["finetune_lr_divisor"]))
                              if "lr" in p else p for p in phases]
                for seed in seeds_for(config, options):
                    tasks.append(make_task(
                        exp, stage, directory, options, spec=spec, seed=seed, n_train=config["n_train_files"],
                        phases=phases, lr=lr, lr_name=lr_name, eval_every=config["eval_every"],
                        select_every=config["eval_every"], variant=spec["name"], family=item["experiment"],
                        force_checkpoints=True, ckpt_every=config["ckpt_every"],
                    ))
    return tasks


# --------------------------------------------------------------------------- registry

BUILDERS = {"E010": e010_tasks, "E011": e011_tasks, "E012": e012_tasks, "E013": e013_tasks,
            "E014": e014_tasks, "E016": e016_tasks, "E017": e017_tasks,
            "E018": e018_tasks, "E019": e019_tasks}
SPECIAL_STAGES = {("E010", "select"): e010_select}
POST_HOOKS: dict = {"merge_check": merge_check_hook, "kan_costs": kan_costs_hook}


def build_tasks(exp: str, stage: str, options: Options) -> list[dict]:
    if exp not in BUILDERS:
        raise SystemExit(f"неизвестный эксперимент {exp}")
    return BUILDERS[exp](stage, options)
