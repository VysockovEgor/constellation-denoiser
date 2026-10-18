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
        for seed in ([rule["seeds"][0]] if options.smoke else rule["seeds"]):
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


# --------------------------------------------------------------------------- registry

BUILDERS = {"E010": e010_tasks, "E011": e011_tasks, "E012": e012_tasks}
SPECIAL_STAGES = {("E010", "select"): e010_select}
POST_HOOKS: dict = {"merge_check": merge_check_hook}


def build_tasks(exp: str, stage: str, options: Options) -> list[dict]:
    if exp not in BUILDERS:
        raise SystemExit(f"неизвестный эксперимент {exp}")
    return BUILDERS[exp](stage, options)
