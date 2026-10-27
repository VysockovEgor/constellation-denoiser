"""E025 summary: recipes per family x level (results/recipes.json) and the draft-rule verdicts.

    PYTHONPATH=src python -m constellation_denoiser.summarize_E025   (SMOKE=1 reads results_smoke/, writes results_smoke/recipes.json)

Writes results/runs_table.csv, results/configs_summary.csv (pair x configuration x steps), results/recipes_table.csv,
results/tuning_verdicts.csv, results/gate_g1.json (ranking changed: yes/no; replaced recipes), the project-level
results/recipes.json read by recipes.load_recipe, and figures E025_search_response.png, E025_before_after.png,
E025_hparam_importance.png.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from .grids_e024 import e025_configs, e025_stage_plan
from .paths import EXPERIMENTS_DIR, PROJECT_ROOT
from .runner import atomic_write_json, git_commit
from .summary_e024 import aggregate, cyc, disjoint, setup, verdict, write_verdicts

OLD_FAMILY = {"complex Kerr MLP": "Kerr MLP", "real tanh MLP": "real MLP"}


def main() -> int:
    smoke, directory, config, results, figures, runs = setup("E025")
    if runs.empty:
        print("нет прогонов")
        return 1
    runs = runs[runs.stage.isin(["s1", "s2", "s3"]) & runs.e025_pair.notna()]
    table = aggregate(runs, ["e025_pair", "e025_config", "e025_kind", "budget"])
    table.to_csv(results / "configs_summary.csv", index=False)
    rows, recipes, replaced = [], {}, []
    for key, pair in config["pairs"].items():
        part = table[table.e025_pair == key]
        if part.empty:
            continue
        plan = e025_stage_plan(key, config)
        items = {i["id"]: i for i in e025_configs(key, pair, config)}
        s3 = part[part.budget == part.budget.max()] if smoke else part[part.budget == plan["s3"]["steps"]]
        ref_id = f"{key}_ref_current"
        compare = s3 if ref_id in set(s3.e025_config) else part[part.budget == (part.budget.max() if smoke else plan["s2"]["steps"])]
        if s3.empty or compare.empty:
            continue
        best = s3.loc[s3.val_mean.idxmax()]
        best_cmp = compare.loc[compare.val_mean.idxmax()]
        ref = compare[compare.e025_config == ref_id]
        if len(ref):
            ref = ref.iloc[0]
            gain = float(best_cmp.val_mean - ref.val_mean)
            new = best_cmp.e025_config != ref_id and gain >= 0.5 and disjoint(best_cmp.val_min, best_cmp.val_max,
                                                                                ref.val_min, ref.val_max)
        else:
            ref, gain, new = None, float("nan"), False
        chosen_id = best.e025_config if new else ref_id
        chosen = s3[s3.e025_config == chosen_id]
        chosen = chosen.iloc[0] if len(chosen) else (ref if ref is not None else best)
        item = items[chosen_id]
        recipes[key] = {"recipe": {"name": chosen_id, "lr": item["lr"], "spec_overrides": item["spec_overrides"]},
                        "family": pair["family"], "level_rm": pair["level_rm"], "val_mean": float(chosen.val_mean),
                        "test_mean": float(chosen.test_mean), "test_min": float(chosen.test_min),
                        "test_max": float(chosen.test_max), "n_seeds": int(chosen.n_seeds), "steps": int(chosen.budget),
                        "ref_current_val_mean": None if ref is None else float(ref.val_mean),
                        "ref_current_test_mean": None if ref is None else float(ref.test_mean),
                        "best_val_gain_over_ref": gain, "replaced": bool(new), "compared_at_steps": int(compare.budget.iloc[0])}
        if new:
            replaced.append(key)
        rows.append({"pair": key, "family": pair["family"], "rm": pair["level_rm"], "recipe": chosen_id,
                     "replaced": new, "val_gain_pp": gain, **{k: recipes[key][k] for k in
                                                              ("val_mean", "test_mean", "test_min", "test_max", "n_seeds",
                                                               "ref_current_val_mean", "ref_current_test_mean")}})
    rtable = pd.DataFrame(rows)
    rtable.to_csv(results / "recipes_table.csv", index=False)
    commit = git_commit()
    target = (PROJECT_ROOT / "results_smoke" / "recipes.json") if smoke else (PROJECT_ROOT / "results" / "recipes.json")
    atomic_write_json(target, {"source": "experiments/E025_family_tuning (summarize_E025)", "git_commit": commit,
                               "rule": config["rule_draft"]["recipe"], "recipes": recipes})

    verdicts = []
    for _, r in rtable.iterrows():
        verdicts.append({"hypothesis": f"{r.pair} ({r.family}, {int(r.rm)} RM): новый рецепт заменяет прежний",
                         "value": r.val_gain_pp, "verdict": verdict(bool(r.replaced)), "detail": r.recipe})
    f5 = table[table.e025_pair == "F5"]
    if len(f5):
        ref = f5[f5.e025_config == "F5_ref_current"]
        top = f5.loc[f5.val_mean.idxmax()]
        if len(ref):
            same = ref[ref.budget == top.budget]
            base = (same if len(same) else ref).iloc[0]
            gain = float(top.val_mean - base.val_mean)
            verdicts.append({"hypothesis": "чистая физика F5 была недообучена (лучший рецепт ≥ ref_current + 3 п.п.)",
                             "value": gain, "verdict": verdict(gain >= 3.0), "detail": f"{top.e025_config}, {int(top.budget)} шагов"})
    changed, detail = ranking_changed(rtable)
    verdicts.append({"hypothesis": "рейтинг семейств изменился (на уровне RM два семейства поменялись местами, ≥ 1 п.п.)",
                     "value": float("nan"), "verdict": verdict(changed), "detail": detail})
    write_verdicts(verdicts, config, results / "tuning_verdicts.csv")
    gate = {"ranking_changed": "да" if changed else "нет", "detail": detail, "replaced_recipes": replaced,
            "recipes_json": str(target.relative_to(PROJECT_ROOT)), "git_commit": commit}
    atomic_write_json((PROJECT_ROOT / "results_smoke" if smoke else PROJECT_ROOT / "results") / "gate_g1.json", gate)
    plots(runs, table, rtable, config, figures)
    print(f"рецепты: {target}; заменены: {', '.join(replaced) or 'нет'}; рейтинг изменился: {gate['ranking_changed']}")
    return 0


def ranking_changed(rtable: pd.DataFrame) -> tuple[bool, str]:
    path = EXPERIMENTS_DIR / "E021_pareto" / "results" / "level_best.csv"
    if not path.is_file() or rtable.empty:
        return False, "нет E021 level_best.csv"
    old = pd.read_csv(path)
    old["family"] = old.family.map(lambda f: OLD_FAMILY.get(f, f))
    swaps = []
    for level, part in old.groupby("level"):
        new = rtable[(rtable.rm >= level / 1.5) & (rtable.rm <= level * 1.5)]
        new = new.groupby("family").test_mean.max()
        prev = part.groupby("family").test_mean.max()
        for a, b in itertools.combinations(sorted(set(new.index) & set(prev.index)), 2):
            if np.sign(prev[a] - prev[b]) != np.sign(new[a] - new[b]) and abs(new[a] - new[b]) >= 1.0:
                swaps.append(f"{int(level)}: {a} / {b} (было {prev[a] - prev[b]:+.2f}, стало {new[a] - new[b]:+.2f})")
    return bool(swaps), "; ".join(swaps) or "перестановок нет"


def plots(runs: pd.DataFrame, table: pd.DataFrame, rtable: pd.DataFrame, config: dict, figures) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from . import plot_style

    plot_style.apply()
    s1 = runs[runs.stage == "s1"].groupby(["e025_pair", "e025_config"]).agg(
        val=("best_val_reduction", "mean"), lr=("lr_peak_lr", "first"), schedule=("lr_schedule", "first"),
        c_mult=("lr_c_mult", "first"), wd=("lr_weight_decay", "first"), warmup=("lr_warmup", "first")).reset_index()
    pairs = list(config["pairs"])
    cols = 5
    rows_n = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.2 * cols, 2.7 * rows_n), squeeze=False)
    schedules = ["const", "cosine_to_zero", "cosine_to_floor"]
    markers = {1.0: "o", 10.0: "s", 30.0: "^", 100.0: "D", 300.0: "v"}
    for ax, key in zip(axes.flat, pairs):
        part = s1[s1.e025_pair == key]
        for si, schedule in enumerate(schedules):
            for cm, marker in markers.items():
                p = part[(part.schedule == schedule) & (part.c_mult == cm)]
                if len(p):
                    ax.plot(p.lr, p.val.clip(lower=-20), marker, color=plot_style.color(si), markersize=4, linestyle="")
        ax.set_xscale("log")
        ax.set_title(key, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(pairs):]:
        ax.axis("off")
    handles = [Line2D([], [], color=plot_style.color(i), marker="o", linestyle="", label=s) for i, s in enumerate(schedules)]
    handles += [Line2D([], [], color=plot_style.TEXT, marker=m, linestyle="", label=f"c_mult {c:g}") for c, m in markers.items()]
    fig.legend(handles=handles, loc="lower right", fontsize=7, ncol=2)
    fig.supxlabel("peak lr (S1, 20k steps)")
    fig.supylabel("validation BER reduction, % (mean of 2 seeds)")
    fig.tight_layout()
    fig.savefig(figures / "E025_search_response.png", dpi=150)
    plt.close(fig)

    if len(rtable):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(rtable))
        ax.bar(x - 0.2, rtable.ref_current_test_mean.astype(float), 0.4, color=plot_style.GRID, edgecolor=plot_style.TEXT,
               hatch="//", label="ref_current (previous recipe)")
        families = list(dict.fromkeys(rtable.family))
        ax.bar(x + 0.2, rtable.test_mean, 0.4, color=[cyc(families.index(f)) for f in rtable.family],
               edgecolor=plot_style.TEXT, label="recipe after tuning",
               yerr=[rtable.test_mean - rtable.test_min, rtable.test_max - rtable.test_mean], capsize=2)
        ax.set_xticks(x, [f"{p}\n{f}" for p, f in zip(rtable.pair, rtable.family)], fontsize=7)
        ax.set_ylabel("test BER reduction, % (S3)")
        ax.set_title("Before / after equal tuning (test at the validation-best point; selection by validation)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "E025_before_after.png", dpi=160)
        plt.close(fig)

    from scipy.stats import spearmanr

    imp = []
    for key in pairs:
        part = s1[s1.e025_pair == key]
        for col in ("lr", "c_mult", "wd", "warmup"):
            if part[col].nunique() > 1:
                rho = spearmanr(part[col], part.val).statistic
                imp.append({"pair": key, "hparam": col, "rho": rho})
    if imp:
        imp = pd.DataFrame(imp).pivot(index="pair", columns="hparam", values="rho").reindex(pairs)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(imp.to_numpy(float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(imp.columns)), imp.columns)
        ax.set_yticks(range(len(imp.index)), imp.index)
        for (i, j), v in np.ndenumerate(imp.to_numpy(float)):
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="Spearman rho with S1 validation")
        ax.set_title("Hyper-parameter importance (S1)")
        fig.tight_layout()
        fig.savefig(figures / "E025_hparam_importance.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
