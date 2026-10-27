"""Generate journal/_auto/*.md from experiment files (numbers are never typed by hand).

    python scripts/journal_build_auto.py            # all of E010-E019 + index, legacy, gallery, slides
    python scripts/journal_build_auto.py E016 E017  # only these experiment blocks (+ shared pages)
    JOURNAL_TITLES=1 python scripts/journal_build_auto.py   # slides with titles (manifest 'title' or file name)

Writes plain Markdown (no executable cells): <E>_rule.md (question + rule_draft from config.json
verbatim), <E>_results.md (tables from per-run / summary CSVs, verdicts, figures), <E>_checks.md
(control_check.json, check-control log, anomalies), index_status.md, legacy.md, gallery.md, slides.md.
Hand-written sections of the pages (journal/experiments/*.qmd, "Выводы") are never touched.
Figures are linked through journal/figs/{experiments,results} (symlinks), never copied.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "journal"
AUTO = JOURNAL / "_auto"
EXPERIMENTS = ROOT / "experiments"
MAX_ROWS = 15
ANOMALY = {"worse_than_control_pp": 5.0, "seed_spread_pp": 3.0, "val_test_gap_pp": 5.0}
HEADER = "<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->\n\n"

# per-run table, grouping keys (configuration) for E010-E018
RUNS = {
    "E010": ("results/lr_sweep_q2.csv", ["variant", "lr_name", "budget"]),
    "E011": ("results/long_runs_q2.csv", ["variant", "lr_name", "n_train_nominal"]),
    "E012": ("results/depth_window_q2.csv", ["variant", "window", "budget", "lr_branch"]),
    "E013": ("results/transformer_runs_q2.csv", ["variant", "window", "n_train_nominal", "budget"]),
    "E014": ("results/kan_hybrid_q2.csv", ["variant", "spec_grid"]),
    "E016": ("results/phys_kerr.csv", ["variant", "lr_name"]),
    "E017": ("results/phys_plus_mlp.csv", ["variant", "lr_name"]),
    "E018": ("results/dbp_depth.csv", ["variant"]),
    "E020": ("results/runs_table.csv", ["variant", "lr_name", "budget"]),
    "E021": ("results/runs_table.csv", ["variant"]),
    "E022": ("results/runs_table.csv", ["variant"]),
    "E023": ("results/runs_table.csv", ["variant", "split_seed"]),
    "E025": ("results/runs_table.csv", ["e025_pair", "e025_config", "budget"]),
    "E026": ("results/runs_table.csv", ["variant", "stage", "budget"]),
    "E027": ("results/runs_table.csv", ["activation", "spec_hidden", "lr_id", "budget"]),
    "E028": ("results/runs_table.csv", ["variant", "stage", "budget"]),
    "E029": ("results/runs_table.csv", ["variant"]),
    "E030": ("results/runs_table.csv", ["model_key", "loss_name", "lr_scale", "budget"]),
    "E031": ("results/runs_table.csv", ["variant"]),
    "E032": ("results/runs_table.csv", ["variant"]),
    "E033": ("results/runs_table.csv", ["variant"]),
}
VERDICTS = {
    "E012": "results/depth_window_verdicts.csv", "E013": "results/transformer_verdicts.csv",
    "E014": "results/kan_hybrid_verdicts.csv", "E016": "results/phys_kerr_verdicts.csv",
    "E017": "results/phys_plus_mlp_verdicts.csv", "E018": "results/dbp_depth_verdicts.csv",
    "E020": "results/arms_verdicts.csv", "E021": "results/level_verdicts.csv",
    "E022": "results/baselines_verdicts.csv", "E023": "results/robustness_verdicts.csv",
    "E024": "results/audit_verdicts.csv", "E025": "results/tuning_verdicts.csv", "E026": "results/map_verdicts.csv",
    "E027": "results/activation_verdicts.csv", "E028": "results/dbp_verdicts.csv", "E029": "results/width_verdicts.csv",
    "E030": "results/loss_verdicts.csv", "E031": "results/triplet_verdicts.csv", "E032": "results/convergence_verdicts.csv",
    "E033": "results/pareto_verdicts.csv",
}
# extra tables shown on the experiment page: (file, {column: header}, title)
EXTRA_TABLES = {
    "E020": [("results/differences.csv", {"branch": "ветка E019", "lr": "lr", "val_5k": "валидация 5k", "val_40k": "валидация 40k",
              "test_valbest_100k": "тест 100k", "test_valbest_200k": "тест 200k", "test_final_200k": "тест, посл. шаг",
              "val_test_gap_200k": "тест − валидация", "train_loss_final": "train loss"}, "Различия аномального и нормального прогонов (E019)"),
             ("results/hypotheses.csv", {"hypothesis": "гипотеза", "text": "формулировка", "status": "статус", "evidence": "данные"},
              "Гипотезы и проверка на существующих данных"),
             ("results/arms.csv", {"arm": "плечо", "lr": "ветка lr", "steps": "шагов", "n_seeds": "запусков (seeds)",
              "test_mean": "тест, %", "test_min": "мин", "test_max": "макс", "derived": "из кривой 200k"}, "Перезапуски")],
    "E021": [("results/level_best.csv", {"level": "уровень", "family": "семейство", "label": "точка", "macs": "умножений",
              "n_seeds": "запусков (seeds)", "val_mean": "валидация", "test_mean": "тест, %", "test_min": "мин",
              "test_max": "макс", "q2_mean": "Q², дБ"}, "Лучшее по валидации семейство на каждом уровне")],
    "E022": [("results/baselines_summary.csv", {"family": "семейство", "variant": "вариант", "macs": "умножений",
              "n_seeds": "запусков (seeds)", "test_mean": "тест, %", "q2_mean": "Q², дБ", "q2_min": "Q² мин", "q2_max": "Q² макс"},
              "Базовые методы")],
    "E023": [("results/split_summary.csv", {"member": "участник", "split_seed": "разбиение", "n_seeds": "запусков (seeds)",
              "test_mean": "тест, %", "test_min": "мин", "test_max": "макс", "macs": "умножений"}, "Результаты по разбиениям")],
    "E024": [("results/e021_before_after.csv", {"level": "уровень RM", "old_best": "было (E021)", "old_test": "тест, %",
              "new_best": "стало (агрегатор v2)", "new_family": "семейство", "new_rm": "RM", "new_test": "тест, %"},
              "Было → стало: лучшая по валидации точка уровня"),
             ("results/tuning_budget.csv", {"family": "семейство", "runs": "прогонов", "lr_recipes": "рецептов lr",
              "c_mult_values": "c_mult", "step_budgets": "шагов", "max_seeds_per_config": "seeds (макс.)"},
              "Бюджет настройки E010–E022")],
    "E025": [("results/recipes_table.csv", {"pair": "пара", "family": "семейство", "rm": "RM", "recipe": "рецепт",
              "replaced": "заменён", "val_gain_pp": "выигрыш валидации, п.п.", "val_mean": "валидация", "test_mean": "тест, %",
              "test_min": "мин", "test_max": "макс", "ref_current_test_mean": "тест ref_current, %"}, "Рецепты после настройки")],
    "E026": [("results/instability.csv", {"spec_window": "окно", "runs": "прогонов", "diverged_share": "доля разошедшихся"},
              "Нестабильность по окнам")],
    "E028": [("results/b_summary.csv", {"variant": "вариант", "n_seeds": "запусков (seeds)", "val_mean": "валидация",
              "test_mean": "тест, %", "test_min": "мин", "test_max": "макс"}, "Диагноз послойного обучения (Б)")],
    "E031": [("results/triplet_summary.csv", {"arm": "рука", "M": "M", "n_terms": "членов", "rm_naive": "RM naive",
              "rm_sym": "RM sym", "val_mean": "валидация", "test_mean": "тест, %", "test_min": "мин", "test_max": "макс",
              "q2_mean": "Q², дБ"}, "Триплеты по рукам и M")],
    "E032": [("results/horizons.csv", {"variant": "модель", "horizon": "горизонт", "test_mean": "тест, %", "test_min": "мин",
              "test_max": "макс"}, "Тест в точке лучшей валидации по горизонтам")],
    "E033": [("results/level_best.csv", {"level": "уровень", "family": "семейство", "label": "точка", "rm": "RM",
              "n_seeds": "запусков (seeds)", "val_mean": "валидация", "test_mean": "тест, %", "q2_mean": "Q², дБ",
              "q2_min": "Q² мин", "q2_max": "Q² макс"}, "Лучшие по валидации на уровнях RM")],
}
TEST, VAL = "test_reduction_valbest_select", "best_val_reduction"
# the control is a different model (complex_a vs physics models): "worse than control" is not an anomaly there
CONTROL_IS_OTHER_MODEL = {"E016", "E017", "E018", "E020", "E021", "E022", "E023", "E024", "E025", "E026", "E027",
                          "E028", "E029", "E030", "E031", "E032", "E033"}
COLUMN_RU = {
    "hypothesis": "гипотеза", "candidate": "кандидат", "candidate_macs": "умножений (кандидат)",
    "reference": "эталон", "reference_macs": "умножений (эталон)", "diff_pp": "разница, п.п.",
    "seed_spread_pp": "разброс seeds, п.п.", "n_seeds": "запусков (seeds)", "threshold": "порог", "verdict": "вердикт",
    "lr_branch": "ветка lr", "window": "окно", "budget": "шагов", "deep": "двухслойный вариант",
    "diff_vs_v3": "разница с v3", "spread_vs_v3": "разброс (v3)", "diff_vs_v4": "разница с v4",
    "spread_vs_v4": "разброс (v4)", "beats_v3_and_v4": "лучше v3 и v4", "window_verdict": "вердикт для окна",
    "v2_minus_v3_short": "v2 − v3 (20k)", "v2_minus_v3_long": "v2 − v3 (100k)",
    "optimization_effect": "эффект оптимизации", "model": "модель", "n_train": "N (файлов)", "steps": "шагов",
    "macs_per_symbol": "умножений", "rival": "соперник", "rival_macs": "умножений (соперник)", "variant": "вариант",
    "grid": "сетка G", "baseline": "база", "added_macs_dense": "добавка умножений (плотно)",
    "added_macs_sparse": "добавка умножений (разреженно)", "macs_increase_dense_vs_k0_pct": "рост к k0, % (плотно)",
    "macs_increase_sparse_vs_k0_pct": "рост к k0, % (разреженно)", "diff_vs_complex_a_pp": "разница с complex_a, п.п.",
    "spread_vs_complex_a_pp": "разброс (complex_a), п.п.", "diff_vs_e016_pp": "разница с E016, п.п.",
    "spread_vs_e016_pp": "разброс (E016), п.п.",
}
VALUE_RU = {
    "better than control": "лучше контроля", "not proven": "не доказано", "control": "контроль",
    "no control": "нет контроля", "True": "да", "False": "нет", "diff >= -2 p.p.": "разница ≥ −2 п.п.",
    ">= 1 p.p. and > seed spread": "≥ 1 п.п. и больше разброса seeds",
}
VERDICT_KEYS_RU = {
    "condition_a_by_R": "условие (а) по R", "condition_a": "условие (а)", "rho_own_validation": "ρ с собственной мощностью (валидация)",
    "rho_nb_validation": "ρ с мощностью соседей (валидация)", "rho_nb_test": "ρ с мощностью соседей (тест)",
    "fit_gap_R7_minus_R1_validation_pp": "подгонка R7 − R1, валидация, п.п.",
    "fit_gap_R7_minus_R1_test_pp": "подгонка R7 − R1, тест, п.п.", "condition_b": "условие (б)",
    "test_agrees": "тест согласуется (в)",
}


# --------------------------------------------------------------------------- helpers

def folder(exp: str) -> Path | None:
    matches = sorted(p for p in EXPERIMENTS.glob(f"{exp}_*") if p.is_dir())
    return matches[0] if matches else None


def fmt(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "да" if value else "нет"
    if isinstance(value, str) and value in VALUE_RU:
        return VALUE_RU[value]
    if isinstance(value, (int,)) or (isinstance(value, float) and float(value).is_integer() and abs(value) >= 100):
        return f"{int(value):d}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|")


def table(frame: pd.DataFrame, columns: dict[str, str]) -> str:
    cols = [c for c in columns if c in frame.columns]
    heads = [COLUMN_RU.get(c, columns[c]) if columns[c] == c else columns[c] for c in cols]
    lines = ["| " + " | ".join(heads) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def question(exp: str) -> str:
    readme = folder(exp) / "README.md"
    match = re.search(r"\*\*Вопрос\.\*\*\s*(.+)", readme.read_text(encoding="utf-8")) if readme.is_file() else None
    return match.group(1).strip() if match else "—"


def title(exp: str) -> str:
    readme = folder(exp) / "README.md"
    first = readme.read_text(encoding="utf-8").splitlines()[0] if readme.is_file() else exp
    return first.lstrip("# ").strip()


def git_date(path: Path) -> str:
    out = subprocess.run(["git", "log", "-1", "--format=%cs", "--", str(path)], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    return out or datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def figure_link(path: Path, depth: int) -> str:
    """Path of a repository file as seen from a page ``depth`` levels below journal/ (through journal/figs)."""
    relative = path.resolve().relative_to(ROOT)
    return "../" * depth + "figs/" + relative.as_posix()


def github_blob(path: str) -> str | None:
    """https://github.com/<owner>/<repo>/blob/<current branch>/<path> from the origin remote, or None."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    match = re.search(r"[:/]([^/:]+)/([^/]+?)(?:\.git)?$", git("remote", "get-url", "origin"))
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    return f"https://github.com/{match.group(1)}/{match.group(2)}/blob/{branch}/{path}" if match and branch else None


def log_link() -> str:
    url = github_blob("LOG.md")
    return f"[LOG.md на GitHub]({url})" if url else "`LOG.md` (в корне репозитория)"


def write(name: str, text: str) -> None:
    AUTO.mkdir(parents=True, exist_ok=True)
    (AUTO / name).write_text(HEADER + text.rstrip() + "\n", encoding="utf-8")


def render_value(value, indent: int = 0) -> list[str]:
    pad = "  " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}- `{key}`:")
                lines += render_value(item, indent + 1)
            else:
                lines.append(f"{pad}- `{key}`: {item}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines += render_value(item, indent + 1)
            else:
                lines.append(f"{pad}- {item}")
        return lines
    return [f"{pad}{value}"]


# --------------------------------------------------------------------------- per-run summaries

def grouped(exp: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if exp not in RUNS:
        return None
    rel, keys = RUNS[exp]
    path = folder(exp) / rel
    if not path.is_file():
        return None
    runs = pd.read_csv(path)
    keys = [k for k in keys if k in runs.columns]
    q2 = next((c for c in ("valbest_q2_gain_db", "valbest_select_q2_gain_db") if c in runs.columns), None)
    runs["is_control"] = runs.get("stage", pd.Series("", index=runs.index)).eq("control") | \
        runs.get("control_key", pd.Series(index=runs.index, dtype=object)).notna()
    agg = runs.groupby(keys, dropna=False).agg(
        n_seeds=("seed", "count"), val_mean=(VAL, "mean"), test_mean=(TEST, "mean"), test_min=(TEST, "min"),
        test_max=(TEST, "max"), macs=("macs_per_symbol_merged", "first"),
        q2=(q2, "mean") if q2 else (TEST, lambda s: float("nan")),
        diverged=("diverged", "sum"), is_control=("is_control", "any")).reset_index()
    agg["config"] = agg[keys].astype(str).agg(" · ".join, axis=1)
    agg["spread"] = agg.test_max - agg.test_min
    return runs, agg


def results_table(agg: pd.DataFrame) -> str:
    controls = agg[agg.is_control]
    others = agg[~agg.is_control].sort_values("val_mean", ascending=False).head(MAX_ROWS - len(controls))
    shown = pd.concat([others, controls])
    shown = shown.assign(range=[f"{fmt(a)} … {fmt(b)}" for a, b in zip(shown.test_min, shown.test_max)],
                         role=["контроль" if c else "" for c in shown.is_control])
    note = (f"Показано {len(shown)} из {len(agg)} конфигураций: лучшие по валидации и контроли. "
            "Тест — в точке лучшей валидации, %; Q² — прирост в той же точке, дБ.\n\n")
    return note + table(shown, {"config": "конфигурация", "role": "", "n_seeds": "запусков (seeds)", "val_mean": "валидация",
                                "test_mean": "тест, среднее", "range": "тест, min … max", "q2": "Q², дБ",
                                "macs": "умножений"})


def anomalies(runs: pd.DataFrame, agg: pd.DataFrame, control_mean: float | None, other_model: bool = False) -> str:
    out = []
    diverged = runs[runs.diverged.astype(bool)] if "diverged" in runs else runs.iloc[0:0]
    out.append(("diverged = true", [str(r) for r in diverged.run_id]))
    if other_model:
        out.append(("«тест хуже контроля более чем на 5 п.п.» не применяется: контроль — другая модель (complex_a)", []))
    elif control_mean is not None:
        worse = agg[(~agg.is_control) & (agg.test_mean < control_mean - ANOMALY["worse_than_control_pp"])]
        out.append((f"тест хуже контроля ({fmt(control_mean)} %) более чем на {ANOMALY['worse_than_control_pp']:g} п.п.",
                    [f"{c}: {fmt(t)} %" for c, t in zip(worse.config, worse.test_mean)]))
    spread = agg[agg.spread > ANOMALY["seed_spread_pp"]]
    out.append((f"разброс seeds больше {ANOMALY['seed_spread_pp']:g} п.п.",
                [f"{c}: {fmt(s)} п.п." for c, s in zip(spread.config, spread.spread)]))
    gap = agg[(agg.val_mean - agg.test_mean).abs() > ANOMALY["val_test_gap_pp"]]
    out.append((f"валидация и тест расходятся больше чем на {ANOMALY['val_test_gap_pp']:g} п.п.",
                [f"{c}: валидация {fmt(v)}, тест {fmt(t)}" for c, v, t in zip(gap.config, gap.val_mean, gap.test_mean)]))
    lines = []
    for name, items in out:
        lines.append(f"- **{name}**" + ("" if name.startswith("«") else f": {len(items)}") + (":" if items else ""))
        lines += [f"  - {item}" for item in items[:10]]
        if len(items) > 10:
            lines.append(f"  - … ещё {len(items) - 10}")
    return "\n".join(lines) + "\n"


def control_block(exp: str) -> tuple[str, float | None]:
    path = folder(exp) / "results" / "control_check.json"
    if not path.is_file():
        return "Нет `results/control_check.json`: контроль не запускался.\n", None
    check = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(check["targets"])
    frame["mean"] = frame["mean"].astype(float)
    frame["ok"] = frame["ok"].map(lambda v: "в допуске" if v else "ВНЕ ДОПУСКА")
    frame["values"] = frame["values"].map(lambda v: ", ".join(fmt(x) for x in v))
    text = f"Итог: **{'пройден' if check['passed'] else 'НЕ ПРОЙДЕН'}** (`results/control_check.json`).\n\n"
    text += table(frame, {"key": "контроль", "target": "цель, %", "tol": "допуск, п.п.", "mean": "среднее, %",
                          "values": "по seeds", "ok": "итог"})
    log = folder(exp) / "logs" / "check-control.log"
    if log.is_file():
        blocks = log.read_text(encoding="utf-8").split("=== ")
        lines = [l for l in blocks[-2].splitlines() if "control " in l] if len(blocks) > 1 else []
        if lines:
            text += "\nПоследняя проверка (`logs/check-control.log`):\n\n" + "\n".join(f"    {l}" for l in lines) + "\n"
    return text, float(frame["mean"].iloc[0])


def figures(exp: str, depth: int) -> str:
    pngs = sorted((folder(exp) / "figures").glob("*.png"))
    if not pngs:
        return "Рисунков нет.\n"
    return "\n".join(f"![{p.name}]({figure_link(p, depth)}){{group=\"{exp}\"}}\n" for p in pngs)


def verdict_block(exp: str) -> str:
    d = folder(exp)
    if exp in VERDICTS and (d / VERDICTS[exp]).is_file():
        frame = pd.read_csv(d / VERDICTS[exp])
        cols = {c: c for c in frame.columns if c not in ("rule_status", "e016_choice")}
        return f"Вердикты по правилу до запуска (`{VERDICTS[exp]}`):\n\n" + table(frame, cols)
    if exp == "E010" and (d / "results/lr_sweep_summary.csv").is_file():
        s = pd.read_csv(d / "results/lr_sweep_summary.csv")
        counts = s.verdict_draft_rule.value_counts().to_dict()
        better = s[s.verdict_draft_rule == "better than control"]
        text = "Вердикты (`results/lr_sweep_summary.csv`, `verdict_draft_rule`): " + \
               ", ".join(f"{k}: {v}" for k, v in counts.items()) + ".\n\n"
        if len(better):
            text += table(better.head(MAX_ROWS), {"variant": "архитектура", "lr_name": "ветка lr", "budget": "шагов",
                                                  "T95_mean": "T95, шагов", "test_reduction_mean": "тест, %"})
        return text
    if exp == "E011" and (d / "results/long_runs_summary.csv").is_file():
        s = pd.read_csv(d / "results/long_runs_summary.csv")
        return "Плато и переобучение (`results/long_runs_summary.csv`):\n\n" + table(s, {
            "variant": "модель", "lr_branch": "ветка lr", "n_train_nominal": "N (файлов)", "final_test_mean": "тест, финал, %",
            "y_inf_mean_curve": "предел y_inf", "plateau_text": "плато", "overfitting_verdict_draft": "переобучение"})
    if exp == "E019" and (d / "results/confirm_summary.csv").is_file():
        s = pd.read_csv(d / "results/confirm_summary.csv")
        return "Длинные прогоны (`results/confirm_summary.csv`):\n\n" + table(s, {
            "member": "участник", "lr_branch": "ветка lr", "horizon": "шагов", "n_seeds": "запусков (seeds)",
            "test_valbest_mean": "тест, %", "test_valbest_min": "тест, мин", "test_valbest_max": "тест, макс",
            "valbest_q2_gain_db_mean": "Q², дБ", "macs_per_symbol_merged": "умножений", "plateau_text": "плато"})
    if exp == "E015" and (d / "results/verdict.json").is_file():
        v = json.loads((d / "results/verdict.json").read_text(encoding="utf-8"))
        return ("Вердикт (`results/verdict.json`): **" + v["verdict"] + "**.\n\n" +
                "\n".join(f"- {VERDICT_KEYS_RU.get(k, k)}: {fmt(val) if not isinstance(val, dict) else ', '.join(f'{a}: {fmt(b, 3)}' for a, b in val.items())}"
                          for k, val in v.items() if k not in ("verdict", "rule_status")) + "\n")
    return "Вердиктов нет (эксперимент не запускался).\n"


CONFIRMED = ("подтверждено", "лучше контроля", "гибрид улучшает", "трансформер превосходит", "память есть",
             "второй слой возвращает", "эффект оптимизации", "плато (", "KAN без Керр-основы лучше",
             "трактуется как переобучение")


def normalize(verdict: str) -> str:
    text = VALUE_RU.get(str(verdict), str(verdict))
    if text.startswith("не доказано") or text.startswith("переобучение не показано") or "не лучше" in text:
        return "не доказано"
    return "подтверждено" if any(text.startswith(c) or c in text for c in CONFIRMED) else "не доказано"


def verdict_items(exp: str) -> list[tuple[str, str]] | None:
    """(hypothesis label, normalized verdict) pairs from the experiment's verdict files; None if not run."""
    d = folder(exp)
    if exp in VERDICTS and (d / VERDICTS[exp]).is_file():
        frame = pd.read_csv(d / VERDICTS[exp])
        if exp == "E012":
            part = frame.drop_duplicates(["lr_branch", "window"])
            items = [(f"{b}, окно {int(w)}: второй слой", v) for b, w, v in
                     zip(part.lr_branch, part.window, part.window_verdict)]
            items += [(f"{b}, окно {int(w)}: эффект оптимизации", v) for b, w, v in
                      zip(part.lr_branch, part.window, part.optimization_effect) if isinstance(v, str)]
        elif exp == "E013":
            items = [(f"{m}, {int(st) // 1000}k", v) for m, st, v in zip(frame.model, frame.steps, frame.verdict)]
        elif exp == "E014":
            items = [(f"{v}, G={int(g)}", r) for v, g, r in zip(frame.variant, frame.grid, frame.verdict)]
        else:
            items = list(zip(frame.hypothesis, frame.verdict))
        return [(label, normalize(v)) for label, v in items] or None   # empty verdict file: not run yet
    if exp == "E010" and (d / "results/lr_sweep_summary.csv").is_file():
        s = pd.read_csv(d / "results/lr_sweep_summary.csv")
        s = s[s.verdict_draft_rule != "control"]
        return [(f"{a} · {lr} · {b}", normalize(v)) for a, lr, b, v in
                zip(s.variant, s.lr_name, s.budget, s.verdict_draft_rule)]
    if exp == "E011" and (d / "results/long_runs_summary.csv").is_file():
        s = pd.read_csv(d / "results/long_runs_summary.csv")
        items = [(f"плато: {m} · {b} · N={n}", normalize(t)) for m, b, n, t in
                 zip(s.variant, s.lr_branch, s.n_train_nominal, s.plateau_text)]
        pairs = s.drop_duplicates(["variant", "lr_branch"])
        items += [(f"переобучение: {m} · {b}", normalize(v)) for m, b, v in
                  zip(pairs.variant, pairs.lr_branch, pairs.overfitting_verdict_draft)]
        return items
    if exp == "E019" and (d / "results/confirm_summary.csv").is_file():
        s = pd.read_csv(d / "results/confirm_summary.csv")
        s = s[s.horizon == s.horizon.max()]
        return [(f"плато за {int(h) // 1000}k: {m} · {b}", normalize(t)) for m, b, h, t in
                zip(s.member, s.lr_branch, s.horizon, s.plateau_text)]
    if exp == "E015" and (d / "results/verdict.json").is_file():
        v = json.loads((d / "results/verdict.json").read_text(encoding="utf-8"))["verdict"]
        return [("память есть (фаза зависит от мощности соседей)", normalize(v))]
    return None


def short_verdict(exp: str) -> str:
    items = verdict_items(exp)
    if items is None:
        return "не запущено"
    confirmed = sum(1 for _, v in items if v == "подтверждено")
    return f"подтверждено: {confirmed}, не доказано: {len(items) - confirmed}"


def verdict_summary_block(exp: str) -> str:
    items = verdict_items(exp)
    if items is None:
        return "**Вердикт:** не запущено.\n"
    lines = [f"**Вердикт** (по правилу до запуска): {short_verdict(exp)}.", ""]
    lines += [f"- {label}: **{verdict}**" for label, verdict in items[:MAX_ROWS]]
    if len(items) > MAX_ROWS:
        lines.append(f"- … ещё {len(items) - MAX_ROWS} (полностью — в таблице ниже)")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- pages

def build_experiment(exp: str) -> None:
    d = folder(exp)
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    rule = config.get("rule_draft") or {"interpretation_rule": config.get("interpretation_rule", "unknown")}
    write(f"{exp}_header.md", f"Журнал запусков: {log_link()} (только дописывается). "
                              f"Папка эксперимента: `{d.relative_to(ROOT).as_posix()}`.\n")
    write(f"{exp}_rule.md", f"**Вопрос.** {question(exp)}\n\n**Правило до запуска** "
                            f"(`{d.relative_to(ROOT)}/config.json` → `rule_draft`, дословно):\n\n"
                            + "\n".join(render_value(rule)) + "\n")

    parts = [verdict_summary_block(exp)]
    if exp == "E015":
        fits = d / "results/alpha_fit_summary.csv"
        if fits.is_file():
            f = pd.read_csv(fits)
            f["ci"] = [f"{fmt(a)} … {fmt(b)}" for a, b in zip(f.reduction_ci_low, f.reduction_ci_high)]
            parts.append("Прямая подгонка фазы (`results/alpha_fit_summary.csv`):\n\n" + table(f, {
                "n_alpha": "R (коэффициентов)", "split": "выборка", "reduction": "снижение BER, %", "ci": "95 % бутстрап",
                "q2_gain_db": "Q², дБ"}))
        diag = d / "results/diagnostics_summary.csv"
        if diag.is_file():
            g = pd.read_csv(diag)
            g = g[(g.metric == "spearman_dphi") & (g.ring == "all")]
            g["ci"] = [f"{fmt(a, 3)} … {fmt(b, 3)}" for a, b in zip(g.ci_low, g.ci_high)]
            parts.append("Спирмен ρ(dphi, мощность), все окружности (`results/diagnostics_summary.csv`):\n\n" +
                         table(g.assign(value=g.value.round(3)), {"split": "выборка", "predictor": "предиктор",
                                                                  "value": "ρ", "ci": "95 % бутстрап"}))
    elif exp == "E019":
        pass
    else:
        result = grouped(exp)
        parts.append(results_table(result[1]) if result else "Результатов нет (эксперимент не запускался).\n")
    for rel, columns, caption in EXTRA_TABLES.get(exp, []):
        if (d / rel).is_file():
            frame = pd.read_csv(d / rel)
            if len(frame):
                parts.append(f"{caption} (`{rel}`):\n\n" + table(frame.head(40), columns))
    parts.append(verdict_block(exp))
    parts.append("Рисунки (клик — крупно, листание стрелками):\n\n" + figures(exp, depth=1))
    write(f"{exp}_results.md", "\n".join(parts))

    control_text, control_mean = control_block(exp)
    checks = [control_text]
    if exp in RUNS and grouped(exp):
        runs, agg = grouped(exp)
        checks.append("**Аномалии** (пороги: хуже контроля > 5 п.п., разброс seeds > 3 п.п., "
                      "|валидация − тест| > 5 п.п.):\n\n" + anomalies(runs, agg, control_mean, exp in CONTROL_IS_OTHER_MODEL))
    elif exp == "E015":
        checks.append("Аномалии по прогонам неприменимы: обучения нет.\n")
    elif exp == "E019" and (d / "results/confirm.csv").is_file():
        c = pd.read_csv(d / "results/confirm.csv")
        agg = c.groupby(["member", "lr_branch", "nominal_horizon"]).agg(
            val_mean=("best_val_reduction", "mean"), test_mean=("test_reduction_valbest", "mean"),
            test_min=("test_reduction_valbest", "min"), test_max=("test_reduction_valbest", "max")).reset_index()
        agg["config"] = agg[["member", "lr_branch", "nominal_horizon"]].astype(str).agg(" · ".join, axis=1)
        agg["spread"], agg["is_control"] = agg.test_max - agg.test_min, agg.member.str.startswith("control")
        checks.append("**Аномалии** (в `confirm.csv` нет поля `diverged`):\n\n" +
                      anomalies(c.assign(diverged=False, run_id=c.run_id), agg, None))
    write(f"{exp}_checks.md", "\n".join(checks))


def build_index(experiments: list[str]) -> None:
    rows = []
    for exp in experiments:
        path = folder(exp) / "results" / "control_check.json"
        control = "—"
        if path.is_file():
            check = json.loads(path.read_text(encoding="utf-8"))
            control = ("в допуске" if check["passed"] else "ВНЕ ДОПУСКА") + ": " + ", ".join(
                f"{fmt(t['mean'])} %" for t in check["targets"])
        elif exp == "E015" and (folder(exp) / "results/diagnostics_summary.csv").is_file():
            d = pd.read_csv(folder(exp) / "results/diagnostics_summary.csv")
            ok = d[d.metric == "control_raw_test_ber"].ok.iloc[0]
            control = "raw BER совпадает с E004" if str(ok) == "True" else "ВНЕ ДОПУСКА"
        rows.append({"exp": f"[{exp}](experiments/{exp}.qmd)", "question": question(exp),
                     "verdict": short_verdict(exp), "control": control})
    text = table(pd.DataFrame(rows), {"exp": "эксп.", "question": "вопрос", "verdict": "вердикт (по правилу)",
                                      "control": "контроль"})
    write("index_status.md", text + "\nE001–E009 — [одной страницей](experiments/legacy.qmd). "
                                    f"История запусков — {log_link()}.\n")


def build_legacy() -> None:
    log = (ROOT / "LOG.md").read_text(encoding="utf-8")
    rows = []
    for match in re.finditer(r"^## (E00[1-9]) — (.+?)(?: — .*)?$\n(.*?)(?=^## |\Z)", log, flags=re.M | re.S):
        body = match.group(3)
        q = re.search(r"^Вопрос:\s*(.+)$", body, flags=re.M)
        v = re.search(r"^Вывод:\s*(.*)$", body, flags=re.M)
        rows.append({"exp": match.group(1), "title": match.group(2).strip(), "question": q.group(1).strip() if q else "—",
                     "conclusion": (v.group(1).strip() if v else "") or "—"})
    write("legacy.md", f"Строки «Вопрос» и «Вывод» из `LOG.md` дословно ({log_link()}).\n\n" +
          table(pd.DataFrame(rows), {"exp": "эксп.", "title": "название", "question": "вопрос", "conclusion": "вывод"}))


def manifest_rows() -> list[dict]:
    path = ROOT / "paper" / "figures_manifest.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: float(r["fig_no"]) if r.get("fig_no", "").strip() else float("inf"))
    return rows


def build_gallery() -> None:
    seen, groups = set(), {}
    for row in manifest_rows():
        path = ROOT / row["file"].strip()
        if path.is_file():
            seen.add(path.resolve())
            groups.setdefault(row["experiment_id"], []).append(path)
    for path in sorted(EXPERIMENTS.glob("E*/figures/*.png")):
        if path.resolve() not in seen:
            groups.setdefault(path.parts[-3][:4], []).append(path)
    parts = []
    for exp, paths in groups.items():
        parts.append(f"## {exp}\n\n::: {{layout-ncol=3}}\n")
        for p in paths:
            caption = f"{exp} · {p.name} · {git_date(p)}"
            parts.append(f"![{caption}]({figure_link(p, 0)}){{group=\"gallery\"}}\n")
        parts.append(":::\n")
    write("gallery.md", "\n".join(parts))


def build_slides(with_titles: bool) -> None:
    parts = []
    for row in manifest_rows():
        path = ROOT / row["file"].strip()
        if not path.is_file():
            continue
        heading = (row.get("title") or "").strip() or path.name
        parts.append(("## " + heading + "\n\n" if with_titles else "---\n\n") + f"![]({figure_link(path, 0)})\n")
    write("slides.md", "\n".join(parts))


def main(argv: list[str]) -> int:
    experiments = sorted(p.name[:4] for p in EXPERIMENTS.glob("E0*_*") if p.is_dir() and p.name[:4] >= "E010")
    selected = [e for e in argv if re.fullmatch(r"E0\d\d", e)] or experiments
    for exp in selected:
        build_experiment(exp)
        print(f"{exp}: _auto/{exp}_rule.md, _results.md, _checks.md")
    build_index(experiments)
    build_legacy()
    build_gallery()
    build_slides(os.environ.get("JOURNAL_TITLES", "0") == "1")
    print(f"index_status.md, legacy.md, gallery.md, slides.md -> {AUTO.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
