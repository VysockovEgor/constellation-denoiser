"""E024: can the numbers of E010-E021 be trusted? Data leakage, data table, tuning budget, noise floor.

    PYTHONPATH=src python -m constellation_denoiser.audit_e024 [leakage|data|budget|noise|all]
    (SMOKE=1: 2 train, 1 validation, 1 test file; writes results_smoke/)

leakage  share of test (validation) windows of K symbol labels whose exact sequence occurs in the train
         files, against the random share 1 - exp(-N_train_windows / 16^K); cross-correlation of the label
         sequences of the first files (max match over all cyclic shifts; random 1/16); rule in config.json.
         Leakage -> results/gate_status.json halt (the queue stops) and run.sh trains the dedup check.
data     per-file table: symbols, raw BER recomputed from rx and the labels, tx/rx power, tx levels;
         raw BER per split (window 31 centres, as the runner) against the recorded value; HD-FEC share.
budget   per family: distinct lr recipes, c_mult values, step budgets, seeds (E010-E022 per-run tables).
noise    seed spread per configuration (E019, E011, E012): std, min-max; minimal detectable difference.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

from .core import BIT_TABLE, CONSTELLATION
from .paths import EXPERIMENTS_DIR, PROJECT_ROOT
from .q2 import q2_db
from .runner import atomic_write_json, exp_dir, results_root

HD_FEC = 3.8e-3


def labels_of(tx_c: np.ndarray) -> np.ndarray:
    """(F, L) complex symbols -> (F, L) uint8 constellation indices (nearest point)."""
    tx_c = np.asarray(tx_c)
    flat = tx_c.reshape(-1)
    index = np.empty(flat.shape, dtype=np.uint8)
    for start in range(0, len(flat), 1 << 18):
        part = flat[start:start + (1 << 18)]
        index[start:start + len(part)] = np.abs(part[:, None] - CONSTELLATION[None, :]).argmin(axis=1)
    return index.reshape(tx_c.shape)


def label_windows(row: np.ndarray, k: int) -> list[bytes]:
    """Byte keys of all windows of ``k`` consecutive labels (window j = labels j .. j + k - 1)."""
    view = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(row.astype(np.uint8), k))
    return [bytes(v) for v in view]


def window_set(labels: np.ndarray, k: int) -> set:
    seen = set()
    for row in labels:
        seen.update(label_windows(row, k))
    return seen


def overlap_share(query: np.ndarray, reference: set, k: int) -> tuple[int, int]:
    hits = total = 0
    for row in query:
        keys = label_windows(row, k)
        hits += sum(key in reference for key in keys)
        total += len(keys)
    return hits, total


def max_cyclic_match(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Max over cyclic shifts s of the share of positions with a[i] == b[i + s] (one-hot FFT correlation)."""
    n = len(a)
    total = np.zeros(n)
    for symbol in range(16):
        fa = np.fft.rfft((a == symbol).astype(float))
        fb = np.fft.rfft((b == symbol).astype(float))
        total += np.fft.irfft(np.conj(fa) * fb, n)
    return float(total.max() / n), int(total.argmax())


def split_files_for(config: dict, smoke: bool) -> dict:
    from .runner import default_files

    return default_files(config["n_train_files"], smoke)


def load_split(files: dict):
    from .window_data import load_fileset, resolve_paths

    return {name: load_fileset(resolve_paths(list(names))) for name, names in files.items()}


# --------------------------------------------------------------------------- 1. leakage

def leakage(config: dict, smoke: bool, out) -> dict:
    files = split_files_for(config, smoke)
    data = load_split(files)
    labels = {name: labels_of(fs.tx_c) for name, fs in data.items()}
    rows = []
    for k in config["leakage"]["k"]:
        train = window_set(labels["train"], k)
        n_train = sum(len(r) - k + 1 for r in labels["train"])
        expected = 1.0 - math.exp(-n_train / 16.0 ** k)
        for query in ("test", "validation"):
            hits, total = overlap_share(labels[query], train, k)
            rows.append({"k": k, "pair": f"{query}->train", "windows": total, "matched": hits,
                         "share": hits / total, "expected_random": expected, "train_windows": n_train,
                         "train_unique_windows": len(train), "ratio_to_random": (hits / total) / expected if expected else float("inf")})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "leakage_windows.csv", index=False)

    names = files["train"][:config["leakage"]["xcorr_files"]] + files["test"][:2] + files["validation"][:2]
    seqs = labels["train"][:config["leakage"]["xcorr_files"]].tolist() + labels["test"][:2].tolist() + \
        labels["validation"][:2].tolist()
    corr = []
    for i in range(len(seqs)):
        for j in range(i, len(seqs)):
            a, b = np.asarray(seqs[i]), np.asarray(seqs[j])
            if i == j:
                n = len(a)
                total = np.zeros(n)                              # largest match at a non-zero shift
                for symbol in range(16):
                    f = np.fft.rfft((a == symbol).astype(float))
                    total += np.fft.irfft(np.conj(f) * f, n)
                total[0] = -1
                share, shift = float(total.max() / n), int(total.argmax())
            else:
                share, shift = max_cyclic_match(a, b)
            corr.append({"file_a": names[i], "file_b": names[j], "self": i == j, "max_match_share": share,
                         "best_shift": shift})
    corr = pd.DataFrame(corr)
    corr.to_csv(out / "leakage_xcorr.csv", index=False)

    rule = config["leakage"]
    row = frame[(frame.k == rule["rule_k"]) & (frame.pair == "test->train")].iloc[0]
    no_leak = bool(row.share <= rule["max_ratio_to_random"] * row.expected_random and row.share < rule["max_share"])
    xcorr_max = float(corr[~corr.self].max_match_share.max()) if (~corr.self).any() else float("nan")
    verdict = {"rule": rule["rule_text"], "k": int(row.k), "share_test_to_train": float(row.share),
               "expected_random": float(row.expected_random), "matched_windows": int(row.matched),
               "no_leakage": no_leak, "verdict": "утечки нет" if no_leak else "УТЕЧКА",
               "max_cross_file_match": xcorr_max, "random_match": 1 / 16,
               "files": {k: len(v) for k, v in files.items()}}
    atomic_write_json(out / "leakage.json", verdict)
    if not no_leak and not smoke:
        from .gate import set_halt

        set_halt("E024: утечка по последовательностям", f"K={row.k}: {row.share:.3g} при случайной {row.expected_random:.3g}")
    plot_leakage(frame, out_fig(out, smoke))
    return verdict


# --------------------------------------------------------------------------- 2. data table

def data_table(config: dict, smoke: bool, out) -> dict:
    from .experiment import ber_from_iq

    files = split_files_for(config, smoke)
    data = load_split(files)
    rows = []
    for split, fs in data.items():
        for f in range(fs.n_files):
            tx, rx = fs.tx_c[f], fs.rx_c[f]
            rows.append({"split": split, "file": fs.names[f], "symbols": len(tx),
                         "raw_ber_all_symbols": ber_from_iq(np.column_stack((rx.real, rx.imag)), tx),
                         "tx_power": float(np.mean(np.abs(tx) ** 2)), "rx_power": float(np.mean(np.abs(rx) ** 2)),
                         "tx_levels": ",".join(f"{v:.4f}" for v in np.unique(np.round(np.abs(np.r_[tx.real, tx.imag]), 4))),
                         "evm_raw": float(np.sqrt(np.mean(np.abs(rx - tx) ** 2) / np.mean(np.abs(tx) ** 2)))})
    table = pd.DataFrame(rows)
    table.to_csv(out / "data_files.csv", index=False)
    window = 31
    splits = {}
    for split, fs in data.items():
        splits[split] = {"files": fs.n_files, "raw_ber_window31_centres": fs.raw_ber(window),
                         "raw_ber_all_symbols": float(np.mean(table[table.split == split].raw_ber_all_symbols))}
    recorded = float(config["recorded_raw_test_ber"])
    raw = splits["test"]["raw_ber_window31_centres"]
    summary = {"splits": splits, "recorded_raw_test_ber": recorded,
               "recomputed_equals_recorded": bool(abs(raw - recorded) < 1e-12) if not smoke else None,
               "raw_q2_db": q2_db(raw), "hd_fec_ber": HD_FEC,
               "hd_fec_reduction_pct": 100 * (raw - HD_FEC) / raw, "hd_fec_q2_gain_db": q2_db(HD_FEC) - q2_db(raw),
               "tx_power_range": [float(table.tx_power.min()), float(table.tx_power.max())],
               "rx_power_range": [float(table.rx_power.min()), float(table.rx_power.max())],
               "tx_levels_unique": sorted(set(",".join(table.tx_levels).split(","))),
               "normalisation_note": "tx levels ±1/√10, ±3/√10 in every file if tx_levels_unique has 2 values; "
                                     "rx power per file in rx_power_range",
               "metadata": config["metadata_note"]}
    atomic_write_json(out / "data_summary.json", summary)
    plot_raw_ber(table, out_fig(out, smoke))
    return summary


# --------------------------------------------------------------------------- 4. tuning budget

def tuning_budget(out, smoke: bool) -> pd.DataFrame:
    from .pareto_v2 import family_of, load_tables

    runs = load_tables()
    runs["family_v2"] = family_of(runs)
    lr_cols = ["lr_peak_lr", "lr_schedule", "lr_warmup", "lr_c_mult", "lr_weight_decay", "lr_optimizer"]
    have = [c for c in lr_cols if c in runs]
    runs["recipe"] = [("|".join(str(r[c]) for c in have) if pd.notna(r.get("lr_peak_lr")) else str(r.get("lr_name")))
                      for _, r in runs.iterrows()]
    rows = []
    for family, part in runs.groupby("family_v2"):
        c_mult = sorted({float(v) for v in part.get("lr_c_mult", pd.Series(dtype=float)).dropna()} |
                        {float(n.rsplit("cm", 1)[1]) for n in part.lr_name.dropna().astype(str) if "_cm" in n})
        seeds = part.groupby(["source", "variant", "recipe", "budget"]).seed.nunique()
        rows.append({"family": family, "runs": len(part), "lr_recipes": part.recipe.nunique(),
                     "c_mult_values": ",".join(f"{v:g}" for v in c_mult) or "—",
                     "c_mult_explored": len(c_mult) > 1, "step_budgets": ",".join(str(int(b)) for b in sorted(part.budget.dropna().unique())),
                     "max_steps": int(part.budget.max()), "max_seeds_per_config": int(seeds.max()),
                     "sources": ",".join(sorted(part.source.unique()))})
    frame = pd.DataFrame(rows).sort_values("lr_recipes", ascending=False)
    frame.to_csv(out / "tuning_budget.csv", index=False)
    plot_budget(frame, out_fig(out, smoke))
    return frame


# --------------------------------------------------------------------------- 5. noise floor

def pp_to_db(reduction: float, raw: float, delta: float = 1.0) -> float:
    """Q2 change (dB) of +delta p.p. of BER reduction around ``reduction`` %."""
    return q2_db(raw * (1 - (reduction + delta) / 100)) - q2_db(raw * (1 - reduction / 100))


def noise_floor(config: dict, out, smoke: bool) -> dict:
    from scipy.stats import t as student

    sources = []
    confirm = EXPERIMENTS_DIR / "E019_confirm_long_runs" / "results" / "confirm.csv"
    if confirm.is_file():
        c = pd.read_csv(confirm)
        sources.append(("E019", c.assign(config=c.member + "|" + c.lr_name + "|" + c.nominal_horizon.astype(str)),
                        "test_reduction_valbest"))
    for folder, name, keys in (("E011_long_runs_convergence", "long_runs_q2.csv", ["variant", "lr_name", "n_train_nominal"]),
                               ("E012_depth_vs_window", "depth_window_q2.csv", ["variant", "window", "lr_name", "budget"])):
        path = EXPERIMENTS_DIR / folder / "results" / name
        if path.is_file():
            f = pd.read_csv(path)
            sources.append((folder[:4], f.assign(config=f[keys].astype(str).agg("|".join, axis=1)),
                            "test_reduction_valbest_select"))
    rows = []
    for source, frame, metric in sources:
        for cfg, part in frame.groupby("config"):
            values = part[metric].dropna()
            if len(values) >= 2:
                rows.append({"source": source, "config": cfg, "n_seeds": len(values), "mean": values.mean(),
                             "std": values.std(ddof=1), "min": values.min(), "max": values.max(),
                             "range": values.max() - values.min()})
    table = pd.DataFrame(rows)
    table.to_csv(out / "noise_floor_configs.csv", index=False)
    five = table[table.n_seeds >= 5]
    pooled = float(np.sqrt(np.mean(five["std"] ** 2))) if len(five) else float(np.sqrt(np.mean(table["std"] ** 2)))
    n = 5
    t975 = float(student.ppf(0.975, 2 * n - 2))
    mdd = t975 * pooled * math.sqrt(2 / n)
    raw = float(config["recorded_raw_test_ber"])
    result = {"pooled_std_pp": pooled, "n_configs_5_seeds": int(len(five)), "n_configs_all": int(len(table)),
              "median_range_pp_5_seeds": float(five["range"].median()) if len(five) else float("nan"),
              "max_range_pp_5_seeds": float(five["range"].max()) if len(five) else float("nan"),
              "mdd_pp_5_seeds": mdd,
              "formula": "MDD = t_{0.975, 2n-2} * s_pooled * sqrt(2/n), n = 5 seeds per arm, s_pooled = sqrt(mean s_i^2) "
                         "over configurations with 5 seeds (E019)",
              "mdd_db_at_reduction": {str(r): pp_to_db(r, raw, mdd) for r in (10, 48, 66, 72)},
              "db_per_pp_at_reduction": {str(r): pp_to_db(r, raw) for r in (10, 48, 66, 72)},
              "file_bootstrap": "not available: per-file BER is not stored in the run JSONs",
              "raw_test_ber": raw}
    atomic_write_json(out / "noise_floor.json", result)
    if not smoke:
        atomic_write_json(PROJECT_ROOT / "results" / "noise_floor.json", result)
    return result


# --------------------------------------------------------------------------- figures

def out_fig(out, smoke: bool):
    path = out / "figures" if smoke else exp_dir("E024") / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_leakage(frame: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for index, (pair, part) in enumerate(frame.groupby("pair")):
        ax.plot(part.k, part.share.clip(lower=1e-12), marker="os"[index % 2], color=plot_style.color(index), label=pair)
    expected = frame.drop_duplicates("k")
    ax.plot(expected.k, expected.expected_random.clip(lower=1e-12), "--", color=plot_style.TEXT, label="random 1-exp(-N/16^K)")
    ax.set_yscale("log")
    ax.set_xlabel("K, symbols in the window")
    ax.set_ylabel("share of windows found in train")
    ax.set_title("Sequence overlap test/validation -> train (N = 30 train files)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "E024_window_overlap.png", dpi=160)
    plt.close(fig)


def plot_raw_ber(table: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    fig, ax = plt.subplots(figsize=(9, 4))
    for index, (split, part) in enumerate(table.groupby("split")):
        ax.plot(range(len(part)), part.raw_ber_all_symbols.to_numpy(), "o^s"[index % 3], color=plot_style.color(index),
                markersize=4, label=f"{split} ({len(part)} files)", linestyle="")
    ax.axhline(HD_FEC, color=plot_style.TEXT, linestyle="--", linewidth=1, label="HD-FEC 3.8e-3")
    ax.set_xlabel("file index within the split")
    ax.set_ylabel("raw BER (all symbols)")
    ax.set_title("Raw BER per file")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "E024_raw_ber_per_file.png", dpi=160)
    plt.close(fig)


def plot_budget(frame: pd.DataFrame, figures) -> None:
    import matplotlib.pyplot as plt

    from . import plot_style

    plot_style.apply()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(frame))
    ax.bar(x, frame.lr_recipes, color=[plot_style.SERIES[i % len(plot_style.SERIES)] for i in range(len(frame))],
           hatch=None, edgecolor=plot_style.TEXT, linewidth=0.4)
    for i, (r, s) in enumerate(zip(frame.lr_recipes, frame.max_steps)):
        ax.text(i, r, f"{int(s) // 1000}k", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, frame.family, rotation=25, ha="right")
    ax.set_ylabel("distinct lr recipes tried")
    ax.set_title("Tuning budget per family in E010-E022 (label: longest run)")
    fig.tight_layout()
    fig.savefig(figures / "E024_tuning_budget.png", dpi=160)
    plt.close(fig)


def main(argv: list[str]) -> int:
    smoke = os.environ.get("SMOKE", "0") == "1"
    parts = argv or ["all"]
    config = json.loads((exp_dir("E024") / "config.json").read_text(encoding="utf-8"))
    out = results_root("E024", smoke)
    out.mkdir(parents=True, exist_ok=True)
    selected = {"leakage", "data", "budget", "noise", "pareto"} if "all" in parts else set(parts)
    if "leakage" in selected:
        v = leakage(config, smoke, out)
        print(f"утечка: {v['verdict']} (K={v['k']}: {v['share_test_to_train']:.3g} при случайной {v['expected_random']:.3g}); "
              f"max совпадение между файлами {v['max_cross_file_match']:.4f} (случайно 1/16 = 0.0625)")
    if "data" in selected:
        s = data_table(config, smoke, out)
        print(f"сырой BER test {s['splits']['test']['raw_ber_window31_centres']:.6g} (записан {s['recorded_raw_test_ber']}); "
              f"HD-FEC 3.8e-3 = снижение {s['hd_fec_reduction_pct']:.1f} %")
    if "budget" in selected:
        b = tuning_budget(out, smoke)
        print(f"бюджет настройки: {len(b)} семейств -> {out / 'tuning_budget.csv'}")
    if "noise" in selected:
        n = noise_floor(config, out, smoke)
        print(f"шумовой пол: s = {n['pooled_std_pp']:.2f} п.п., MDD(5 seeds) = {n['mdd_pp_5_seeds']:.2f} п.п.")
    if "pareto" in selected:
        from .pareto_v2 import e024_report

        e024_report(config, smoke, out, out_fig(out, smoke))
    write_verdicts(config, out)
    return 0


def write_verdicts(config: dict, out) -> None:
    """results/audit_verdicts.csv from the files written above (whatever parts exist)."""
    rows = []
    if (out / "leakage.json").is_file():
        v = json.loads((out / "leakage.json").read_text(encoding="utf-8"))
        rows.append({"hypothesis": f"утечки по последовательностям нет (K = {v['k']}: ≤ 5 × случайной и < 1 %)",
                     "value": v["share_test_to_train"], "verdict": "подтверждено" if v["no_leakage"] else "не доказано",
                     "detail": f"случайная доля {v['expected_random']:.3g}; max совпадение файлов {v['max_cross_file_match']:.4f}"})
    if (out / "data_summary.json").is_file():
        d = json.loads((out / "data_summary.json").read_text(encoding="utf-8"))
        same = d.get("recomputed_equals_recorded")
        rows.append({"hypothesis": "сырой test BER совпадает с записанным 0.009551859551859552",
                     "value": d["splits"]["test"]["raw_ber_window31_centres"],
                     "verdict": "подтверждено" if same else "не доказано",
                     "detail": f"HD-FEC 3.8e-3 = снижение {d['hd_fec_reduction_pct']:.1f} %, Q² {d['hd_fec_q2_gain_db']:.3f} дБ"})
    if (out / "rm_audit_mismatches.csv").is_file():
        bad = pd.read_csv(out / "rm_audit_mismatches.csv")
        total = len(pd.read_csv(out / "rm_audit.csv"))
        rows.append({"hypothesis": "RM всех таблиц E010–E023 совпадает с независимыми формулами", "value": len(bad),
                     "verdict": "подтверждено" if bad.empty else "не доказано", "detail": f"проверено {total} конфигураций"})
    if (out / "noise_floor.json").is_file():
        n = json.loads((out / "noise_floor.json").read_text(encoding="utf-8"))
        rows.append({"hypothesis": "минимально различимая разница при 5 seeds (MDD), п.п.", "value": n["mdd_pp_5_seeds"],
                     "verdict": "справочно", "detail": n["formula"]})
    frame = pd.DataFrame(rows, columns=["hypothesis", "value", "verdict", "detail"])
    frame["rule_status"] = config["rule_draft"]["status"]
    frame.to_csv(out / "audit_verdicts.csv", index=False)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
