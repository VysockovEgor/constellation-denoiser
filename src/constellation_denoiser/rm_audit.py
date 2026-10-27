"""E024 part 3: recount real multiplies per symbol (RM) with independent formulas and compare with the
``macs_per_symbol`` / ``macs_per_symbol_merged`` columns of the E010-E023 per-run tables.

    PYTHONPATH=src python -m constellation_denoiser.rm_audit     -> experiments/E024_data_audit/results/rm_audit*.csv

Convention (docs/complexity.md): complex product 4 RM, Kerr activation 7 RM per neuron
(|z|^2 2 + c|z|^2 1 + rotation 4, exp from a table), real activations, additions and biases 0.
KAN and the CNN have no closed form here: their stored value is compared with model_zoo.cost.
"""

from __future__ import annotations

import math
import sys

import pandas as pd

from .paths import EXPERIMENTS_DIR


def phys_step(R: int, L: int, sym: bool) -> float:
    fir = 0 if L <= 1 else (2 * (L + 1) if sym else 4 * L)
    if R <= 0:
        return fir
    return fir + 2 + ((R + 1) / 2 if sym else R) + 4


def transformer(W: int, d: int, layers: int, ffn: int = 2) -> int:
    per_layer = W * (3 * d * d + d * d + 2 * d * ffn * d) + 2 * W * W * d
    return W * 2 * d + layers * per_layer + d * d + 2 * d


def formula(spec: dict) -> tuple[float | None, float | None, str]:
    """(RM as trained, RM after merging, formula text); None where no closed form is used."""
    kind = spec.get("model")
    W = int(spec.get("window", 31))
    h = int(spec["hidden"]) if spec.get("hidden") == spec.get("hidden") and spec.get("hidden") is not None else None
    if kind == "complex_both":
        m = 4 * h * (W + h + 1) + 14 * h
        return m, m, "4h(W+h+1)+14h"
    if kind == "complex_a":
        return 4 * W * h + 4 * h * h + 4 * h + 7 * h, h * (4 * W + 11), "trained 4Wh+4h^2+4h+7h; merged h(4W+11)"
    if kind == "complex_one_hidden":
        return h * (4 * W + 11), h * (4 * W + 11), "h(4W+11)"
    if kind in ("real_tanh",):
        m = 2 * W * h + h * h + 2 * h
        return m, m, "2Wh+h^2+2h"
    if kind == "real_mlp":
        m = 2 * W * h + (h * h if int(spec["layers"]) == 2 else 0) + 2 * h
        return m, m, "2Wh(+h^2)+2h"
    if kind == "transformer":
        m = transformer(W, int(spec["d_model"]), int(spec["layers"]), int(spec.get("ffn_mult", 2)))
        return m, m, "W*2d + L[W(4d^2+2*ffn*d^2)+2W^2d] + d^2+2d"
    if kind in ("phys_kerr", "phys_dbp", "phys_plus_mlp", "dbp_shared", "dbp_mlp", "dbp_noise"):
        R, L, sym = int(spec["n_alpha"]), int(spec["fir_len"]), str(spec["sym"]) in ("True", "true", "1", "1.0")
        n = int(spec.get("n_steps", 1)) if spec.get("n_steps") == spec.get("n_steps") and spec.get("n_steps") else 1
        m = phys_step(R, L, sym) * n
        if kind in ("phys_plus_mlp", "dbp_mlp"):
            m += int(spec["mlp_hidden"]) * (4 * W + 11)
        return m, m, "Ns * [FIR (2(L+1) sym / 4L) + 2 + R' + 4] (+ h(4W+11))"
    if kind == "act_mlp":
        from .models_e024 import ACT_RM

        m = 4 * W * h + 4 * h + ACT_RM[spec["activation"]] * h
        return m, (4 * W if spec["activation"] == "identity" else m), "4Wh+4h+act*h"
    return None, None, "no closed form (compared with model_zoo.cost)"


def rm_linear(spec: dict, rm_total: float) -> float:
    """Multiplies of the linear layers only: rm_total minus the activation / nonlinear-phase part."""
    kind = spec.get("model")
    h = spec.get("hidden")
    if kind == "complex_both":
        return rm_total - 14 * int(h)
    if kind in ("complex_a", "complex_one_hidden"):
        return rm_total - 7 * int(h)
    if kind == "act_mlp":
        from .models_e024 import ACT_RM

        return rm_total - ACT_RM[spec["activation"]] * int(h)
    if kind in ("phys_kerr", "phys_dbp", "phys_plus_mlp", "dbp_shared", "dbp_mlp", "dbp_noise"):
        L, sym = int(spec["fir_len"]), str(spec["sym"]) in ("True", "true", "1", "1.0")
        n = int(spec.get("n_steps") or 1) if spec.get("n_steps") == spec.get("n_steps") else 1
        fir = 0 if L <= 1 else (2 * (L + 1) if sym else 4 * L)
        mlp = 4 * int(spec["window"]) * int(spec["mlp_hidden"]) + 4 * int(spec["mlp_hidden"]) \
            if kind in ("phys_plus_mlp", "dbp_mlp") else 0
        return fir * n + mlp
    return rm_total


def spec_of_row(row: pd.Series) -> dict:
    spec = {k[5:]: v for k, v in row.items() if isinstance(k, str) and k.startswith("spec_") and v == v}
    spec.setdefault("model", row.get("model"))
    if "window" not in spec and row.get("window") == row.get("window"):
        spec["window"] = row.get("window")
    if "hidden" not in spec and row.get("hidden") == row.get("hidden"):
        spec["hidden"] = row.get("hidden")
    return spec


def audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    from . import model_zoo
    from .pareto_v2 import TABLES

    rows = []
    for folder, name in TABLES:
        path = EXPERIMENTS_DIR / folder / "results" / name
        if not path.is_file() or folder >= "E024":
            continue
        frame = pd.read_csv(path)
        for _, row in frame.drop_duplicates([c for c in ("variant", "window", "spec_hidden", "spec_n_steps", "spec_grid")
                                             if c in frame]).iterrows():
            spec = spec_of_row(row)
            trained, merged, text = formula(spec)
            try:
                cost = model_zoo.cost({k: (int(v) if isinstance(v, float) and v.is_integer() and k != "sym" else v)
                                       for k, v in spec.items()})
                zoo = (cost.macs, cost.macs_merged)
            except Exception as exc:     # missing spec fields in an old table
                zoo = (math.nan, math.nan)
                text += f" (model_zoo: {type(exc).__name__})"
            stored = (row.get("macs_per_symbol"), row.get("macs_per_symbol_merged"))
            ref_trained = trained if trained is not None else zoo[0]
            ref_merged = merged if merged is not None else zoo[1]
            ok = all(abs(float(a) - float(b)) < 0.5 for a, b in zip(stored, (ref_trained, ref_merged))
                     if a == a and b == b)
            rows.append({"source": folder[:4], "model": spec.get("model"), "variant": row.get("variant"),
                         "window": spec.get("window"), "stored_rm": stored[0], "stored_rm_merged": stored[1],
                         "formula_rm": trained, "formula_rm_merged": merged, "model_zoo_rm_merged": zoo[1],
                         "formula": text, "match": ok})
    table = pd.DataFrame(rows)
    return table, table[~table.match]


def main(argv: list[str]) -> int:
    import os

    from .runner import results_root

    out = results_root("E024", os.environ.get("SMOKE", "0") == "1")
    out.mkdir(parents=True, exist_ok=True)
    table, bad = audit()
    table.to_csv(out / "rm_audit.csv", index=False)
    bad.to_csv(out / "rm_audit_mismatches.csv", index=False)
    print(f"RM: проверено {len(table)} конфигураций, несовпадений {len(bad)} -> {out / 'rm_audit_mismatches.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
