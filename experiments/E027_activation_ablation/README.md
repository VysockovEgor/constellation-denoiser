# E027 — Абляция активации

**Вопрос.** Даёт ли именно Kerr-активация exp(i·c·|z|²)·z преимущество над другими комплексными активациями при равной ширине?

**Правило (до запуска, черновик).** Kerr лучше активации A при равном h, если разность средних теста ≥ 1 п.п. (≈ 0.035 дБ) и интервалы min–max не пересекаются, иначе не различимы; обучаемость c нужна, если `kerr_c_train` − `kerr_c_fixed` ≥ 0.5 п.п.; нелинейность нужна, если лучшая нелинейная − `identity` ≥ 20 п.п. Полностью — `config.json` → `rule_draft`.

**Активации** (`models_e024.ActivationMLP`, раскладка `complex_one_hidden`; RM на нейрон): `kerr_c_train` 7 (в текущем коде c один на слой, а не на нейрон), `kerr_c_fixed` 7 (c — буфер, медиана обученных c у `kerr_c_train` той же ширины и длины), `kerr_taylor1` z + a|z|²z 8, `modrelu` 5, `crelu` 0, `split_tanh` 0, `cardioid` 6, `sigmoid_gate` 5, `identity` 0. RM_total = 4Wh + 4h + RM_act·h.

**Стадии.** S1: 6 lr-конфигураций (cos 5e-4, 1.5e-3, 3e-3, 6e-3, 1e-2; const 1e-3; для `kerr_c_train`, `kerr_taylor1` ещё `c_mult` 1 и 30) × 2 seeds × 20k, h = 48 → S2: 2 лучших рецепта × h ∈ {16, 48, 119} × 5 seeds × 100k → S3: лучший рецепт, h = 48, 5 seeds × 200k. Стадии `*_fixed` идут после соответствующих (им нужны обученные c).

**Команда.** `bash experiments/E027_activation_ablation/run.sh <стадия>`; стадии: control | s1 | s1_fixed | s2 | s2_fixed | s3 | s3_fixed | all | summarize | smoke.

**Результаты.** `results/activation_summary.csv`, `results/activation_verdicts.csv`; рисунки `figures/E027_*.png`.
