# E009 — Абляция Керра и слияние слоёв

**Вопрос.** Варианты Керра a–d и проверка слияния линейных слоёв (окно 31, hidden 48).

**Правило (до запуска).** `unknown` — исторический эксперимент, правило не сохранилось
(`config.json` → `interpretation_rule`).

**Команда.** `bash experiments/E009_kerr_ablation/run.sh --help` — исторический запуск; `run.sh`
отказывается перезаписывать сохранённые результаты.

**Результаты.** `results/complex_mlp_ablation_summary.csv`, `results/complex_mlp_ablation_merge_check.csv`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
