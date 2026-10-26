# E002 — tanh sweep и paper-like MLP

**Вопрос.** Проверить tanh-MLP sweep и paper-like вещественную и комплексную MLP на PR9–PR10.

**Правило (до запуска).** `unknown` — исторический эксперимент, правило не сохранилось
(`config.json` → `interpretation_rule`).

**Команда.** `bash experiments/E002_tanh_complex_paper/run.sh --help` — исторический запуск; `run.sh`
отказывается перезаписывать сохранённые результаты.

**Результаты.** `results/tanh_mlp_sweep_summary.csv`, `results/paper_real_mlp.csv`, `results/paper_complex_mlp.csv`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
