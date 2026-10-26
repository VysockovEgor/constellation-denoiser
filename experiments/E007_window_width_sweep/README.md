# E007 — Окно и ширина complex MLP

**Вопрос.** Сравнить окна 31/65/127 и ширину 32/48 комплексной MLP (N=30, 20k, seeds 17, 29).

**Правило (до запуска).** `unknown` — исторический эксперимент, правило не сохранилось
(`config.json` → `interpretation_rule`).

**Команда.** `bash experiments/E007_window_width_sweep/run.sh --help` — исторический запуск; `run.sh`
отказывается перезаписывать сохранённые результаты.

**Результаты.** `results/complex_mlp_sweep_summary.csv`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
