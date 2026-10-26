# E006 — Плато и разброс по seed

**Вопрос.** Плато complex MLP при N = 7/210 до 100 000 шагов; покрытие seeds неполное.

**Правило (до запуска).** `unknown` — исторический эксперимент, правило не сохранилось
(`config.json` → `interpretation_rule`).

**Команда.** `bash experiments/E006_plateau_seeds/run.sh --help` — исторический запуск; `run.sh`
отказывается перезаписывать сохранённые результаты.

**Результаты.** `results/plateau_seed_study.csv`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
