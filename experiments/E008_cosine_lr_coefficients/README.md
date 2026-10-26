# E008 — Cosine lr и коэффициенты c

**Вопрос.** Сравнить const и cosine lr, сохранить диагностику коэффициентов Керра c.

**Правило (до запуска).** `unknown` — исторический эксперимент, правило не сохранилось
(`config.json` → `interpretation_rule`).

**Команда.** `bash experiments/E008_cosine_lr_coefficients/run.sh --help` — исторический запуск; `run.sh`
отказывается перезаписывать сохранённые результаты.

**Результаты.** `results/complex_mlp_lr_summary.csv`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
