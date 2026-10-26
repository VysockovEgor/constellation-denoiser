# E003 — Большой пилот и аудит данных

**Вопрос.** Большой пилот на уникальных файлах (split 210/45/45) и аудит набора данных.

**Правило (до запуска).** `unknown` — исторический эксперимент, правило не сохранилось
(`config.json` → `interpretation_rule`).

**Команда.** `bash experiments/E003_large_scale_pilot/run.sh --help` — исторический запуск; `run.sh`
отказывается перезаписывать сохранённые результаты.

**Результаты.** `results/large_scale_pilot_summary.csv`, `results/data_audit_all.csv`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
