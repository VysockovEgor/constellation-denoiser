# E019 — Длинные подтверждения (confirm)

**Вопрос.** Выходят ли лучшие конфигурации E016–E018 и complex_a/complex_both на плато за 100k–200k шагов?

**Правило (до запуска, черновик).** плато — как в E011 (прирост за последние 50k < 0.2 п.п.). Полностью — `config.json` → `rule_draft`.

**Команда.** `bash experiments/E019_confirm_long_runs/run.sh <стадия>`; стадии: probe | plan <stage|all> | estimate | list <stage> | control | main | summarize | smoke.
Проверка без сервера: `SMOKE=1 bash experiments/E019_confirm_long_runs/run.sh smoke`.

**Результаты.** `results/confirm_summary.csv`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
