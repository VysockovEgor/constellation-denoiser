# E011 — Длинные прогоны и сходимость

**Вопрос.** Выходят ли complex_both/a/one_hidden и real_tanh на плато за 200k шагов при N = 30 и 210?

**Правило (до запуска, черновик).** плато — прирост сглаженной валидации за 50k < 0.2 п.п.; переобучение — только если при N=30 кривая падает, а при N=210 нет. Полностью — `config.json` → `rule_draft`.

**Команда.** `bash experiments/E011_long_runs_convergence/run.sh <стадия>`; стадии: probe | plan <stage|all> | list <stage> | control | const | best | summarize | smoke.
Проверка без сервера: `SMOKE=1 bash experiments/E011_long_runs_convergence/run.sh smoke`.

**Результаты.** `results/long_runs_summary.csv`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
