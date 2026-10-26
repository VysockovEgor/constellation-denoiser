# E018 — Глубина как шаги DBP

**Вопрос.** Помогает ли глубина Ns шагов [КИХ + нелинейность с памятью] и нужна ли послойная инициализация?

**Правило (до запуска, черновик).** Ns ∈ {4, 8} лучше Ns=1 на ≥ 1 п.п. и больше разброса; послойная лучше joint на ≥ 1 п.п. и больше разброса. Полностью — `config.json` → `rule_draft`.

**Команда.** `bash experiments/E018_dbp_depth/run.sh <стадия>`; стадии: probe | plan <stage|all> | estimate | list <stage> | control | main | summarize | smoke.
Проверка без сервера: `SMOKE=1 bash experiments/E018_dbp_depth/run.sh smoke`.

**Результаты.** `results/dbp_depth_verdicts.csv`, `results/dbp_depth_summary.csv`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
