# E010 — Расписания learning rate

**Вопрос.** Сокращают ли больший пиковый lr, warmup, отдельный lr для c и затухание число шагов до 95 % качества контроля (T95)?

**Правило (до запуска, черновик).** лучше контроля, если T95 ≤ 0.7 контрольного, тест не хуже на 0.5 п.п. и нет расходящихся seeds. Полностью — `config.json` → `rule_draft`.

**Команда.** `bash experiments/E010_lr_schedule_study/run.sh <стадия>`; стадии: probe | plan <stage|all> | list <stage> | baseline-check | stage1 | select | stage2 | stage3.
Проверка без сервера: `SMOKE=1 bash experiments/E010_lr_schedule_study/run.sh smoke`.

**Результаты.** `results/lr_sweep_summary.csv`, `results/stage3_choice.json`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
