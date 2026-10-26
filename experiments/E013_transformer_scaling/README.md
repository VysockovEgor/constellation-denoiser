# E013 — Трансформер при сопоставимом бюджете

**Вопрос.** Сохраняется ли преимущество трансформера E001 при N = 30/210 и 20k/100k шагах против комплексной MLP?

**Правило (до запуска, черновик).** превосходит, если при окне 31, N=210 и том же числе шагов лучше комплексной MLP не дороже по умножениям больше разброса и ≥ 0.5 п.п.. Полностью — `config.json` → `rule_draft`.

**Команда.** `bash experiments/E013_transformer_scaling/run.sh <стадия>`; стадии: probe | plan <stage|all> | list <stage> | control | stageA | stageB | summarize | smoke.
Проверка без сервера: `SMOKE=1 bash experiments/E013_transformer_scaling/run.sh smoke`.

**Результаты.** `results/comparison_points.csv`, `results/transformer_verdicts.csv`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
