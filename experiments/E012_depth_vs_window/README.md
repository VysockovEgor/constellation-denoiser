# E012 — Глубина и ширина против окна

**Вопрос.** Возвращает ли второй слой преимущество при окнах 65/127 и 100k шагов (v1–v4 при равных умножениях)?

**Правило (до запуска, черновик).** v1 или v2 выше и v3, и v4 больше разброса seeds и ≥ 0.5 п.п.; сужение разрыва v2 − v3 к 100k — эффект оптимизации. Полностью — `config.json` → `rule_draft`.

**Команда.** `bash experiments/E012_depth_vs_window/run.sh <стадия>`; стадии: probe | plan <stage|all> | list <stage> | control | const | best | summarize | smoke.
Проверка без сервера: `SMOKE=1 bash experiments/E012_depth_vs_window/run.sh smoke`.

**Результаты.** `results/depth_window_verdicts.csv`, `results/merge_check.csv`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
