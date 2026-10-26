# E014 — KAN и гибрид Керр + KAN

**Вопрос.** Помогает ли лёгкий KAN (свой, B-сплайны) доочищать остаток после слоя Керра и стоит ли это умножений?

**Правило (до запуска, черновик).** гибрид улучшает основу k0/k0b больше разброса и ≥ 0.5 п.п.; добавка умножений приводится рядом. Полностью — `config.json` → `rule_draft`.

**Команда.** `bash experiments/E014_kerr_kan_hybrid/run.sh <стадия>`; стадии: probe | plan <stage|all> | list <stage> | control | main | summarize | residual | smoke.
Проверка без сервера: `SMOKE=1 bash experiments/E014_kerr_kan_hybrid/run.sh smoke`.

**Результаты.** `results/kan_hybrid_verdicts.csv`, `results/residual_by_point.csv`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
