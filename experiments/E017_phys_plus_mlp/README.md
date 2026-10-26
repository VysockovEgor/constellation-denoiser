# E017 — Физика + малая нейросеть на остаток

**Вопрос.** Сколько нейросети (h = 4–48) нужно на остаток после PhysKerr, frozen и joint?

**Правило (до запуска, черновик).** лучше при равной стоимости (≤ 6480) и complex_a, и E016 на ≥ 1 п.п.; «не нужна», если h=4 отличается от E016 меньше разброса. Полностью — `config.json` → `rule_draft`.

**Команда.** `bash experiments/E017_phys_plus_mlp/run.sh <стадия>`; стадии: probe | plan <stage|all> | estimate | list <stage> | control | main | summarize | smoke.
Проверка без сервера: `SMOKE=1 bash experiments/E017_phys_plus_mlp/run.sh smoke`.

**Результаты.** `results/phys_plus_mlp_verdicts.csv`, `results/frozen_vs_joint.csv`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
