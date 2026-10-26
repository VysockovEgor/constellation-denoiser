# E016 — Чисто физическая модель PhysKerr

**Вопрос.** Сколько даёт КИХ + нелинейная фаза с памятью (7–161 умножение) против комплексных MLP на 6–16 тыс.?

**Правило (до запуска, черновик).** «достаточна» — не хуже complex_a на 2 п.п. при ≤ 648 умножений; «полезна» — R > 1 лучше КИХ(31) + одного Керра на ≥ 1 п.п. и больше разброса. Полностью — `config.json` → `rule_draft`.

**Команда.** `bash experiments/E016_phys_kerr_only/run.sh <стадия>`; стадии: probe | plan <stage|all> | estimate | list <stage> | control | main | summarize | smoke.
Проверка без сервера: `SMOKE=1 bash experiments/E016_phys_kerr_only/run.sh smoke`.

**Результаты.** `results/phys_kerr_verdicts.csv`, `results/phys_kerr_summary.csv`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
