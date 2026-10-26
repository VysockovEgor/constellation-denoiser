# E015 — Диагностика физики (без обучения)

**Вопрос.** Зависит ли фазовая ошибка символа от мощности соседей, а не только от собственной мощности?

**Правило (до запуска, черновик).** |ρ(dphi, P_nb)| ≥ |ρ(dphi, |z|²)| хотя бы для одного R и подгонка R=7 лучше R=1 на ≥ 2 п.п.; тест согласуется. Полностью — `config.json` → `rule_draft`.

**Команда.** `bash experiments/E015_physics_diagnostics/run.sh <стадия>`; стадии: diagnostics | smoke | plan | estimate | probe.
Проверка без сервера: `SMOKE=1 bash experiments/E015_physics_diagnostics/run.sh smoke`.

**Результаты.** `results/verdict.json`, `results/diagnostics_summary.csv`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
