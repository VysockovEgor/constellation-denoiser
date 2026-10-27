<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Какое семейство моделей даёт лучшее снижение BER и Q² при заданном числе умножений на символ (уровни ≈2k, 6.5k, 16k, 40k)?

**Правило до запуска** (`experiments/E021_pareto/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `selection`: for each family and level the configuration with the best mean validation reduction among its points in the level window
- `better_if`: family X is better than family Y at level K if the difference of mean test reductions is > 1 p.p. and the min-max intervals over seeds do not overlap
- `otherwise`: не различимы
- `budget_note`: points come from different step budgets (20k-200k); the budget is shown on the plot and in the table, the comparison is not adjusted for it
