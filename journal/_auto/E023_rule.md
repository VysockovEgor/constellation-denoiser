<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Не зависит ли вывод (порядок семейств и разницы) от одного разбиения train / validation / test?

**Правило до запуска** (`experiments/E023_split_robustness/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `robust_if`: the order of the families by mean test reduction and the sign of every pairwise difference are the same in all three splits, and for every family the range of its split means is < 2 p.p.
- `otherwise`: не доказано (вывод зависит от разбиения)
