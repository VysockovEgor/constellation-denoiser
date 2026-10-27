<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Меняется ли рейтинг, если обучать до плато (в E011 и E018 плато не достигалось за 200k)?

**Правило до запуска** (`experiments/E032_convergence_1m/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `plateau`: plateau reached if the gain 500k -> 1M is < 0.3 p.p. (mean over seeds, validation-best test within each horizon read from the curves)
- `ranking`: ranking stable if the order of the models at 200k and at 1M is the same and no difference changes sign
- `otherwise`: не доказано
