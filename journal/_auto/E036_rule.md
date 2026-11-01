<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Где останавливается рост качества триплетов T3 + КИХ с памятью M, и сколько членов сохраняют качество большой памяти?

**Правило до запуска** (`experiments/E036_triplets_ceiling/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `memory`: growth continues if M = 51 - M = 31 (E034) >= 1 p.p. (test mean, 5 replicas)
- `pruning`: K = 2048 terms keep >= 95 % of the reduction of the full M = 51 model
- `otherwise`: не доказано
