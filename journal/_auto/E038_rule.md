<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Улучшают ли несколько самых сильных триплетов настроенный DBP в области 600–2 500 RM, сильнее, чем триплеты сами по себе?

**Правило до запуска** (`experiments/E038_dbp_triplets/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `adds_to_dbp`: K triplets add to the DBP if (DBP + K) - DBP alone (same Ns, same recipe) >= 1 p.p., intervals do not overlap
- `beats_triplets`: DBP + K beats triplets alone (E031/E034, best by validation with RM <= its RM) by >= 0.1 dB, non-overlapping intervals
- `otherwise`: не доказано
