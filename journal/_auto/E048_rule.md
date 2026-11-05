<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Догоняет ли архитектура A настроенный DBP при большей памяти (M = 11, 15; N_c = 31, 63), и помогают ли DBP внеосевые триплеты там, где осевые триплеты E038 дублировали его фазу Керра?

**Правило до запуска** (`experiments/E048_kerr_offaxis_dbp/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `a_wins`: as E046: A beats triplets-only and Kerr-only at equal RM by > 2 std
- `offaxis_adds`: off-axis triplets add to DBP if (DBP + K off-axis) - DBP alone >= 1 p.p. with non-overlapping intervals (as E038)
- `a_vs_dbp`: A beats DBP + off-axis at an RM level if its test mean is higher by > 2 std (the larger std of the two)
- `otherwise`: не доказано
