<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** —

**Правило до запуска** (`experiments/E051_am_liang_kumar/config.json` → `rule_draft`, дословно):

- `lk_vs_a1`: LK beats A1 (or A3) if, at the same (M, N_c, set) and the same or lower RM, mean test difference over 5 replicas > 2 SE AND > 2 std (larger of the two); on the best configuration by validation also on the 3 E023 splits (same sign on all 3)
- `pruning`: cross beats the hyperbola if at equal number of terms (nearest K) the test reduction is higher by > 2 SE and > 2 std; otherwise 'не доказано'
- `otherwise`: не доказано
- `limits`: one channel, one polarization, 1 sps, 5 bootstrap replicas of the train files (closed form, not training seeds), one stage only
