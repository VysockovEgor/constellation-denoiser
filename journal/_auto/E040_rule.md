<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Даёт ли каскад малых триплетных ступеней выигрыш над одношаговыми триплетами (M31, K2048, M51) при равных RM на моих данных?

**Правило до запуска** (`experiments/E040_triplet_cascade/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `cascade_better`: the cascade beats the one-step triplets at RM level K (6k, 12k, 18k; RM within +-5 %) if its test BER reduction is higher by >= max(1.5 p.p., 2 x the seed spread of this experiment's control) and the min-max intervals over seeds do not overlap, on at least 2 of 3 levels; otherwise not distinguishable
- `shared`: reported: shared minus independent C at the same configuration (5 seeds, confirm)
- `otherwise`: не доказано
