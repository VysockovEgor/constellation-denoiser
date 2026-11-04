<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Что меняют у малого каскада (2 ступени, M = 4, K_th = 16, КИХ7; 1 136 RM) общие C, пошаговое обучение, ридж-инициализация и отказ от симметрий?

**Правило до запуска** (`experiments/E047_cascade_variants/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `factor`: a variant differs from v1 if |test mean difference| > 2 std over 5 seeds (std = the larger of the two); sign as measured
- `otherwise`: не доказано
