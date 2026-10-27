<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Какая картина «RM против Q² gain» получается при равной настройке, и что из неё доказано?

**Правило до запуска** (`experiments/E033_pareto_final/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `dominance`: A dominates B if RM_A <= RM_B and Q2_A - Q2_B >= 0.1 dB with non-overlapping min-max intervals
- `selection`: points and envelope chosen by validation; nothing is selected by test
- `otherwise`: не доказано
