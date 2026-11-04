<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Если тип каждой ступени выбирать отдельно (фазовая Kerr-ступень, триплетная, обе, пропуск) под общий бюджет RM, лучше ли это равномерного каскада E040 при тех же RM?

**Правило до запуска** (`experiments/E041_mixed_stages/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `mixed_better`: A (mixed) beats the uniform cascade at level K if its S3 best by validation is higher in test by >= max(1.5 p.p., 2 x the seed spread of this experiment's control) with non-overlapping seed intervals, on at least 2 of 3 levels; otherwise A is closed
- `report`: map of stage type by position for the top 10 configurations of S1 per level
- `otherwise`: не доказано
