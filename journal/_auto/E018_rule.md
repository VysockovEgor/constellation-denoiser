<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Помогает ли глубина Ns шагов [КИХ + нелинейность с памятью] и нужна ли послойная инициализация?

**Правило до запуска** (`experiments/E018_dbp_depth/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `depth_helps`: among layerwise configurations with Ns in {4, 8} and multiplies <= 16032 (complex_both), the one with the best mean validation reduction beats Ns = 1 with the same L, R and mode on test by >= 1 p.p. and by more than the seed spread
- `layerwise_needed`: among layerwise configurations with Ns > 1, the one with the best mean validation reduction beats joint_from_scratch with the same Ns, L, R on test by >= 1 p.p. and by more than the seed spread
- `otherwise`: не доказано
