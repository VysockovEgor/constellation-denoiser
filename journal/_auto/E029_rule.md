<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Окупается ли физическая часть при каждой ширине или только при малой (E020: гибрид h48 cm30 67.6 % против 66.0 % у класса a, 3 против 5 seeds)?

**Правило до запуска** (`experiments/E029_frontend_vs_width/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `pays_off`: the front-end pays off at width h if hybrid - pure >= 1 p.p. (5 seeds, intervals do not overlap)
- `decreasing`: the gain decreases with h if the difference at h = 16 exceeds that at h = 119 by >= 1 p.p.
- `thesis`: a gain only at small h is the thesis 'physics saves multiplies'
- `noise`: thresholds in p.p. are compared with results/noise_floor.json (E024): MDD(5 seeds) = t_{0.975,8} * s_pooled * sqrt(2/5); 0.1 dB of Q2 = 6 p.p. at 10 %, 4.1 at 48 %, 3.0 at 66 %, 2.7 at 70 % (anchors); every table prints both
- `otherwise`: не доказано
