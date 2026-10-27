<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** А: насыщается ли выигрыш от числа шагов Ns в DBP (joint from scratch), и какие L, R нужны? Б: почему послойное обучение в E018 дало ~2 %?

**Правило до запуска** (`experiments/E028_dbp_depth_layerwise/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `saturation`: saturation in Ns: doubling Ns gives < 1 p.p. (mean validation at 100k, same L and R)
- `diagnosis`: the diagnosis is found if some variant of B brings layer-wise to >= 90 % of joint (f) at the same Ns (mean test, 3 seeds); if (b) works, the E018 page gets a note that its layer-wise was a weak control, not a reproduction of the paper recipe
- `shared`: shared is better than independent if the difference is >= 1 p.p.
- `noise`: thresholds in p.p. are compared with results/noise_floor.json (E024): MDD(5 seeds) = t_{0.975,8} * s_pooled * sqrt(2/5); 0.1 dB of Q2 = 6 p.p. at 10 %, 4.1 at 48 %, 3.0 at 66 %, 2.7 at 70 % (anchors); every table prints both
- `otherwise`: не доказано
