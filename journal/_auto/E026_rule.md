<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Какая простая Kerr-сеть лучшая в области 100–40 000 RM, и где окупаются окно шире 31 или глубина?

**Правило до запуска** (`experiments/E026_kerr_complexity_map/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `width`: a window wider than 31 pays off at a given RM if best(w) for w > 31 is higher than best(w = 31) by >= 0.5 p.p. (3 seeds, intervals do not overlap); RM bins: +-25 % log windows of results/pareto levels
- `depth`: depth (both) pays off if both is better than one_hidden at equal RM (same bin) by >= 0.5 p.p.
- `saturation`: the smallest RM after which doubling RM gives < 0.5 p.p. (envelope of the validation means)
- `noise`: thresholds in p.p. are compared with results/noise_floor.json (E024): MDD(5 seeds) = t_{0.975,8} * s_pooled * sqrt(2/5); 0.1 dB of Q2 = 6 p.p. at 10 %, 4.1 at 48 %, 3.0 at 66 %, 2.7 at 70 % (anchors); every table prints both
- `otherwise`: не доказано
