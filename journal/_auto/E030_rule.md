<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Даёт ли обучение под BER-подобную потерю выигрыш над MSE?

**Правило до запуска** (`experiments/E030_loss_functions/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `better`: loss L is better than MSE if the difference is >= 0.7 p.p. (5 seeds, intervals do not overlap) on at least two of the four models (S2)
- `note`: E016: linear FIR + own power gave -5 % (worse than raw), a hint that MSE and BER disagree
- `noise`: thresholds in p.p. are compared with results/noise_floor.json (E024): MDD(5 seeds) = t_{0.975,8} * s_pooled * sqrt(2/5); 0.1 dB of Q2 = 6 p.p. at 10 %, 4.1 at 48 %, 3.0 at 66 %, 2.7 at 70 % (anchors); every table prints both
- `otherwise`: не доказано
