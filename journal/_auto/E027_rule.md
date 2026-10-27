<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Даёт ли именно Kerr-активация exp(i·c·|z|²)·z преимущество над другими комплексными активациями при равной ширине?

**Правило до запуска** (`experiments/E027_activation_ablation/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `kerr_vs_a`: Kerr (kerr_c_train) is better than activation A at equal h if the difference of mean test >= 1 p.p. (~0.035 dB) and the min-max intervals do not overlap; otherwise not distinguishable (S2 per width, S3 at h48)
- `trainable_c`: trainability of c is needed if kerr_c_train - kerr_c_fixed >= 0.5 p.p.
- `nonlinearity`: nonlinearity is needed if the best non-linear activation - identity >= 20 p.p.
- `rm`: RM_total printed: Kerr adds 7h RM (h = 48: 336 of 6 480, ~5 %)
- `noise`: thresholds in p.p. are compared with results/noise_floor.json (E024): MDD(5 seeds) = t_{0.975,8} * s_pooled * sqrt(2/5); 0.1 dB of Q2 = 6 p.p. at 10 %, 4.1 at 48 %, 3.0 at 66 %, 2.7 at 70 % (anchors); every table prints both
- `otherwise`: не доказано
