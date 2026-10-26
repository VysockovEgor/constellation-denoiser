<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Выходят ли complex_both/a/one_hidden и real_tanh на плато за 200k шагов при N = 30 и 210?

**Правило до запуска** (`experiments/E011_long_runs_convergence/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `plateau`: improvement of the smoothed validation reduction over the last 50k steps < 0.2 p.p.
- `no_plateau`: report 'plateau not reached within N steps'; y_inf only as a rough estimate
- `overfitting`: a gap between N=30 and N=210 is interpreted as overfitting only if the N=30 validation curve falls (smoothed maximum - smoothed final > 0.2 p.p.) while the N=210 curve does not
