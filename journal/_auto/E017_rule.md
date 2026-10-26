<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Сколько нейросети (h = 4–48) нужно на остаток после PhysKerr, frozen и joint?

**Правило до запуска** (`experiments/E017_phys_plus_mlp/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `better_at_equal_cost`: the hybrid (any h, any mode) with multiplies <= 6480 and the best mean validation reduction beats on test both complex_a (this experiment, 5 seeds) and the E016 configuration chosen by validation (5 seeds) by >= 1 p.p. and by more than the seed spread
- `residual_net_not_needed`: for h = 4, both frozen and joint differ on test from the chosen E016 configuration by less than the seed spread
- `frozen_vs_joint`: reported as an observation only, no verdict
- `otherwise`: не доказано
