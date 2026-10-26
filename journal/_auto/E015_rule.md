<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Зависит ли фазовая ошибка символа от мощности соседей, а не только от собственной мощности?

**Правило до запуска** (`experiments/E015_physics_diagnostics/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `memory_present_if_all`:
  - (a) on validation |rho(dphi, P_nb(R))| >= |rho(dphi, |z|^2)| for at least one R in {1, 3, 7, 15} (Spearman, all rings)
  - (b) on validation the fit with R = 7 reduces BER by at least fit_gap_pp more than the fit with R = 1
  - (c) on test the sign is the same and |test / validation| is within test_ratio_range, for rho of the R with the largest validation |rho| and for the R7 - R1 gap
- `fit_gap_pp`: 2.0
- `test_ratio_range`:
  - 0.1
  - 10.0
- `otherwise`: не доказано
- `uncertainty`: 95% file-bootstrap intervals are reported for every number
