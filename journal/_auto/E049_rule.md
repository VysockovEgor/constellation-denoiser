<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Следуют ли обученные шаги DBP (E018, E025, E028) физическому фильтру дисперсии H(ν) = exp(j c ν²), c = β₂·dz·Rs²/2, dz = 2000 км / Ns, и дают ли физические начальные значения (КИХ, α) выигрыш до обучения?

**Правило до запуска** (`experiments/E049_dbp_physical_init/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `calibration`: reported: median c_fit * Ns per (Ns, L) against c_total = -21.5 rad, fit quality r2; agreement if |c_fit * Ns| is within a factor 1.5 of 21.5 with r2 >= 0.5
- `signal`: a physical arm shows a signal if, at the same RM, it beats delta in test by > 2 SE (SE = s / sqrt(10)), or reaches 90 % of the final validation of delta in at most half the steps (mean over seeds), or halves the seed std; 2 std is reported as well
- `otherwise`: не доказано
