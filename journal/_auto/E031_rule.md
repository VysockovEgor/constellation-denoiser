<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Чего не хватает фазовой модели PhysKerr (9 %), и как стоит классический метод возмущений при равных RM?

**Правило до запуска** (`experiments/E031_perturbation_triplets/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `xpm_insufficient`: diagonal (XPM-type) physics is insufficient if T3 - T2 >= 5 p.p. at equal M (test mean over 5 replicas, M chosen by validation)
- `t1_vs_physkerr`: T1 ~ PhysKerr if |T1 - 9.3| < 3 p.p.
- `ours_vs_pbnlc`: our approach is better than PB-NLC at equal RM if the Q2 gain difference >= 0.1 dB and the intervals do not overlap (E033 plot)
- `otherwise`: не доказано
