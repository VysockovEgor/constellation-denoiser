<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Лучше ли при равных RM, чем одни триплеты и один Керр, связка «Керр с памятью (в первом порядке это осевые триплеты) + внеосевые триплеты (остаток IFWM)», и сколько внеосевых членов стоит оставлять?

**Правило до запуска** (`experiments/E046_kerr_offaxis/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `a_wins`: architecture A wins if, at equal RM (rival RM <= 1.05 x RM of A), its best configuration by validation is better in test than both the best triplets-only (T) and the best Kerr-only (K) by more than 2 std over replicas (std = the largest of the compared); otherwise negative
- `double_count`: reported: (Kerr + T3 with axes) - (Kerr + off-axis), same M and N_c
- `otherwise`: не доказано
