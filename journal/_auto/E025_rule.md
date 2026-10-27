<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Меняется ли рейтинг семейств и их абсолютные числа, если у каждого семейства одинаковый бюджет настройки?

**Правило до запуска** (`experiments/E025_family_tuning/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `recipe`: a new recipe replaces the previous one (ref_current) if on S3 its mean validation is higher by >= 0.5 p.p. and the min-max intervals over seeds do not overlap; otherwise ref_current stays
- `ranking`: the family ranking changed if at one RM level two families swapped places with a difference >= 1 p.p. compared with E021 (results/level_best.csv)
- `physics`: pure physics (F5) was under-trained if its best recipe gives >= ref_current + 3 p.p. (validation)
- `control`: complex_a 20k const 62.191 +- 0.7 (gate before S1); code check complex_a cm30 200k 66.253 +- 0.7 (seeds 17, 29; after S1) - outside -> halt and check the code
- `noise`: thresholds in p.p. are compared with results/noise_floor.json (E024): MDD(5 seeds) = t_{0.975,8} * s_pooled * sqrt(2/5); 0.1 dB of Q2 = 6 p.p. at 10 %, 4.1 at 48 %, 3.0 at 66 %, 2.7 at 70 % (anchors); every table prints both
- `otherwise`: не доказано
