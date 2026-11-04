<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Как выглядят одношаговые триплеты, каскад E040, смешанные ступени E041, Kerr-сети и DBP на одной картине «ΔQ² и снижение BER против RM», и меняется ли картина при другой конвенции подсчёта RM?

**Правило до запуска** (`experiments/E043_triplet_summary/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `robust_leader`: a family is the robust leader at a level if its best-by-validation point has the highest test mean under all three RM conventions and its interval does not overlap the next family's
- `otherwise`: не доказано
