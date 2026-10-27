<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Превосходит ли наш подход сильные классические базовые методы при равном числе умножений?

**Правило до запуска** (`experiments/E022_baselines/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `better_if`: our approach (best complex Kerr MLP or Phys + MLP point of E021 at the same level) beats a baseline if its mean Q2 gain is higher by > 0.1 dB and the min-max intervals over seeds do not overlap
- `dbp`: training the nonlinearity matters if dbp_train_NsX beats dbp_fixed_NsX by > 0.1 dB Q2 with non-overlapping seed intervals
- `otherwise`: не доказано
