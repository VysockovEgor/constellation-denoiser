<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Сокращают ли больший пиковый lr, warmup, отдельный lr для c и затухание число шагов до 95 % качества контроля (T95)?

**Правило до запуска** (`experiments/E010_lr_schedule_study/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `better_than_control_if_all`:
  - mean T95 <= 0.7 * control mean T95
  - final test reduction >= control final test reduction - 0.5 p.p. (means over seeds)
  - no seed diverged: no NaN and final validation reduction >= control mean - 2 p.p.
- `otherwise`: not proven
- `always_report`: spread over seeds (min, max, std)
