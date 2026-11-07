<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** —

**Правило до запуска** (`experiments/E054_dbpx_pruning/config.json` → `rule_draft`, дословно):

- `report`: per (Ns, length): mean, std, SE, range over seeds; share of the L = 255 level kept; RM in both conventions
- `limits`: one schedule (no tuning of the schedule); circular files; one channel, one polarization, 1 sps
- `progressive_vs_others`: progressive better than direct / direct_ld / abrupt at a length if the difference > 2 SE and > 2 std (reverse also reported)
- `phase_gate`: signal: progressive or direct_ld keeps, at some L <= 31, a mean test reduction (5 seeds) > 0 and >= 0.5 of the progressive L = 255 level; otherwise phase 2 is not run
