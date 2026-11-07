<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** —

**Правило до запуска** (`experiments/E053_chirp_triplets/config.json` → `rule_draft`, дословно):

- `h1_confirmed`: for one of chirp_complex / chirp_imag / chirp_joint_imag with S <= 5 and <= 20 real parameters: mean test (5 replicas) lower than (i) by less than std(i); both significance rules (2 SE, 2 std) reported
- `h1_rejected`: at S = 8 every chirp variant is worse than (i) by more than 2 SE AND more than 2 std
- `h1_not_proven`: otherwise
- `physical_uniqueness`: chirp_imag beats radial (iii) at the nearest number of parameters by > 2 SE and > 2 std; otherwise the chirp form is only a small-parameter form, not unique
- `splits`: the best chirp configuration (by validation) on the 3 E023 splits: the same sign of (chirp - full) on all 3
- `stop`: H1 rejected -> task 3 is not run; task 2A is done anyway; 2B only if H1 confirmed or within 2 std of (i)
- `limits`: one channel, one polarization, 1 sps, one stage (A1), spread over bootstrap replicas of the train files
