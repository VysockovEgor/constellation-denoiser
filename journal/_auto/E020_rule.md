<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Почему гибрид Phys + MLP (h48, joint) с веткой `best_e010` (cosine_to_zero, пик 3e-3, `c_mult` 1) даёт в E019 51.5 % вместо 62.8 % у const 1e-3?

**Правило до запуска** (`experiments/E020_e017_anomaly/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `restored_threshold`: 61.039 % = min over seeds of test_reduction_valbest of E019 'E017_hyb_R31_L7_sym_h48_joint' const_1e-3 at 200k (experiments/E019_confirm_long_runs/results/confirm.csv)
- `restored_if`: mean test_reduction_valbest over the 3 seeds >= 61.039 % at the same number of steps
- `interpretation`:
  - A1 (c_mult 30) or A2 (const schedule) restored -> the cause is the combination c_mult 1 + schedule, not the hybrid idea
  - A3 (cosine, peak 1e-3) restored while A2 (const, peak 3e-3) is not -> the cause is the peak lr 3e-3
  - none restored -> check code and checkpoint selection (hypothesis d)
  - hypothesis c (overfitting) is judged on existing data: rejected if the 100k and 200k anomalous results differ by less than 1 p.p. and the training loss of the anomalous runs is not lower than that of the normal runs
- `otherwise`: не доказано
