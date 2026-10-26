<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

Итог: **пройден** (`results/control_check.json`).

| контроль | цель, % | допуск, п.п. | среднее, % | по seeds | итог |
|---|---|---|---|---|---|
| a_const_1e-3_20k | 62.19 | 0.70 | 62.19 | 62.02, 62.37 | в допуске |

Последняя проверка (`logs/check-control.log`):

    2026-10-26T15:10:41+07:00 E019 check-control: .venv/bin/python -m constellation_denoiser.runner --exp E019 --stage control --check-control ===
    [2026-10-26 15:10:43] control a_const_1e-3_20k: mean test_reduction_valbest_select = 62.19% по 2 seed; цель 62.191±0.7 п.п. -> OK

**Аномалии** (в `confirm.csv` нет поля `diverged`):

- **diverged = true**: 0
- **разброс seeds больше 3 п.п.**: 7:
  - E017_hyb_R31_L7_sym_h48_joint · best_e010 · 100000: 4.18 п.п.
  - E017_hyb_R31_L7_sym_h48_joint · best_e010 · 200000: 4.15 п.п.
  - E017_hyb_R31_L7_sym_h48_joint · const_1e-3 · 200000: 3.23 п.п.
  - E018_dbp_Ns8_L31_R7_joint_from_scratch · best_e010 · 100000: 3.69 п.п.
  - E018_dbp_Ns8_L31_R7_joint_from_scratch · best_e010 · 200000: 4.09 п.п.
  - E018_dbp_Ns8_L31_R7_joint_from_scratch · const_1e-3 · 100000: 6.43 п.п.
  - E018_dbp_Ns8_L31_R7_joint_from_scratch · const_1e-3 · 200000: 3.98 п.п.
- **валидация и тест расходятся больше чем на 5 п.п.**: 0
