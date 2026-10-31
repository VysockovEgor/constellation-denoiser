<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

Итог: **пройден** (`results/control_check.json`).

| контроль | цель, % | допуск, п.п. | среднее, % | по seeds | итог |
|---|---|---|---|---|---|
| a_const_1e-3_20k | 62.19 | 0.70 | 62.19 | 62.02, 62.37 | в допуске |

Последняя проверка (`logs/check-control.log`):

    2026-10-28T04:51:59+07:00 E023 check-control: .venv/bin/python -m constellation_denoiser.runner --exp E023 --stage control --check-control ===
    [2026-10-28 04:52:00] control a_const_1e-3_20k: mean test_reduction_valbest_select = 62.19% по 2 seed; цель 62.191±0.7 п.п. -> OK

**Аномалии** (пороги: хуже контроля > 5 п.п., разброс seeds > 3 п.п., |валидация − тест| > 5 п.п.):

- **diverged = true**: 0
- **«тест хуже контроля более чем на 5 п.п.» не применяется: контроль — другая модель (complex_a)**
- **разброс seeds больше 3 п.п.**: 8:
  - dbp_E018_dbp_Ns8_L31_R7_joint_from_scratch · 101.0: 7.27 п.п.
  - dbp_E018_dbp_Ns8_L31_R7_joint_from_scratch · 202.0: 5.48 п.п.
  - dbp_E018_dbp_Ns8_L31_R7_joint_from_scratch · 303.0: 4.53 п.п.
  - real_E011_real_tanh · 101.0: 5.11 п.п.
  - real_E011_real_tanh · 303.0: 3.44 п.п.
  - transformer_E021_tr_w7_d8_L2 · 101.0: 4.59 п.п.
  - transformer_E021_tr_w7_d8_L2 · 202.0: 4.95 п.п.
  - transformer_E021_tr_w7_d8_L2 · 303.0: 4.66 п.п.
- **валидация и тест расходятся больше чем на 5 п.п.**: 0
