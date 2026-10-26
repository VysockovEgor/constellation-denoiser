<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

Итог: **пройден** (`results/control_check.json`).

| контроль | цель, % | допуск, п.п. | среднее, % | по seeds | итог |
|---|---|---|---|---|---|
| a_const_1e-3_20k | 62.19 | 0.70 | 62.19 | 62.02, 62.37 | в допуске |

Последняя проверка (`logs/check-control.log`):

    2026-10-26T15:10:32+07:00 E018 check-control: .venv/bin/python -m constellation_denoiser.runner --exp E018 --stage control --check-control ===
    [2026-10-26 15:10:34] control a_const_1e-3_20k: mean test_reduction_valbest_select = 62.19% по 2 seed; цель 62.191±0.7 п.п. -> OK

**Аномалии** (пороги: хуже контроля > 5 п.п., разброс seeds > 3 п.п., |валидация − тест| > 5 п.п.):

- **diverged = true**: 0
- **«тест хуже контроля более чем на 5 п.п.» не применяется: контроль — другая модель (complex_a)**
- **разброс seeds больше 3 п.п.**: 2:
  - dbp_Ns4_L31_R7_joint_from_scratch: 4.47 п.п.
  - dbp_Ns8_L31_R7_joint_from_scratch: 4.40 п.п.
- **валидация и тест расходятся больше чем на 5 п.п.**: 0
