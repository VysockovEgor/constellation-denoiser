<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

Итог: **пройден** (`results/control_check.json`).

| контроль | цель, % | допуск, п.п. | среднее, % | по seeds | итог |
|---|---|---|---|---|---|
| a_const_1e-3_20k | 62.19 | 0.70 | 62.19 | 62.02, 62.37 | в допуске |
| one_hidden_const_1e-3_20k | 59.44 | 0.70 | 59.44 | 58.27, 60.61 | в допуске |

Последняя проверка (`logs/check-control.log`):

    2026-10-26T14:29:49+07:00 E017 check-control: .venv/bin/python -m constellation_denoiser.runner --exp E017 --stage control --check-control ===
    [2026-10-26 14:29:51] control a_const_1e-3_20k: mean test_reduction_valbest_select = 62.19% по 2 seed; цель 62.191±0.7 п.п. -> OK
    [2026-10-26 14:29:51] control one_hidden_const_1e-3_20k: mean test_reduction_valbest_select = 59.44% по 2 seed; цель 59.437±0.7 п.п. -> OK

**Аномалии** (пороги: хуже контроля > 5 п.п., разброс seeds > 3 п.п., |валидация − тест| > 5 п.п.):

- **diverged = true**: 0
- **«тест хуже контроля более чем на 5 п.п.» не применяется: контроль — другая модель (complex_a)**
- **разброс seeds больше 3 п.п.**: 4:
  - h48_frozen · const_1e-3: 11.78 п.п.
  - h4_frozen · const_1e-3: 5.27 п.п.
  - h8_frozen · const_1e-3: 5.01 п.п.
  - one_hidden · const_1e-3: 3.30 п.п.
- **валидация и тест расходятся больше чем на 5 п.п.**: 0
