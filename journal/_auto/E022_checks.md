<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

Итог: **пройден** (`results/control_check.json`).

| контроль | цель, % | допуск, п.п. | среднее, % | по seeds | итог |
|---|---|---|---|---|---|
| a_const_1e-3_20k | 62.19 | 0.70 | 62.19 | 62.02, 62.37 | в допуске |

Последняя проверка (`logs/check-control.log`):

    2026-10-27T21:57:54+07:00 E022 check-control: .venv/bin/python -m constellation_denoiser.runner --exp E022 --stage control --check-control ===
    [2026-10-27 21:57:55] control a_const_1e-3_20k: mean test_reduction_valbest_select = 62.19% по 2 seed; цель 62.191±0.7 п.п. -> OK

**Аномалии** (пороги: хуже контроля > 5 п.п., разброс seeds > 3 п.п., |валидация − тест| > 5 п.п.):

- **diverged = true**: 0
- **«тест хуже контроля более чем на 5 п.п.» не применяется: контроль — другая модель (complex_a)**
- **разброс seeds больше 3 п.п.**: 5:
  - cnn_C13_k9: 4.37 п.п.
  - cnn_C22_k11: 11.55 п.п.
  - cnn_C36_k11: 5.07 п.п.
  - dbp_fixed_Ns8: 5.32 п.п.
  - dbp_train_Ns8: 5.49 п.п.
- **валидация и тест расходятся больше чем на 5 п.п.**: 0
