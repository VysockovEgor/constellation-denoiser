<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

Итог: **пройден** (`results/control_check.json`).

| контроль | цель, % | допуск, п.п. | среднее, % | по seeds | итог |
|---|---|---|---|---|---|
| both_const_1e-3_20k | 61.73 | 0.70 | 61.63 | 61.76, 61.49 | в допуске |
| a_const_1e-3_20k | 62.19 | 0.70 | 62.19 | 62.02, 62.37 | в допуске |

Последняя проверка (`logs/check-control.log`):

    2026-10-19T02:27:48+07:00 E011 check-control: /home/redyuk/DDC/OS/NY/Egor/constellation-denoiser/.venv/bin/python -m constellation_denoiser.runner --exp E011 --stage control --check-control ===
    [2026-10-19 02:27:50] control both_const_1e-3_20k: mean test_reduction_valbest_select = 61.63% по 2 seed; цель 61.727±0.7 п.п. -> OK
    [2026-10-19 02:27:50] control a_const_1e-3_20k: mean test_reduction_valbest_select = 62.19% по 2 seed; цель 62.191±0.7 п.п. -> OK

**Аномалии** (пороги: хуже контроля > 5 п.п., разброс seeds > 3 п.п., |валидация − тест| > 5 п.п.):

- **diverged = true**: 0
- **тест хуже контроля (61.63 %) более чем на 5 п.п.**: 2:
  - real_tanh · const_1e-3 · 30: 32.19 %
  - real_tanh · const_1e-3 · 210: 38.86 %
- **разброс seeds больше 3 п.п.**: 0
- **валидация и тест расходятся больше чем на 5 п.п.**: 0
