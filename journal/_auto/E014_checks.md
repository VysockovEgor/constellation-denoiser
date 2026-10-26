<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

Итог: **пройден** (`results/control_check.json`).

| контроль | цель, % | допуск, п.п. | среднее, % | по seeds | итог |
|---|---|---|---|---|---|
| one_hidden_const_1e-3_20k | 59.44 | 0.70 | 59.44 | 58.27, 60.61 | в допуске |

Последняя проверка (`logs/check-control.log`):

    2026-10-19T01:34:37+07:00 E014 check-control: /home/redyuk/DDC/OS/NY/Egor/constellation-denoiser/.venv/bin/python -m constellation_denoiser.runner --exp E014 --stage control --check-control ===
    [2026-10-19 01:34:39] control one_hidden_const_1e-3_20k: mean test_reduction_valbest_select = 59.44% по 2 seed; цель 59.437±0.7 п.п. -> OK

**Аномалии** (пороги: хуже контроля > 5 п.п., разброс seeds > 3 п.п., |валидация − тест| > 5 п.п.):

- **diverged = true**: 0
- **тест хуже контроля (59.44 %) более чем на 5 п.п.**: 4:
  - k1_kan_only · 5.0: 9.57 %
  - k1_kan_only · 8.0: 12.06 %
  - k2_parallel_joint · 5.0: 33.24 %
  - k2_parallel_joint · 8.0: 15.10 %
- **разброс seeds больше 3 п.п.**: 4:
  - k2_parallel_joint · 5.0: 14.29 п.п.
  - k2_parallel_joint · 8.0: 12.43 п.п.
  - k3_parallel_staged · 5.0: 3.30 п.п.
  - k3_parallel_staged · 8.0: 3.30 п.п.
- **валидация и тест расходятся больше чем на 5 п.п.**: 0
