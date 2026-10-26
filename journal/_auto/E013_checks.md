<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

Итог: **пройден** (`results/control_check.json`).

| контроль | цель, % | допуск, п.п. | среднее, % | по seeds | итог |
|---|---|---|---|---|---|
| both_const_1e-3_20k | 61.73 | 0.70 | 61.63 | 61.76, 61.49 | в допуске |

Последняя проверка (`logs/check-control.log`):

    2026-10-19T10:30:49+07:00 E013 check-control: /home/redyuk/DDC/OS/NY/Egor/constellation-denoiser/.venv/bin/python -m constellation_denoiser.runner --exp E013 --stage control --check-control ===
    [2026-10-19 10:30:51] control both_const_1e-3_20k: mean test_reduction_valbest_select = 61.63% по 2 seed; цель 61.727±0.7 п.п. -> OK

**Аномалии** (пороги: хуже контроля > 5 п.п., разброс seeds > 3 п.п., |валидация − тест| > 5 п.п.):

- **diverged = true**: 0
- **тест хуже контроля (61.63 %) более чем на 5 п.п.**: 12:
  - transformer_d32_L1 · 7 · 30 · 20000: 6.00 %
  - transformer_d32_L1 · 7 · 210 · 20000: 6.28 %
  - transformer_d32_L1 · 7 · 210 · 100000: 11.86 %
  - transformer_d32_L1 · 31 · 30 · 20000: 14.49 %
  - transformer_d32_L1 · 31 · 210 · 20000: 16.28 %
  - transformer_d32_L1 · 31 · 210 · 100000: 24.14 %
  - transformer_d64_L2 · 7 · 30 · 20000: 29.68 %
  - transformer_d64_L2 · 7 · 210 · 20000: 32.15 %
  - transformer_d64_L2 · 7 · 210 · 100000: 48.21 %
  - transformer_d64_L2 · 31 · 30 · 20000: 33.91 %
  - … ещё 2
- **разброс seeds больше 3 п.п.**: 4:
  - transformer_d32_L1 · 7 · 210 · 100000: 4.51 п.п.
  - transformer_d32_L1 · 31 · 210 · 20000: 4.24 п.п.
  - transformer_d32_L1 · 31 · 210 · 100000: 3.67 п.п.
  - transformer_d64_L2 · 31 · 210 · 100000: 8.43 п.п.
- **валидация и тест расходятся больше чем на 5 п.п.**: 0
