<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

Итог: **пройден** (`results/control_check.json`).

| контроль | цель, % | допуск, п.п. | среднее, % | по seeds | итог |
|---|---|---|---|---|---|
| both_const_1e-3_20k | 61.73 | 0.70 | 61.63 | 61.76, 61.49 | в допуске |
| a_const_1e-3_20k | 62.19 | 0.70 | 62.19 | 62.02, 62.37 | в допуске |
| both_cosine_to_zero_1e-3_20k | 58.76 | 0.70 | 58.76 | 59.27, 58.25 | в допуске |

Последняя проверка (`logs/check-control.log`):

    2026-10-18T22:24:49+07:00 E010 check-control: /home/redyuk/DDC/OS/NY/Egor/constellation-denoiser/.venv/bin/python -m constellation_denoiser.runner --exp E010 --stage control --check-control ===
    [2026-10-18 22:24:50] control both_const_1e-3_20k: mean test_reduction_valbest_select = 61.63% по 2 seed; цель 61.727±0.7 п.п. -> OK
    [2026-10-18 22:24:50] control a_const_1e-3_20k: mean test_reduction_valbest_select = 62.19% по 2 seed; цель 62.191±0.7 п.п. -> OK
    [2026-10-18 22:24:50] control both_cosine_to_zero_1e-3_20k: mean test_reduction_valbest_select = 58.76% по 2 seed; цель 58.763±0.7 п.п. -> OK

**Аномалии** (пороги: хуже контроля > 5 п.п., разброс seeds > 3 п.п., |валидация − тест| > 5 п.п.):

- **diverged = true**: 0
- **тест хуже контроля (61.63 %) более чем на 5 п.п.**: 9:
  - a · const_lr0.001_wu0_cm1 · 5000: 8.84 %
  - a · const_lr0.001_wu0_cm1 · 10000: 54.12 %
  - a · const_lr0.01_wu0_cm1 · 20000: 40.01 %
  - a · cosine_to_zero_lr0.003_wu0_cm30 · 5000: 43.91 %
  - both · const_lr0.001_wu0_cm1 · 5000: 7.55 %
  - both · const_lr0.001_wu0_cm1 · 10000: 53.33 %
  - both · const_lr0.01_wu0_cm1 · 20000: 38.90 %
  - both · cosine_to_floor_lr0.01_wu0_cm1 · 20000: 55.51 %
  - both · cosine_to_zero_lr0.003_wu0_cm30 · 5000: 43.21 %
- **разброс seeds больше 3 п.п.**: 8:
  - a · const_lr0.001_wu0_cm1 · 5000: 14.96 п.п.
  - a · const_lr0.01_wu0_cm1 · 20000: 3.91 п.п.
  - a · cosine_to_floor_lr0.01_wu0_cm1 · 20000: 4.69 п.п.
  - a · cosine_to_zero_lr0.003_wu0_cm30 · 5000: 7.80 п.п.
  - a · cosine_to_zero_lr0.003_wu0_cm30 · 10000: 4.10 п.п.
  - a · cosine_to_zero_lr0.01_wu0_cm1 · 20000: 4.34 п.п.
  - both · const_lr0.001_wu0_cm1 · 5000: 14.69 п.п.
  - both · cosine_to_zero_lr0.003_wu0_cm30 · 5000: 8.77 п.п.
- **валидация и тест расходятся больше чем на 5 п.п.**: 0
