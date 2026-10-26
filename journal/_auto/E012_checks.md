<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

Итог: **пройден** (`results/control_check.json`).

| контроль | цель, % | допуск, п.п. | среднее, % | по seeds | итог |
|---|---|---|---|---|---|
| v1_w31_20k | 61.73 | 0.70 | 61.63 | 61.76, 61.49 | в допуске |
| v2_w31_20k | 62.19 | 0.70 | 62.19 | 62.02, 62.37 | в допуске |
| v3_w31_20k | 59.44 | 0.70 | 59.44 | 58.27, 60.61 | в допуске |

Последняя проверка (`logs/check-control.log`):

    2026-10-18T23:46:08+07:00 E012 check-control: /home/redyuk/DDC/OS/NY/Egor/constellation-denoiser/.venv/bin/python -m constellation_denoiser.runner --exp E012 --stage control --check-control ===
    [2026-10-18 23:46:09] control v1_w31_20k: mean test_reduction_valbest_select = 61.63% по 2 seed; цель 61.727±0.7 п.п. -> OK
    [2026-10-18 23:46:09] control v2_w31_20k: mean test_reduction_valbest_select = 62.19% по 2 seed; цель 62.191±0.7 п.п. -> OK
    [2026-10-18 23:46:09] control v3_w31_20k: mean test_reduction_valbest_select = 59.44% по 2 seed; цель 59.437±0.7 п.п. -> OK

**Аномалии** (пороги: хуже контроля > 5 п.п., разброс seeds > 3 п.п., |валидация − тест| > 5 п.п.):

- **diverged = true**: 0
- **тест хуже контроля (61.63 %) более чем на 5 п.п.**: 13:
  - v1_two_kerr_both · 127 · 20000 · best_e010: 53.97 %
  - v1_two_kerr_both · 127 · 20000 · const_1e-3: 52.33 %
  - v2_two_kerr_last · 127 · 20000 · best_e010: 54.58 %
  - v2_two_kerr_last · 127 · 20000 · const_1e-3: 52.22 %
  - v2_two_kerr_last · 127 · 100000 · best_e010: 55.12 %
  - v3_one_hidden · 65 · 20000 · best_e010: 54.43 %
  - v3_one_hidden · 65 · 20000 · const_1e-3: 48.50 %
  - v3_one_hidden · 127 · 20000 · best_e010: 40.20 %
  - v3_one_hidden · 127 · 20000 · const_1e-3: 35.40 %
  - v3_one_hidden · 127 · 100000 · best_e010: 46.18 %
  - … ещё 3
- **разброс seeds больше 3 п.п.**: 21:
  - v1_two_kerr_both · 127 · 20000 · best_e010: 3.91 п.п.
  - v1_two_kerr_both · 127 · 20000 · const_1e-3: 3.36 п.п.
  - v1_two_kerr_both · 127 · 100000 · best_e010: 7.31 п.п.
  - v2_two_kerr_last · 65 · 20000 · const_1e-3: 3.04 п.п.
  - v2_two_kerr_last · 127 · 20000 · best_e010: 5.73 п.п.
  - v2_two_kerr_last · 127 · 20000 · const_1e-3: 3.20 п.п.
  - v3_one_hidden · 31 · 20000 · const_1e-3: 3.30 п.п.
  - v3_one_hidden · 65 · 20000 · const_1e-3: 6.04 п.п.
  - v3_one_hidden · 65 · 100000 · const_1e-3: 5.36 п.п.
  - v3_one_hidden · 127 · 20000 · best_e010: 5.67 п.п.
  - … ещё 11
- **валидация и тест расходятся больше чем на 5 п.п.**: 0
