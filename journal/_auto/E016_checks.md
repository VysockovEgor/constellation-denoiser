<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

Итог: **пройден** (`results/control_check.json`).

| контроль | цель, % | допуск, п.п. | среднее, % | по seeds | итог |
|---|---|---|---|---|---|
| a_const_1e-3_20k | 62.19 | 0.70 | 62.19 | 62.02, 62.37 | в допуске |

Последняя проверка (`logs/check-control.log`):

    2026-10-26T13:50:20+07:00 E016 check-control: .venv/bin/python -m constellation_denoiser.runner --exp E016 --stage control --check-control ===
    [2026-10-26 13:50:22] control a_const_1e-3_20k: mean test_reduction_valbest_select = 62.19% по 2 seed; цель 62.191±0.7 п.п. -> OK

**Аномалии** (пороги: хуже контроля > 5 п.п., разброс seeds > 3 п.п., |валидация − тест| > 5 п.п.):

- **diverged = true**: 0
- **тест хуже контроля (62.19 %) более чем на 5 п.п.**: 111:
  - fir_L31_nosym · best_e010_cm1: -5.22 %
  - fir_L31_nosym · best_e010_cm30: -5.22 %
  - fir_L31_nosym · const_1e-3: -5.17 %
  - fir_L31_sym · best_e010_cm1: -5.12 %
  - fir_L31_sym · best_e010_cm30: -5.12 %
  - fir_L31_sym · const_1e-3: -4.97 %
  - phys_R11_L1_nosym · best_e010_cm1: -0.30 %
  - phys_R11_L1_nosym · best_e010_cm30: -0.35 %
  - phys_R11_L1_nosym · const_1e-3: -0.32 %
  - phys_R11_L1_sym · best_e010_cm1: -0.28 %
  - … ещё 101
- **разброс seeds больше 3 п.п.**: 0
- **валидация и тест расходятся больше чем на 5 п.п.**: 0
