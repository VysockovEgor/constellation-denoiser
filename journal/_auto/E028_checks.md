<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

Итог: **пройден** (`results/control_check.json`).

| контроль | цель, % | допуск, п.п. | среднее, % | по seeds | итог |
|---|---|---|---|---|---|
| a_const_1e-3_20k | 62.19 | 0.70 | 62.19 | 62.02, 62.37 | в допуске |

Последняя проверка (`logs/check-control.log`):

    2026-10-30T11:50:31+07:00 E028 check-control: .venv/bin/python -m constellation_denoiser.runner --exp E028 --stage control --check-control ===
    [2026-10-30 11:50:33] control a_const_1e-3_20k: mean test_reduction_valbest_select = 62.19% по 2 seed; цель 62.191±0.7 п.п. -> OK
