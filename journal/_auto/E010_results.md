<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вердикт** (по правилу до запуска): подтверждено: 9, не доказано: 35.

- a · cosine_to_zero_lr0.003_wu0_cm30 · 5000: **не доказано**
- a · const_lr0.001_wu0_cm1 · 5000: **не доказано**
- a · cosine_to_zero_lr0.003_wu0_cm30 · 10000: **не доказано**
- a · const_lr0.001_wu0_cm1 · 10000: **не доказано**
- a · cosine_to_zero_lr0.003_wu0_cm30 · 20000: **не доказано**
- a · cosine_to_zero_lr0.003_wu1000_cm30 · 20000: **не доказано**
- a · cosine_to_zero_lr0.003_wu0_cm10 · 20000: **подтверждено**
- a · cosine_to_zero_lr0.003_wu1000_cm10 · 20000: **не доказано**
- a · cosine_to_zero_lr0.003_wu1000_cm1 · 20000: **не доказано**
- a · cosine_to_floor_lr0.003_wu0_cm30 · 20000: **подтверждено**
- a · cosine_to_floor_lr0.003_wu0_cm10 · 20000: **подтверждено**
- a · cosine_to_floor_lr0.003_wu1000_cm30 · 20000: **не доказано**
- a · cosine_to_floor_lr0.003_wu1000_cm10 · 20000: **не доказано**
- a · cosine_to_zero_lr0.003_wu0_cm1 · 20000: **не доказано**
- a · cosine_to_floor_lr0.003_wu1000_cm1 · 20000: **не доказано**
- … ещё 29 (полностью — в таблице ниже)

Показано 15 из 46 конфигураций: лучшие по валидации и контроли. Тест — в точке лучшей валидации, %; Q² — прирост в той же точке, дБ.

| конфигурация |  | запусков (seeds) | валидация | тест, среднее | тест, min … max | Q², дБ | умножений |
|---|---|---|---|---|---|---|---|
| both · cosine_to_zero_lr0.003_wu0_cm30 · 20000 |  | 3 | 64.52 | 65.68 | 65.64 … 65.75 | 1.39 | 16032 |
| both · cosine_to_zero_lr0.003_wu1000_cm10 · 20000 |  | 3 | 64.24 | 65.41 | 64.96 … 66.07 | 1.38 | 16032 |
| both · cosine_to_zero_lr0.003_wu1000_cm30 · 20000 |  | 3 | 64.23 | 65.27 | 64.93 … 65.59 | 1.38 | 16032 |
| both · cosine_to_zero_lr0.003_wu0_cm10 · 20000 |  | 3 | 64.05 | 65.38 | 65.11 … 65.63 | 1.38 | 16032 |
| both · cosine_to_floor_lr0.003_wu0_cm30 · 20000 |  | 3 | 63.90 | 65.15 | 64.99 … 65.26 | 1.37 | 16032 |
| a · cosine_to_zero_lr0.003_wu0_cm30 · 20000 |  | 3 | 63.90 | 65.11 | 64.37 … 65.51 | 1.37 | 6480 |
| both · cosine_to_floor_lr0.003_wu1000_cm10 · 20000 |  | 3 | 63.87 | 65.22 | 64.69 … 65.88 | 1.38 | 16032 |
| both · cosine_to_floor_lr0.003_wu1000_cm30 · 20000 |  | 3 | 63.86 | 64.90 | 64.56 … 65.30 | 1.36 | 16032 |
| both · cosine_to_zero_lr0.003_wu1000_cm1 · 20000 |  | 3 | 63.82 | 64.68 | 64.17 … 64.99 | 1.36 | 16032 |
| a · cosine_to_zero_lr0.003_wu1000_cm30 · 20000 |  | 3 | 63.79 | 64.81 | 64.50 … 65.07 | 1.36 | 6480 |
| a · cosine_to_zero_lr0.003_wu0_cm10 · 20000 |  | 3 | 63.69 | 64.88 | 64.36 … 65.53 | 1.36 | 6480 |
| both · cosine_to_floor_lr0.003_wu0_cm10 · 20000 |  | 3 | 63.57 | 64.95 | 64.62 … 65.33 | 1.37 | 16032 |
| a · const_lr0.001_wu0_cm1 · 20000 | контроль | 3 | 61.08 | 62.26 | 62.02 … 62.38 | 1.28 | 6480 |
| both · const_lr0.001_wu0_cm1 · 20000 | контроль | 3 | 61.27 | 61.96 | 61.49 … 62.64 | 1.27 | 16032 |
| both · cosine_to_zero_lr0.001_wu0_cm1 · 20000 | контроль | 3 | 58.07 | 59.00 | 58.25 … 59.48 | 1.19 | 16032 |

Вердикты (`results/lr_sweep_summary.csv`, `verdict_draft_rule`): not proven: 35, better than control: 9, control: 2.

| архитектура | ветка lr | шагов | T95, шагов | тест, % |
|---|---|---|---|---|
| a | cosine_to_zero_lr0.003_wu0_cm10 | 20000 | 9000 | 64.91 |
| a | cosine_to_floor_lr0.003_wu0_cm30 | 20000 | 9333.33 | 64.34 |
| a | cosine_to_floor_lr0.003_wu0_cm10 | 20000 | 9333.33 | 64.13 |
| both | cosine_to_zero_lr0.003_wu0_cm30 | 10000 | 7250 | 61.94 |
| both | cosine_to_zero_lr0.003_wu0_cm30 | 20000 | 9833.33 | 65.67 |
| both | cosine_to_zero_lr0.003_wu0_cm10 | 20000 | 10166.67 | 65.38 |
| both | cosine_to_zero_lr0.003_wu0_cm1 | 20000 | 10416.67 | 64.45 |
| both | cosine_to_floor_lr0.003_wu0_cm10 | 20000 | 10583.33 | 64.56 |
| both | cosine_to_floor_lr0.003_wu0_cm30 | 20000 | 10000 | 64.35 |

Рисунки (клик — крупно, листание стрелками):

![E010_t95_vs_peak_lr.png](../figs/experiments/E010_lr_schedule_study/figures/E010_t95_vs_peak_lr.png){group="E010"}

![E010_val_curves.png](../figs/experiments/E010_lr_schedule_study/figures/E010_val_curves.png){group="E010"}
