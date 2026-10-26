<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

Показано 15 из 33 конфигураций: лучшие по валидации и контроли. Тест — в точке лучшей валидации, %; Q² — прирост в той же точке, дБ.

| конфигурация |  | seeds | валидация | тест, среднее | тест, min … max | Q², дБ | умножений |
|---|---|---|---|---|---|---|---|
| dbp_Ns8_L31_R7_joint_from_scratch |  | 5 | 47.40 | 48.28 | 46.12 … 50.52 | 0.90 | 592 |
| dbp_Ns8_L31_R3_joint_from_scratch |  | 5 | 35.16 | 35.39 | 34.28 … 36.51 | 0.62 | 576 |
| dbp_Ns4_L31_R7_joint_from_scratch |  | 5 | 33.14 | 33.94 | 32.22 … 36.69 | 0.59 | 296 |
| dbp_Ns8_L7_R7_joint_from_scratch |  | 5 | 8.41 | 8.97 | 8.18 … 9.82 | 0.14 | 208 |
| dbp_Ns2_L31_R7_joint_from_scratch |  | 5 | 6.35 | 6.73 | 6.53 … 7.03 | 0.10 | 148 |
| dbp_Ns4_L7_R7_joint_from_scratch |  | 5 | 4.32 | 4.68 | 4.43 … 4.88 | 0.07 | 104 |
| dbp_Ns8_L7_R3_joint_from_scratch |  | 5 | 3.32 | 3.98 | 3.61 … 4.59 | 0.06 | 192 |
| dbp_Ns8_L7_R7_layerwise |  | 5 | 2.23 | 2.80 | 2.20 … 3.26 | 0.04 | 208 |
| dbp_Ns2_L7_R7_joint_from_scratch |  | 5 | 2.11 | 2.46 | 2.24 … 2.71 | 0.04 | 52 |
| dbp_Ns8_L31_R7_layerwise |  | 5 | 1.84 | 2.34 | 1.68 … 3.19 | 0.04 | 592 |
| dbp_Ns4_L31_R3_joint_from_scratch |  | 5 | 1.77 | 2.10 | 1.34 … 2.64 | 0.03 | 288 |
| dbp_Ns4_L31_R7_layerwise |  | 5 | 1.66 | 1.99 | 1.77 … 2.57 | 0.03 | 296 |
| dbp_Ns2_L31_R7_layerwise |  | 5 | 1.53 | 1.85 | 1.59 … 2.07 | 0.03 | 148 |
| dbp_Ns4_L7_R7_layerwise |  | 5 | 1.45 | 1.81 | 1.45 … 2.17 | 0.03 | 104 |
| complex_a | контроль | 2 | 60.60 | 62.19 | 62.02 … 62.37 | 1.28 | 6480 |

Вердикты по правилу до запуска (`results/dbp_depth_verdicts.csv`):

| hypothesis | candidate | candidate_macs | reference | diff_pp | seed_spread_pp | n_seeds | verdict |
|---|---|---|---|---|---|---|---|
| глубина помогает | dbp_Ns8_L7_R7_layerwise | 208 | Ns=1, L=7, R=7, layerwise | 1.45 | 1.06 | 5 | подтверждено |
| послойная инициализация нужна | dbp_Ns8_L7_R7_layerwise | 208 | joint_from_scratch, Ns=8, L=7, R=7 | -6.17 | 1.65 | 5 | не доказано |

Рисунки (клик — крупно, листание стрелками):

![E018_quality_vs_macs.png](../figs/experiments/E018_dbp_depth/figures/E018_quality_vs_macs.png){group="E018"}

![E018_quality_vs_ns.png](../figs/experiments/E018_dbp_depth/figures/E018_quality_vs_ns.png){group="E018"}
