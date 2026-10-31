<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вердикт** (по правилу до запуска): подтверждено: 4, не доказано: 0.

- порядок участников одинаков во всех разбиениях: **подтверждено**
- знак каждой попарной разницы сохраняется: **подтверждено**
- разброс средних между разбиениями < 2 п.п. у каждого участника: **подтверждено**
- вывод устойчив к смене разбиения: **подтверждено**

Показано 15 из 21 конфигураций: лучшие по валидации и контроли. Тест — в точке лучшей валидации, %; Q² — прирост в той же точке, дБ.

| конфигурация |  | запусков (seeds) | валидация | тест, среднее | тест, min … max | Q², дБ | умножений |
|---|---|---|---|---|---|---|---|
| complex_E021_kerr_oh_h296 · 303.0 |  | 5 | 72.26 | 73.30 | 73.18 … 73.44 | 1.67 | 39960 |
| complex_E021_kerr_oh_h296 · 202.0 |  | 5 | 71.89 | 72.11 | 72.03 … 72.20 | 1.63 | 39960 |
| complex_E021_kerr_oh_h296 · 101.0 |  | 5 | 71.65 | 72.89 | 72.79 … 73.03 | 1.67 | 39960 |
| phys_E021_hyb_h118 · 101.0 |  | 5 | 67.13 | 68.35 | 67.62 … 69.22 | 1.50 | 15968 |
| phys_E021_hyb_h118 · 303.0 |  | 5 | 67.12 | 68.46 | 67.35 … 69.49 | 1.49 | 15968 |
| phys_E021_hyb_h118 · 202.0 |  | 5 | 67.10 | 67.15 | 66.17 … 68.01 | 1.45 | 15968 |
| kan_E014_k4_cascade_staged · 202.0 |  | 5 | 65.52 | 65.66 | 65.23 … 66.42 | 1.40 | 8088 |
| kan_E014_k4_cascade_staged · 303.0 |  | 5 | 65.08 | 66.15 | 65.72 … 66.60 | 1.41 | 8088 |
| kan_E014_k4_cascade_staged · 101.0 |  | 5 | 64.91 | 65.66 | 65.01 … 66.55 | 1.40 | 8088 |
| dbp_E018_dbp_Ns8_L31_R7_joint_from_scratch · 202.0 |  | 5 | 42.93 | 43.57 | 41.16 … 46.63 | 0.80 | 592 |
| dbp_E018_dbp_Ns8_L31_R7_joint_from_scratch · 303.0 |  | 5 | 41.31 | 42.86 | 41.46 … 45.99 | 0.78 | 592 |
| dbp_E018_dbp_Ns8_L31_R7_joint_from_scratch · 101.0 |  | 5 | 41.20 | 41.62 | 39.62 … 46.89 | 0.76 | 592 |
| real_E011_real_tanh · 101.0 |  | 5 | 33.65 | 33.71 | 31.40 … 36.51 | 0.59 | 15360 |
| real_E011_real_tanh · 303.0 |  | 5 | 31.31 | 32.26 | 30.59 … 34.02 | 0.55 | 15360 |
| real_E011_real_tanh · 202.0 |  | 5 | 31.27 | 31.82 | 30.11 … 33.09 | 0.55 | 15360 |

Результаты по разбиениям (`results/split_summary.csv`):

| участник | разбиение | запусков (seeds) | тест, % | мин | макс | умножений |
|---|---|---|---|---|---|---|
| DBP: dbp_E018_dbp_Ns8_L31_R7_joint_from_scratch | 101 | 5 | 41.62 | 39.62 | 46.89 | 592 |
| DBP: dbp_E018_dbp_Ns8_L31_R7_joint_from_scratch | 202 | 5 | 43.57 | 41.16 | 46.63 | 592 |
| DBP: dbp_E018_dbp_Ns8_L31_R7_joint_from_scratch | 303 | 5 | 42.86 | 41.46 | 45.99 | 592 |
| KAN: kan_E014_k4_cascade_staged | 101 | 5 | 65.66 | 65.01 | 66.55 | 8088 |
| KAN: kan_E014_k4_cascade_staged | 202 | 5 | 65.66 | 65.23 | 66.42 | 8088 |
| KAN: kan_E014_k4_cascade_staged | 303 | 5 | 66.15 | 65.72 | 66.60 | 8088 |
| Phys + MLP: phys_E021_hyb_h118 | 101 | 5 | 68.35 | 67.62 | 69.22 | 15968 |
| Phys + MLP: phys_E021_hyb_h118 | 202 | 5 | 67.15 | 66.17 | 68.01 | 15968 |
| Phys + MLP: phys_E021_hyb_h118 | 303 | 5 | 68.46 | 67.35 | 69.49 | 15968 |
| PhysKerr: physkerr_E016_phys_R31_L7_sym | 101 | 5 | 8.90 | 8.54 | 9.05 | 38 |
| PhysKerr: physkerr_E016_phys_R31_L7_sym | 202 | 5 | 8.66 | 8.47 | 8.88 | 38 |
| PhysKerr: physkerr_E016_phys_R31_L7_sym | 303 | 5 | 9.12 | 8.90 | 9.33 | 38 |
| complex Kerr MLP: complex_E021_kerr_oh_h296 | 101 | 5 | 72.89 | 72.79 | 73.03 | 39960 |
| complex Kerr MLP: complex_E021_kerr_oh_h296 | 202 | 5 | 72.11 | 72.03 | 72.20 | 39960 |
| complex Kerr MLP: complex_E021_kerr_oh_h296 | 303 | 5 | 73.30 | 73.18 | 73.44 | 39960 |
| real tanh MLP: real_E011_real_tanh | 101 | 5 | 33.71 | 31.40 | 36.51 | 15360 |
| real tanh MLP: real_E011_real_tanh | 202 | 5 | 31.82 | 30.11 | 33.09 | 15360 |
| real tanh MLP: real_E011_real_tanh | 303 | 5 | 32.26 | 30.59 | 34.02 | 15360 |
| transformer: transformer_E021_tr_w7_d8_L2 | 101 | 5 | 10.90 | 8.21 | 12.80 | 8928 |
| transformer: transformer_E021_tr_w7_d8_L2 | 202 | 5 | 11.11 | 8.39 | 13.34 | 8928 |
| transformer: transformer_E021_tr_w7_d8_L2 | 303 | 5 | 10.72 | 8.63 | 13.30 | 8928 |

Вердикты по правилу до запуска (`results/robustness_verdicts.csv`):

| гипотеза | detail | вердикт |
|---|---|---|
| порядок участников одинаков во всех разбиениях | 101.0: complex Kerr MLP: complex_E021_kerr_oh_h296 > Phys + MLP: phys_E021_hyb_h118 > KAN: kan_E014_k4_cascade_staged > DBP: dbp_E018_dbp_Ns8_L31_R7_joint_from_scratch > real tanh MLP: real_E011_real_tanh > transformer: transformer_E021_tr_w7_d8_L2 > PhysKerr: physkerr_E016_phys_R31_L7_sym \| 202.0: complex Kerr MLP: complex_E021_kerr_oh_h296 > Phys + MLP: phys_E021_hyb_h118 > KAN: kan_E014_k4_cascade_staged > DBP: dbp_E018_dbp_Ns8_L31_R7_joint_from_scratch > real tanh MLP: real_E011_real_tanh > transformer: transformer_E021_tr_w7_d8_L2 > PhysKerr: physkerr_E016_phys_R31_L7_sym \| 303.0: complex Kerr MLP: complex_E021_kerr_oh_h296 > Phys + MLP: phys_E021_hyb_h118 > KAN: kan_E014_k4_cascade_staged > DBP: dbp_E018_dbp_Ns8_L31_R7_joint_from_scratch > real tanh MLP: real_E011_real_tanh > transformer: transformer_E021_tr_w7_d8_L2 > PhysKerr: physkerr_E016_phys_R31_L7_sym | подтверждено |
| знак каждой попарной разницы сохраняется | — | подтверждено |
| разброс средних между разбиениями < 2 п.п. у каждого участника | DBP: dbp_E018_dbp_Ns8_L31_R7_joint_from_scratch: 1.95; KAN: kan_E014_k4_cascade_staged: 0.49; Phys + MLP: phys_E021_hyb_h118: 1.31; PhysKerr: physkerr_E016_phys_R31_L7_sym: 0.46; complex Kerr MLP: complex_E021_kerr_oh_h296: 1.19; real tanh MLP: real_E011_real_tanh: 1.89; transformer: transformer_E021_tr_w7_d8_L2: 0.39 | подтверждено |
| вывод устойчив к смене разбиения | все три условия | подтверждено |

Рисунки (клик — крупно, листание стрелками):

![E023_split_robustness.png](../figs/experiments/E023_split_robustness/figures/E023_split_robustness.png){group="E023"}
