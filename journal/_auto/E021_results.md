<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вердикт** (по правилу до запуска): подтверждено: 3, не доказано: 1.

- 6500: complex Kerr MLP лучше Phys + MLP: **подтверждено**
- 6500: complex Kerr MLP лучше KAN: **подтверждено**
- 6500: Phys + MLP лучше KAN: **не доказано**
- 16000: complex Kerr MLP лучше real tanh MLP: **подтверждено**

Результатов нет (эксперимент не запускался).

Лучшее по валидации семейство на каждом уровне (`results/level_best.csv`):

| уровень | семейство | точка | умножений | запусков (seeds) | валидация | тест, % | мин | макс | Q², дБ |
|---|---|---|---|---|---|---|---|---|---|
| 2000 | Phys + MLP | E017 · h16_joint · const_1e-3 · 40k | 2198 | 5 | 41.35 | 42.91 | 42.38 | 43.37 | 0.78 |
| 6500 | complex Kerr MLP | E019 · control_complex_a · best_e010_cm30 · 200k | 6480 | 5 | 64.87 | 66.01 | 65.15 | 66.68 | 1.40 |
| 6500 | Phys + MLP | E019 · E017_hyb_R31_L7_sym_h48_joint · const_1e-3 · 200k | 6518 | 5 | 61.60 | 62.84 | 61.04 | 64.27 | 1.30 |
| 6500 | KAN | E014 · k4_cascade_staged · const_1e-3 · 40k | 8088 | 3 | 60.23 | 61.77 | 60.57 | 63.06 | 1.27 |
| 16000 | complex Kerr MLP | E011 · complex_both · const_1e-3 · 200k | 16032 | 3 | 66.38 | 67.21 | 67.00 | 67.32 | 1.44 |
| 16000 | real tanh MLP | E011 · real_tanh · const_1e-3 · 200k | 15360 | 3 | 32.09 | 32.19 | 31.88 | 32.75 | 0.55 |
| 40000 | complex Kerr MLP | E012 · v4_one_hidden_wide · const_1e-3 · 100k | 34254 | 3 | 67.18 | 68.06 | 59.15 | 74.47 | 1.48 |

Вердикты по правилу до запуска (`results/level_verdicts.csv`):

| level | гипотеза | разница, п.п. | intervals_overlap | вердикт |
|---|---|---|---|---|
| 6500 | 6500: complex Kerr MLP лучше Phys + MLP | 3.18 | нет | подтверждено |
| 6500 | 6500: complex Kerr MLP лучше KAN | 4.24 | нет | подтверждено |
| 6500 | 6500: Phys + MLP лучше KAN | 1.06 | да | не различимы |
| 16000 | 16000: complex Kerr MLP лучше real tanh MLP | 35.02 | нет | подтверждено |

Рисунки (клик — крупно, листание стрелками):

![E021_pareto_ber.png](../figs/experiments/E021_pareto/figures/E021_pareto_ber.png){group="E021"}

![E021_pareto_q2.png](../figs/experiments/E021_pareto/figures/E021_pareto_q2.png){group="E021"}
