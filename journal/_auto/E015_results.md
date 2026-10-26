<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

Прямая подгонка фазы (`results/alpha_fit_summary.csv`):

| R | выборка | снижение BER, % | 95 % бутстрап | Q², дБ |
|---|---|---|---|---|
| 1 | validation | -0.75 | -0.86 … -0.61 | -0.01 |
| 1 | test | -0.59 | -0.77 … -0.44 | -0.01 |
| 3 | validation | -0.08 | -0.36 … 0.19 | -0.00 |
| 3 | test | -0.25 | -0.49 … 0.03 | -0.00 |
| 7 | validation | -0.17 | -0.47 … 0.09 | -0.00 |
| 7 | test | -0.29 | -0.56 … -0.02 | -0.00 |
| 15 | validation | -0.53 | -0.84 … -0.19 | -0.01 |
| 15 | test | -0.34 | -0.66 … -0.08 | -0.01 |
| 31 | validation | -0.14 | -0.43 … 0.19 | -0.00 |
| 31 | test | -0.13 | -0.51 … 0.23 | -0.00 |

Спирмен ρ(dphi, мощность), все окружности (`results/diagnostics_summary.csv`):

| выборка | предиктор | ρ | 95 % бутстрап |
|---|---|---|---|
| validation | own_power | 0.06 | 0.058 … 0.065 |
| validation | P_nb_R1 | 0.10 | 0.102 … 0.107 |
| validation | P_nb_R3 | 0.15 | 0.148 … 0.155 |
| validation | P_nb_R7 | 0.19 | 0.188 … 0.197 |
| validation | P_nb_R15 | 0.23 | 0.224 … 0.235 |
| test | own_power | 0.06 | 0.060 … 0.068 |
| test | P_nb_R1 | 0.11 | 0.106 … 0.110 |
| test | P_nb_R3 | 0.15 | 0.150 … 0.156 |
| test | P_nb_R7 | 0.20 | 0.192 … 0.201 |
| test | P_nb_R15 | 0.23 | 0.230 … 0.242 |

Вердикт (`results/verdict.json`): **не доказано**.

- `condition_a_by_R`: {"P_nb_R1": true, "P_nb_R3": true, "P_nb_R7": true, "P_nb_R15": true}
- `condition_a`: true
- `rho_own_validation`: 0.06220118687169737
- `rho_nb_validation`: {"P_nb_R1": 0.10501116310423834, "P_nb_R3": 0.15146576337228318, "P_nb_R7": 0.19251645428787592, "P_nb_R15": 0.22973989688453042}
- `rho_nb_test`: {"P_nb_R1": 0.10779719225033893, "P_nb_R3": 0.15329486107182888, "P_nb_R7": 0.19641555457792018, "P_nb_R15": 0.23544732017550882}
- `fit_gap_R7_minus_R1_validation_pp`: 0.5826476923620085
- `fit_gap_R7_minus_R1_test_pp`: 0.2951845792730551
- `condition_b`: false
- `test_agrees`: true

Рисунки (клик — крупно, листание стрелками):

![E015_alpha_profile.png](../figs/experiments/E015_physics_diagnostics/figures/E015_alpha_profile.png){group="E015"}

![E015_dphi_heatmap.png](../figs/experiments/E015_physics_diagnostics/figures/E015_dphi_heatmap.png){group="E015"}

![E015_dphi_vs_power.png](../figs/experiments/E015_physics_diagnostics/figures/E015_dphi_vs_power.png){group="E015"}

![E015_fit_vs_R.png](../figs/experiments/E015_physics_diagnostics/figures/E015_fit_vs_R.png){group="E015"}
