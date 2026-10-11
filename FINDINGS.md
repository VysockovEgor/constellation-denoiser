# Подтверждённые численные факты

«Предварительно» означает, что результат присутствует в существующем файле, но
ограничения опыта не позволяют считать его независимым окончательным
подтверждением.

| № | утверждение | уровень (предварительно/подтверждено) | эксперименты | ограничения |
|---:|---|---|---|---|
| 1 | `model_comparison_summary.csv` содержит test BER mean 0.009120466 для Transformer; test raw BER — 0.010219502. | предварительно | [E001](./experiments/E001_pilot_models/README.md) | Три seed; PR1–PR7 train, PR8 validation, PR9–PR10 test; один фиксированный split. |
| 2 | `paper_real_mlp.csv` содержит test BER 0.008690534 для seed 43; `paper_complex_mlp.csv` содержит 0.010211569 для seed 17. | предварительно | [E002](./experiments/E002_tanh_complex_paper/README.md) | Значения отдельных строк, три seed; PR9–PR10 использовались ранее. |
| 3 | `large_scale_pilot_summary.csv` содержит complex test BER 0.003528529 для seed 29 и raw test BER 0.009551860. | предварительно | [E003](./experiments/E003_large_scale_pilot/README.md) | Выборка 300 из 1 356 уникальных файлов; split 210/45/45; три seed. |
| 4 | `learning_curve_seed17.csv` содержит complex test BER 0.010160273 при N=7 и 0.003578 при N=210. | предварительно | [E004](./experiments/E004_training_file_curve/README.md) | Seed 17; ранее фиксированный holdout из 45 test-файлов. |
| 5 | `fixed_step_learning_curve.csv` содержит complex, N=210, step=20000, test BER 0.004626799. | предварительно | [E005](./experiments/E005_equal_steps/README.md) | Один фиксированный split; CSV не содержит seed. |
| 6 | В `plateau_seed_study.csv` максимум step равен 100000 для N=7/seed 17 и N=210/seed 17; N=7/seed 29 и 43 представлены до 20000. | подтверждено — покрытие файла | [E006](./experiments/E006_plateau_seeds/README.md) | Покрытие seed/N различается; отсутствующие комбинации не подтверждены. |
| 7 | `complex_mlp_sweep_summary.csv` содержит при window=31, hidden=48 mean test BER 0.003655749. | предварительно | [E007](./experiments/E007_window_width_sweep/README.md) | Два seed; один split. |
| 8 | `complex_mlp_lr_summary.csv` при window=31, hidden=48 содержит reduction A=61.727363% и B=58.763070%. | предварительно | [E008](./experiments/E008_cosine_lr_coefficients/README.md) | Два seed; один split; график c отображает только коэффициент c. |
| 9 | `complex_mlp_ablation_summary.csv` содержит mean improvement 62.191479% для variant a; merge-check содержит max abs diff 4.549127e-07, 1000 окон, `ok=True` для variant a/seed 17. | предварительно | [E009](./experiments/E009_kerr_ablation/README.md) | Два seed; проверка merge приведена только для конкретных variant/seed. |

Полные строки и остальные результаты находятся в CSV соответствующих
`experiments/E00N_*/results/`. Здесь не сделано обобщений о воспроизводимости.
