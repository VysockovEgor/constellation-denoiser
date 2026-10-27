# E029 — Физический front-end против простого MLP по ширине

**Вопрос.** Окупается ли физическая часть при каждой ширине или только при малой (E020: гибрид h48 cm30 67.6 % против 66.0 % у класса a, 3 против 5 seeds)?

**Правило (до запуска, черновик).** Front-end окупается при ширине h, если гибрид − чистая ≥ 1 п.п. (5 seeds, интервалы не пересекаются); выигрыш убывает с h, если разность при h = 16 больше, чем при h = 119, на ≥ 1 п.п. Выигрыш только при малых h — это тезис «физика экономит умножения». Полностью — `config.json` → `rule_draft`.

**Пары.** `complex_one_hidden` w31 против `phys_plus_mlp` (PhysKerr R31 L7 sym + one_hidden, параллельно, joint), h ∈ {8, 16, 32, 48, 96, 119, 192}; рецепт пары E025 с ближайшей шириной (F1/F2/F3 и F8/F9); варианты front-end при h = 48: (R7, L7), (R31, L7), (R31, L15). 5 seeds × 200k.

**Команда.** `bash experiments/E029_frontend_vs_width/run.sh <стадия>`; стадии: control | main | all | summarize | smoke.

**Результаты.** `results/width_summary.csv`, `results/width_verdicts.csv`; рисунок `figures/E029_hybrid_vs_width.png`.
