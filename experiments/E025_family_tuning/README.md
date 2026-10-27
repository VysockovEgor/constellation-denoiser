# E025 — Честная настройка по семействам

**Вопрос.** Меняется ли рейтинг семейств и их абсолютные числа, если у каждого семейства одинаковый бюджет настройки?

**Правило (до запуска, черновик).** Новый рецепт заменяет прежний (`ref_current`), если на S3 средняя валидация выше на ≥ 0.5 п.п. и интервалы min–max по seeds не пересекаются; рейтинг изменился, если на одном уровне RM два семейства поменялись местами с разницей ≥ 1 п.п. против E021; F5 (чистая физика) была недообучена, если лучший рецепт ≥ ref_current + 3 п.п. Контроль: complex_a 20k 62.191 ± 0.7; кодовая проверка complex_a cm30 200k 66.253 ± 0.7 (seeds 17, 29). Полностью — `config.json` → `rule_draft`.

**Пары (классы в коде).** F1–F3 `complex_one_hidden` h16/48/119 (h119 = v4 E012); F4 `complex_both` h48; F5 `phys_kerr` R31 L7 sym; F6/F7 `phys_dbp` Ns8/16 L31 R7; F8/F9 `phys_plus_mlp` R31 L7 h16/48; F10/F11 `real_mlp` 1×100 / 2×96 (новый класс, при tanh и масштабе 1 совпадает с `real_tanh`); F12 `kerr_kan` k4 cascade grid 8, фазы staged; F13 `transformer` d64 L2.

**Поиск.** Sobol (`scipy.stats.qmc`, seed 2025): 32 точки + `ref_current` + `ref_e010best` (cos 3e-3, wu0, cm30); KAN 14 + 2, трансформер 10 + 2. Размер батча не настраивается: `runner.BATCH_SIZE = 1024` — константа раннера. Weight decay > 0 — через AdamW. Для физики `c_mult` — множитель lr коэффициентов α.

**Стадии.** S1: все конфигурации × 2 seeds × 20k (и кодовая проверка) → `check-code` → S2: топ-4 по валидации + ref_current × 3 seeds × 60k → S3: топ-2 + ref_current × 5 seeds × 200k (KAN 1 × 3 × 100k, трансформер 1 × 3 × 60k).

**Команда.** `bash experiments/E025_family_tuning/run.sh <стадия>`; стадии: control | s1 | check-code | s2 | s3 | all | summarize | smoke | plan | estimate.

**Результаты.** `results/recipes_table.csv`, `results/tuning_verdicts.csv`, `../../results/recipes.json` (читает `recipes.load_recipe`), `../../results/gate_g1.json`; рисунки `figures/E025_*.png`.
