# E001 — Пилот моделей

**Вопрос.** Сравнить MLP, 1D-CNN, Bi-LSTM, Transformer и линейный FFE при окне 7 (split PR1–7 / PR8 / PR9–10).

**Правило (до запуска).** `unknown` — исторический эксперимент, правило не сохранилось
(`config.json` → `interpretation_rule`).

**Команда.** `bash experiments/E001_pilot_models/run.sh --help` — исторический запуск; `run.sh`
отказывается перезаписывать сохранённые результаты.

**Результаты.** `results/model_comparison_summary.csv`, `results/inference_benchmark.csv`; рисунки в `figures/`. Параметры, контроль и отклонения — `config.json`.
