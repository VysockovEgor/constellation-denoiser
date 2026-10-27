# E020 — Аномалия гибрида E017 в E019

**Вопрос.** Почему гибрид Phys + MLP (h48, joint) с веткой `best_e010` (cosine_to_zero, пик 3e-3, `c_mult` 1) даёт в E019 51.5 % вместо 62.8 % у const 1e-3?

**Правило (до запуска, черновик).** Плечо «восстанавливает уровень», если его средний тест ≥ 61.039 % (минимум нормального прогона E019); A1 (`c_mult` 30) или A2 (const) — причина в сочетании `c_mult` + расписание; только A3 (cosine, пик 1e-3) — причина в пиковом lr; ни одно — проверять код. Полностью — `config.json` → `rule_draft`.

**Без обучения** (`run.sh analysis`): переобучение и выбор чекпоинта отвергнуты по данным E019 (`results/hypotheses.csv`, `results/differences.csv`, `figures/E020_val_test_vs_steps.png`).

**Команда.** `bash experiments/E020_e017_anomaly/run.sh <стадия>`; стадии: analysis | probe | plan | estimate | list | control | main | summarize | smoke. Перезапуски A0–A3 — 3 seeds, 40k и 200k (длинные прогоны входят в `main`).

**Результаты.** `results/arms.csv`, `results/arms_verdicts.csv`; параметры и гипотезы — `config.json`.
