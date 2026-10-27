# E026 — Карта сложности простой Kerr-сети

**Вопрос.** Какая простая Kerr-сеть лучшая в области 100–40 000 RM, и где окупаются окно шире 31 или глубина?

**Правило (до запуска, черновик).** Ширина окна окупается при данном RM, если лучшая сеть с w > 31 выше лучшей с w = 31 на ≥ 0.5 п.п. (3 seeds, интервалы не пересекаются; RM-окна ±25 %); глубина (both) окупается, если both лучше one_hidden при равном RM на ≥ 0.5 п.п.; насыщение — наименьший RM, после которого удвоение RM даёт < 0.5 п.п. Полностью — `config.json` → `rule_draft`.

**Сетка.** one_hidden (`complex_one_hidden`, RM = h(4w + 11)): w ∈ {5, 9, 15, 21, 31, 41, 51, 65} × h ∈ {2 … 256}, RM ∈ [60, 45 000], рецепт F2, 3 seeds × 100k; both w ∈ {15, 31} × h ∈ {8, 16, 32, 48, 64}, рецепт F4; confirm: до 12 точек фронта (RM, валидация) × 5 seeds × 200k. Для w = 65 — доля разошедшихся seeds.

**Команда.** `bash experiments/E026_kerr_complexity_map/run.sh <стадия>`; стадии: control | grid | confirm | all | summarize | smoke | plan | estimate.

**Результаты.** `results/map_summary.csv`, `results/map_verdicts.csv`, `results/instability.csv`; рисунки `figures/E026_*.png`.
