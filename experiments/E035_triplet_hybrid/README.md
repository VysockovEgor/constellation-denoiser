# E035 — Триплеты градиентом и триплеты + Kerr-сеть

**Вопрос.** Воспроизводит ли обучение градиентом триплеты закрытой формы, и добавляет ли маленькая Kerr-сеть на остаток?

**Правило (до запуска, черновик).** Сеть на остаток помогает, если th − tg_warm (то же M) ≥ 1 п.п. (5 seeds, интервалы не пересекаются); градиент сохраняет закрытую форму, если |tg_warm − закрытая форма E031/E034| < 1 п.п.; градиент с нуля достигает её, если tg_M15_scratch ≥ закрытая форма − 1 п.п. Полностью — `config.json` → `rule_draft`.

**Модель** (`triplets_ext.TripletNet`, вид `triplet_net`): y = x_c + Σ C T(x) + КИХ7(x) (+ one_hidden Kerr MLP на остаток, выход с нуля); окно W ≥ 2M + 1 (M = 15: W = 31; M = 21: W = 43). RM = 12·членов + 28 + h(4W + 11). Тёплый старт — коэффициенты закрытой формы на всех 30 файлах train, λ по валидации (стадия `fit`).

**Конфигурации.** tg_M15_scratch, tg_M15_warm, th_M15_h4/h8/h16, tg_M21_warm, th_M21_h8/h16. S1: lr cosine_to_zero {1e-4, 3e-4, 1e-3} (c_mult 30) × 2 seeds × 20k → S2: лучший по валидации × 5 seeds × 200k.

**Команда.** `bash experiments/E035_triplet_hybrid/run.sh <стадия>`; стадии: control | fit | s1 | s2 | all | summarize | smoke.

**Результаты.** `results/triplet_net_summary.csv`, `triplet_net_verdicts.csv`; рисунок `figures/E035_triplet_net.png`.
