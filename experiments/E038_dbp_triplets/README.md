# E038 — DBP + отобранные триплеты

**Вопрос.** Улучшают ли несколько самых сильных триплетов настроенный DBP в области 600–2 500 RM, сильнее, чем триплеты сами по себе?

**Правило (до запуска, черновик).** Триплеты добавляют к DBP, если (DBP + K) − DBP (то же Ns и рецепт) ≥ 1 п.п. и интервалы не пересекаются; DBP + K лучше одних триплетов (E031/E034, лучшие по валидации с RM не больше) на ≥ 0.1 дБ при непересекающихся интервалах. Полностью — `config.json` → `rule_draft`.

**Модель** (`triplets_ext.DBPTriplets`, вид `dbp_triplets`): y = DBP_Ns(x) + Σ по K сильнейшим членам T3 C_j T_j(x), C = 0 в начале. Члены ранжирует закрытая форма T3 + КИХ7 (M = 15) на всех 30 файлах train (стадия `select`). DBP L31 R7 sym, Ns 8 (рецепт F6) и 16 (F7); K ∈ {0, 16, 32, 64}; RM = 74·Ns + 12K. 5 seeds × 200k.

**Команда.** `bash experiments/E038_dbp_triplets/run.sh <стадия>`; стадии: control | select | main | all | summarize | smoke.

**Результаты.** `results/dbp_triplets_summary.csv`, `dbp_triplets_verdicts.csv`; рисунок `figures/E038_dbp_triplets.png`.
