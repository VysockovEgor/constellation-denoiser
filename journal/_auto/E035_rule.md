<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Воспроизводит ли обучение градиентом триплеты закрытой формы, и добавляет ли маленькая Kerr-сеть на остаток?

**Правило до запуска** (`experiments/E035_triplet_hybrid/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `hybrid`: a Kerr MLP on the residual helps if th_M*_h* - tg_M*_warm (same M) >= 1 p.p. (5 seeds, intervals do not overlap)
- `gradient_warm`: gradient training keeps the closed form if |tg_M*_warm - closed form (E031/E034, same M)| < 1 p.p.
- `gradient_scratch`: gradient from scratch reaches the closed form if tg_M15_scratch >= closed form - 1 p.p.
- `otherwise`: не доказано
