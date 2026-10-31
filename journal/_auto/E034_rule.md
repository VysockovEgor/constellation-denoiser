<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Растёт ли качество T3 + КИХ дальше M = 21, и сколько членов-триплетов достаточно (RM против качества)?

**Правило до запуска** (`experiments/E034_triplets_extended/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `memory`: growth with M continues if T3 + FIR7 at M = 31 - M = 21 (E031) >= 1 p.p. (test mean, 5 replicas)
- `pruning`: K = 128 terms keep >= 90 % of the reduction of the full M = 25 model
- `vs_networks`: pruned triplets beat the Kerr MLP at equal RM if Q2 gain difference >= 0.1 dB with non-overlapping intervals (checked in E033 with the E025 recipes)
- `otherwise`: не доказано
