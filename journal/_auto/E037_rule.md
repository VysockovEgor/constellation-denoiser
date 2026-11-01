<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Обходят ли триплеты нейросети на каждом разбиении файлов, и устойчив ли их результат?

**Правило до запуска** (`experiments/E037_triplets_splits/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `beats_networks`: T3 + FIR7 at M = 21 beats the best E023 network of every split (min of triplets > max of network)
- `stable`: spread of the T3 + FIR7 M = 21 test means across the three splits < 2 p.p.
- `otherwise`: не доказано
