<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Сохраняется ли преимущество трансформера E001 при N = 30/210 и 20k/100k шагах против комплексной MLP?

**Правило до запуска** (`experiments/E013_transformer_scaling/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `outperforms`: at window 31, N=210 and the same number of steps, the transformer's mean test reduction exceeds that of the best complex MLP with no more multiplies per symbol by more than the seed spread and by at least 0.5 p.p.
- `comparability`: only matching windows and step counts are compared; if the transformer diverged or the step counts differ, the comparison is marked unequal
- `metric`: test reduction at the final step for the transformer (cosine ends at lr 0) and at the same step of the const-lr E011 curve for the complex MLP; seed spread = range
