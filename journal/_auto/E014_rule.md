<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Помогает ли лёгкий KAN (свой, B-сплайны) доочищать остаток после слоя Керра и стоит ли это умножений?

**Правило до запуска** (`experiments/E014_kerr_kan_hybrid/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `improves`: a hybrid improves the Kerr base if its mean test reduction exceeds k0_kerr (for k2) or k0b_kerr_continued (for k3, k4) at the same total number of steps by more than the seed spread (range) and by at least 0.5 p.p.
- `cost`: the added multiplies (dense and sparse, absolute and relative to k0_kerr) are reported next to the gain; 'worth it' is only concluded together with them
- `kerr_needed`: comparison with k1_kan_only answers whether the Kerr base is needed at all
- `metric`: test_reduction_valbest_select (test at the validation-best evaluation, every 1000 steps)
