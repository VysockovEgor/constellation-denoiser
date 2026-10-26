<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Сколько даёт КИХ + нелинейная фаза с памятью (7–161 умножение) против комплексных MLP на 6–16 тыс.?

**Правило до запуска** (`experiments/E016_phys_kerr_only/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `physics_sufficient`: the PhysKerr configuration (any lr branch) with the best mean validation reduction among those with multiplies <= 10% of 6480 (648) has a mean test reduction over the 5 seeds no worse than complex_a (const 1e-3, 20k, same 5 seeds) by more than 2 p.p.
- `physics_useful`: reference = linear FIR + one Kerr coefficient (R = 1, L = 31, same sym, same lr branch); among configurations with R > 1 and multiplies <= the reference, the one with the best mean validation reduction beats the reference on test by >= 1 p.p. and by more than the seed spread
- `otherwise`: не доказано
