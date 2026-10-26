<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Выходят ли лучшие конфигурации E016–E018 и complex_a/complex_both на плато за 100k–200k шагов?

**Правило до запуска** (`experiments/E019_confirm_long_runs/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `plateau`: as E011: improvement of the smoothed (10k window) validation reduction over the last 50k steps < 0.2 p.p.; otherwise 'plateau not reached within N steps' and y_inf only as a rough estimate
- `long_vs_short`: the 200k result is reported next to the 20k/40k result of the same configuration in its experiment; no verdict beyond the plateau
