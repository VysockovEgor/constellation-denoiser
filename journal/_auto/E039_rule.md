<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Что можно узнать о триплетах из уже полученных результатов и подсчётов (определения E038 и K2048, конвенции RM, структура матрицы C, сырой BER), прежде чем тратить часы сервера?

**Правило до запуска** (`experiments/E039_triplet_diagnostics/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `structure`: the structure is exploitable if substituting a parametric form with <= 5 % of the free real parameters keeps >= 95 % of the gain of the free coefficients (gain = BER reduction in p.p. above M0 = FIR7 only, no triplets), evaluated at M = 31 on validation (test reported); otherwise E042 is closed
- `raw_ber`: the raw test BER of the window-31 centres must equal 0.009551859551859552 (E024); the plan value 9.58e-3 is checked against all splits
- `otherwise`: не доказано
