<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Можно ли доверять числам E010–E021: нет ли утечки последовательностей между train и test, верен ли сырой BER, правильно ли посчитаны умножения (RM), одинаково ли настраивались семейства, каков шумовой пол и почему агрегатор E021 пропускал точки?

**Правило до запуска** (`experiments/E024_data_audit/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `leakage`: no leakage if for K = 12 the share of matched test->train windows <= 5 x random and < 1 %; otherwise results/gate_status.json halt: true (queue stops), dedup test + v4 h119 1 seed 60k
- `raw_ber`: recomputed raw test BER (window-31 centres) must equal 0.009551859551859552 (E004-E023)
- `rm`: every stored macs_per_symbol(_merged) of E010-E023 must equal the independent formula; mismatches listed
- `noise_floor`: minimal detectable difference with 5 seeds = t_{0.975,8} s_pooled sqrt(2/5) (E019 configurations)
- `aggregator`: a point = one configuration with >= 3 seeds; family from results/family_map.csv; level window +-25 % on a log scale; best by validation
- `otherwise`: не доказано
