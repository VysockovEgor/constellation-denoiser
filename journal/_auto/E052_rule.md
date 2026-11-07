<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** —

**Правило до запуска** (`experiments/E052_dbpx_triplets/config.json` → `rule_draft`, дословно):

- `reached`: in a given convention: Ns 8 + triplets (mean over 5 replicas) >= DBP-X Ns 40 test value at RM <= RM(Ns 40)
- `not_worse`: gap (Ns40 - Ns8+T) < 2 std and < 2 SE of the replicas (Ns 40 is one deterministic value: no seeds)
- `otherwise`: not reached
- `limits`: no training of DBP-X; one channel, one polarization, 1 sps; files close to periodic (circular filters); spread only from the ridge replicas
