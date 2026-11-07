<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** —

**Правило до запуска** (`experiments/E055_small_real_mlp/config.json` → `rule_draft`, дословно):

- `reached`: a configuration reaches 55 % if its mean test reduction (5 seeds) >= 55 %; 'close' if 55 % is within 2 std of the mean; otherwise not reached
- `limits`: one recipe (no tuning: the F10 recipe), 200k steps; the external number is on other test files
