<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** —

**Правило до запуска** (`experiments/E050_dbpx_physical/config.json` → `rule_draft`, дословно):

- `audit`: if the first Kerr of E018/E028/E046 acts on the CDC signal without a dispersion block, write 'DBP E028 это не классический DBP'
- `start_signal`: DBP-X at the start (no training) shows a signal against the E028 DBP if its test reduction (honest lengths) exceeds the E028 mean at the nearest RM (xiao column for both) by more than 2 std of E028 (and 2 SE); one start value per config (no seeds), so only the E028 spread enters
- `train_signal`: a trained DBP-X arm beats the E028 reference at the same or lower RM if mean test difference > 2 SE (SE = sqrt(s1^2/n1 + s2^2/n2), 10 seeds each) AND > 2 std (max of the two std); one stage vs two stages: the same rule; convergence: plateau iteration, mean over seeds
- `not_proven`: otherwise 'не доказано'; literature statements (Xiao: two stages better, K = 0/1 optimum) are only hypotheses on our data
- `limits`: one channel, one polarization, 1 sps simulation; files close to periodic, so circular processing is a property of the data; Rs and beta2 from literature (the multiplier is fitted on validation)
