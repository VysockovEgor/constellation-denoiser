<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** —

**Правило до запуска** (`experiments/E056_kerr_offaxis_ablation/config.json` → `rule_draft`, дословно):

- `level_choice`: per arm and RM level: the configuration (M, N_c, pruning set, lambda) with RM <= level * 1.05 and the best mean validation over 5 replicas; test reported (mean, SE, std)
- `memory_effect`: arm 4 - arm 2 at the same level > 2 SE and > 2 std: memory of the Kerr phase matters
- `offaxis_effect`: arm 4 - arm 3 (best Kerr-only, any RM <= level) > 2 SE and > 2 std: off-axis terms matter
- `form_effect`: arm 2 - arm 1 at the same level > 2 SE and > 2 std: the exponential instantaneous Kerr + off-axis beats free axial coefficients
- `otherwise`: не доказано
- `limits`: 5 bootstrap replicas of the train files (closed form), one split (E004); Kerr-only arms have one fixed RM each (35-66), they cannot be placed at higher levels
