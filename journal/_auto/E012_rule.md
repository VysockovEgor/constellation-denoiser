<!-- Сгенерировано scripts/journal_build_auto.py из файлов результатов. Не править вручную. -->

**Вопрос.** Возвращает ли второй слой преимущество при окнах 65/127 и 100k шагов (v1–v4 при равных умножениях)?

**Правило до запуска** (`experiments/E012_depth_vs_window/config.json` → `rule_draft`, дословно):

- `status`: DRAFT: to be confirmed by the user before the first run; must not change after results are seen
- `advantage`: for window W and equal step budget, the second layer 'regains the advantage' if v1 or v2 beats both v3 and v4 in mean test reduction by more than the seed spread and by at least 0.5 p.p.
- `optimization_effect`: if the v2 - v3 gap at 20k exceeds the seed spread but at 100k falls below the seed spread, interpret as an optimization effect, not expressiveness
- `otherwise`: not proven
