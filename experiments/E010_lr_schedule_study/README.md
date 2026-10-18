# E010 — Расписания learning rate

## Вопрос
Сокращает ли больший пиковый lr, линейный warmup, отдельный повышенный lr для
коэффициентов Керра `c` и затухание со старта с большого lr число шагов до
заданного качества (замечание Сидельникова: начинать с большего lr и потом
снижать)? Сравнение по числу шагов до 90 % и 95 % качества контроля.

## Правило интерпретации (до запуска) — ЧЕРНОВИК
Статус: черновик, пользователь подтверждает его до первого запуска; после
просмотра результатов не меняется. Копия в `config.json` → `rule_draft`.

Конфигурация **лучше контроля** (const 1e-3, 20k шагов, та же архитектура),
если одновременно:
1. средний по seeds `T95` не более 0.7 от контрольного `T95`;
2. среднее тестовое снижение BER в конце не хуже контроля более чем на 0.5 п.п.;
3. ни один seed не разошёлся: нет NaN, финальное валидационное снижение каждого
   seed не ниже среднего контроля более чем на 2 п.п.

Иначе вывод «не доказано». Разброс по seeds (min, max, std) приводится всегда.

`T90`/`T95`: первый шаг, после которого валидационное снижение BER три замера
подряд (замер каждые 250 шагов) не ниже 90 % / 95 % финального валидационного
снижения контроля (среднее по seeds). Выбор конфигураций (`select`, `stage3`)
делается только по валидации; тест только для отчёта.

## Протокол
- Модели: окно 31, hidden 48, два скрытых слоя, Glorot, `c` с нуля.
  `both` — Керр после обоих слоёв (`PaperComplexMLP`), `a` — Керр только после
  2-го слоя (`AblationComplexMLP`, вариант a из E009). Классы не изменялись.
- Данные: 30 train-файлов `train_pool_files[:30]`, 45 validation и 45 test из
  `experiments/E004_training_file_curve/config.json` — тот же split, что в E007–E009.
- Adam, batch 1024, complex MSE. lr для `c` = `c_mult` · lr остальных весов
  (`src/constellation_denoiser/lr_schedules.py`). Расписания: `const`,
  `cosine_to_zero`, `cosine_to_floor` (до 10 % пика); перед ними линейный warmup
  `warmup` шагов; горизонт затухания равен числу шагов прогона.
  Значения по умолчанию (const 1e-3, warmup 0, `c_mult` 1) воспроизводят E007–E009.
- Валидация и тест каждые 250 шагов. Для сравнения с E007–E009 также
  считается `test_reduction_valbest_select`: тест в точке лучшей валидации среди
  шагов, кратных 1000 (так выбирались числа в E007–E009).

## Стадии и число прогонов
Seeds 17, 29, 43. Run id не содержит имя стадии, поэтому совпадающие
конфигурации считаются один раз и переиспользуются.

| стадия | что | прогонов (новых) | шагов на прогон |
|---|---|---|---|
| `baseline-check` | const 1e-3 для `both`, `a`; cosine_to_zero 1e-3 для `both` (проверка E008) | 9 | 20 000 |
| `stage1` | пик {1e-3, 3e-3, 1e-2} × {const, cosine_to_zero, cosine_to_floor} × 2 архитектуры | 54 (45) | 20 000 |
| `select` | 2 лучшие конфигурации stage1 на архитектуру по валидации → `results/stage1_selection.json` | 0 | — |
| `stage2` | выбранные × warmup {0, 1000} × `c_mult` {1, 10, 30} | 72 (60) | 20 000 |
| `stage3` | лучшая по валидации из stage1+2 и контроль; бюджет {5k, 10k, 20k}, горизонт = бюджет | 36 (24) | 5 000–20 000 |

Всего около 138 уникальных прогонов, около 2.5 млн шагов.

Контроль (допуск ±0.7 п.п. по среднему seeds 17 и 29, метрика
`test_reduction_valbest_select`): `both` 61.727 %
(`experiments/E007_window_width_sweep/results/complex_mlp_sweep_summary.csv`),
`a` 62.191 % (`experiments/E009_kerr_ablation/results/complex_mlp_ablation_summary.csv`),
`both` cosine_to_zero 58.763 % (`experiments/E008_cosine_lr_coefficients/results/complex_mlp_lr_summary.csv`).
Если контроль не прошёл, `results/control_check.json` содержит `passed: false`,
и стадии `stage1`–`stage3` отказываются запускаться.

## Команды (из корня репозитория)
```bash
bash experiments/E010_lr_schedule_study/run.sh probe            # один раз на сервере
bash experiments/E010_lr_schedule_study/run.sh plan all         # число прогонов и оценка времени
bash experiments/E010_lr_schedule_study/run.sh baseline-check   # контроль + проверка допуска
bash experiments/E010_lr_schedule_study/run.sh stage1
bash experiments/E010_lr_schedule_study/run.sh select
bash experiments/E010_lr_schedule_study/run.sh stage2
bash experiments/E010_lr_schedule_study/run.sh stage3
bash experiments/E010_lr_schedule_study/run.sh summarize
SMOKE=1 bash experiments/E010_lr_schedule_study/run.sh smoke    # проверка пайплайна, results_smoke/
```
Переменные: `JOBS`, `THREADS` (по умолчанию `auto` из `resources.json`),
`DEVICE=auto|cpu|cuda`, `SEEDS=17,29`, `FORCE=1` (пересчитать готовое),
`TEST_EVERY=1000` (реже считать тест; на обучение и выбор не влияет).

## Оценка времени
Оценку печатает `run.sh plan <стадия>` на сервере по скорости, измеренной
`probe`. Локальный smoke (300 шагов, 2 файла) показывает только порядок
величины: на этой машине один шаг ≈ 12 мс на 1 поток, а оценка на 45+45 файлах
занимает ≈ 10 с на точку, то есть при замере каждые 250 шагов валидация и тест
занимают около 3/4 времени прогона. На сервер эти числа переносить нельзя.

## Выходы
- `results/runs/*.json` — по прогону; `results/curves/*.csv` — кривые
  (val каждые 250 шагов, test только для отчёта).
- `results/lr_sweep.csv` — все прогоны с `T90`, `T95`;
  `results/lr_sweep_summary.csv` — средние, разброс, вердикт по черновику правила.
- `results/stage1_selection.json`, `results/stage3_choice.json`, `results/control_check.json`.
- `figures/E010_val_curves.png`, `figures/E010_t95_vs_peak_lr.png`.
- Веса: `checkpoints/E010/` (не в Git).

## Ограничения
- Один фиксированный split; 3 seed (если бюджет вынудит оставить 17 и 29, это
  указывается здесь и в LOG).
- `T95` привязан к контролю 20k шагов; для расписаний с затуханием качество
  в середине прогона зависит от горизонта, поэтому stage3 повторяет лучшую
  конфигурацию с горизонтом, равным бюджету.
- lr для `c` задаётся одной группой на все слои: `c` хранится по одному числу
  на слой, а не на нейрон.

## Что делать с результатами
Запустить `summarize`, забрать `results/*.csv`, `results/*.json` (кроме
`runs/`) и `figures/`, дописать запись в конец `LOG.md` с числами и ссылками на
файлы. `stage1_selection.json` читают E011 и E012 (ветка `best_e010`).
