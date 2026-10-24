# Runbook: ночной запуск E015–E019

Коротко. Подробности про tmux, probe, остановку и возврат результатов — в
`experiments/RUNBOOK_E010_E014.md`: там всё то же самое.

## 1. Подключиться и обновить ветку
```bash
ssh user@server                      # адрес свой
cd ~/constellation-denoiser
git fetch origin
git checkout exp/E015-E018
git pull
```

## 2. Проверить данные и ресурсы
```bash
PYTHONPATH=src python scripts/check_data.py                   # «Данные на месте.»
bash experiments/E016_phys_kerr_only/run.sh probe             # 2–3 мин, пишет resources.json
```

## 3. Оценить время на этом сервере (по 1–2 мин на стадию)
```bash
for e in E016_phys_kerr_only E017_phys_plus_mlp E018_dbp_depth; do bash experiments/$e/run.sh estimate all; done
CONFIRM_ESTIMATE_FROM_SMOKE=1 bash experiments/E019_confirm_long_runs/run.sh estimate all
```
Оценки ниже измерены на WSL (12 потоков, JOBS=2, THREADS=1) и на сервер
**не переносятся** — запускайте `estimate` там. E015 считается минуты.

| стадия | прогонов | оценка часов (WSL, JOBS=2) |
|---|---|---|
| E015 diagnostics | 1 анализ | минуты |
| E016 control / main | 5 / 555 | 0.2 / 6.0 |
| E017 control / main | 10 / 40 | 0.4 / 2.3 |
| E018 control / main | 2 / 160 | 0.1 / 61.8 |
| E019 control / main (confirm) | 2 / 75 | 0.1 / 19.4 |
| **part1** (E015–E017) | | **≈ 9** |
| **part2** (E018 + confirm) | | **≈ 81** |
| **всего** | | **≈ 90** |

Сумма больше 48 часов, поэтому запуск разбит на две команды. Основная доля —
E018: при Ns = 8 один прогон идёт 170k шагов. На сервере с большим `JOBS`
время сократится примерно пропорционально, но не ниже самого длинного
прогона (~2.4 ч на WSL).

Если сумма больше 48 часов, запускайте двумя командами (шаг 4) и смотрите
результаты между ними. Сетки не сокращены.

## 4. Запуск в tmux одной строкой
```bash
tmux new -s night
bash scripts/overnight_e015_e018.sh part1      # E015 -> E016 -> E017
# отключиться: Ctrl+b, затем d; вернуться: tmux attach -t night
```
После `part1`:
```bash
bash scripts/overnight_e015_e018.sh part2      # E018 -> E019 (confirm) -> итоговая сводка
```
Если сумма меньше 48 ч, можно сразу запустить `bash scripts/overnight_e015_e018.sh all`.
Повторный запуск той же команды пропускает готовые прогоны, а длинные
продолжает с чекпоинта.

## 5. Прогресс и ошибки
```bash
tail -f logs/overnight_part1_*.log
grep -n "ОШИБКА" logs/overnight_*.log                          # пусто — всё хорошо
ls experiments/E016_phys_kerr_only/results/runs/*.json | wc -l  # готово из 560
uptime
```
`ОШИБКА: E0xx прерван` означает, что стадия этого эксперимента не прошла, а
цепочка перешла к следующему. Причина — в `experiments/E0xx_*/logs/*.log`.
Если не прошёл контроль, смотрите `results/control_check.json`. Прогон с
`"diverged": true` ошибкой не считается.

## 6. Остановить
В tmux нажать `Ctrl+C`. Либо из другого терминала:
`kill $(cat experiments/E016_phys_kerr_only/logs/main.pid)`.

## 7. Собрать результаты и принести через git
```bash
git add experiments/E015_physics_diagnostics/results/*.csv experiments/E015_physics_diagnostics/results/*.json \
        experiments/E015_physics_diagnostics/figures/*.png \
        experiments/E01[6-9]_*/results/*.csv experiments/E01[6-9]_*/results/*.json \
        experiments/E01[6-9]_*/results/curves experiments/E01[6-9]_*/figures/*.png experiments/E01[5-9]_*/logs/*.log \
        results/E015_E018_verdicts.csv results/E015_E018_quality_points.csv results/E015_E018_quality_vs_macs.png \
        results/q2_key_points.csv logs/overnight_*.log
git status            # в списке не должно быть data/, checkpoints/, resources.json, results/runs/
git commit -m "E015-E019: результаты запуска на сервере"
git push
```
Архивом вместо git:
`tar czf e015_e019_results.tgz experiments/E01[5-9]_*/results experiments/E01[5-9]_*/figures experiments/E01[5-9]_*/logs results logs --exclude='*/runs/*'`,
затем `scp` архива в WSL.

Итог: `results/E015_E018_verdicts.csv` (одна строка на гипотезу) и
`results/E015_E018_quality_vs_macs.png`.
