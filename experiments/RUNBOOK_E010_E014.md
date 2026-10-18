# Runbook: запуск E010–E014 на сервере

Пошаговая инструкция. Все команды выполняются **из корня репозитория**
(папка `constellation-denoiser`), если не сказано иное. Блоки кода можно
копировать целиком.

Порядок экспериментов: **E010 → E012 → E011 → E013 → E014**. E010 даёт
lr-конфигурацию для веток `best_e010` в E011/E012; E012 короткий и отвечает на
вопрос про окно и глубину; E011 самый долгий; E013 для вердикта нужен E011.

Перед первым запуском подтвердите правила интерпретации (блоки «ЧЕРНОВИК» в
README каждой папки и `rule_draft` в `config.json`). После того как вы увидели
результаты, правила менять нельзя.

---

## A. Скачать код

Если репозиторий на сервере уже есть:
```bash
cd ~/constellation-denoiser          # путь подставьте свой
git fetch origin
git checkout exp/E010-E014
git pull
```
Если репозитория нет:
```bash
cd ~
git clone --branch exp/E010-E014 https://github.com/VysockovEgor/constellation-denoiser.git
cd constellation-denoiser
```
Если ветка не находится (`error: pathspec 'exp/E010-E014' did not match`),
проверьте, откуда тянется код:
```bash
git remote -v                 # должен быть .../VysockovEgor/constellation-denoiser.git
git branch -r | grep E010     # должна быть строка origin/exp/E010-E014
```
Если строки нет, ветка ещё не отправлена на GitHub: выполните `git push` с WSL.

## B. Данные

Нужны 300 CSV из `data/raw/archive/` (train-пул 210 файлов, validation 45,
test 45; split из `experiments/E004_training_file_curve/config.json`),
всего около **258 МБ**. Список — `experiments/E010_E014_required_files.txt`.

Проверка на сервере:
```bash
ls data/raw | head
ls data/raw/archive | wc -l                      # в WSL-копии 2000 файлов; для E010–E014 достаточно 300
ls data/manifest                                  # должен быть data_audit_all.csv
PYTHONPATH=src python scripts/check_data.py       # «Данные на месте.» — всё хорошо
PYTHONPATH=src python scripts/check_data.py --sha256   # то же с проверкой SHA-256 (несколько секунд)
```
`data/manifest/` лежит в Git, поэтому после `git checkout` он уже есть.
`data/raw/` в Git не входит, и его нужно скопировать.

Копирование недостающего **с WSL на сервер** (выполнять в WSL, из корня
репозитория; `user@server` и путь на сервере подставьте свои):
```bash
rsync -av --files-from=experiments/E010_E014_required_files.txt ./ user@server:~/constellation-denoiser/
```
Команда копирует только 300 нужных файлов с сохранением путей
`data/raw/archive/...`. Затем на сервере повторите `check_data.py`.
Папка `data/raw/` только для чтения: ничего в ней не меняйте и не удаляйте.

При первом запуске код создаёт кэш `data/cache/symbols_f64/` (~155 МБ, бинарные
копии CSV для быстрого чтения). Он не входит в Git, его можно удалить в любой момент.

## C. Окружение

```bash
python3 -V                                                          # нужен Python ≥ 3.10
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -c "import numpy, pandas, scipy, matplotlib; print('ok')"
```
Если чего-то не хватает, создайте виртуальное окружение в папке проекта (без
прав администратора, без `sudo`):
```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```
Скрипты `run.sh` сами используют `.venv/bin/python`, если он есть, иначе
`python3`. Другой интерпретатор можно задать так: `PYTHON=/путь/к/python bash ...`.
Если `pip install torch` тянет огромный CUDA-пакет, а GPU нет, можно поставить
CPU-версию: `.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`,
затем снова `pip install -e .`.

## D. Оценка мощности сервера (probe)

```bash
bash experiments/E010_lr_schedule_study/run.sh probe
```
Занимает 2–3 минуты. Probe смотрит доступные ядра (с учётом ограничений
контейнера), память, текущую нагрузку, GPU и диск, затем делает короткий
бенчмарк настоящего шага обучения для нескольких комбинаций (JOBS, THREADS).
Результат записывается в `resources.json` (в Git не входит) и печатается
по-русски, например:

    Найдено 32 доступных ядер ... свободно примерно 20.5 ядер. Память: доступно 50.1 ГБ ...
    Рекомендую JOBS=8, THREADS=1, DEVICE=cpu: примерно 450 шагов/с суммарно ...
    Вежливый режим: новая задача ждёт, если чужая нагрузка выше 24.

(числа выдуманы для примера)

- `JOBS` — сколько прогонов идёт одновременно (отдельные процессы);
- `THREADS` — потоков torch на процесс;
- `DEVICE` — `cpu` или `cuda`.

По умолчанию probe использует не больше половины свободных ядер (`SHARE=0.5`) и
не больше половины доступной памяти. Переопределения:
`SHARE=0.25`, `MAX_CORES=8`, `MAX_JOBS=4`, `MAX_MEM_GB=16`, например
`SHARE=0.25 bash experiments/E010_lr_schedule_study/run.sh probe`.
Если probe пишет «ВНИМАНИЕ: ресурсов мало», запускайте ночью или уменьшите
сетку (`SEEDS=17,29`).

`resources.json` общий для всех экспериментов: probe достаточно выполнить один
раз и повторять, если нагрузка на сервере заметно изменилась.

`PROBE_QUICK=1 bash ... run.sh probe` — быстрый режим (~30 с): только проверка,
что probe работает. Его рекомендация грубая; для настоящих запусков на сервере
выполняйте probe без `PROBE_QUICK`.

## E. Smoke (проверка, что всё работает)

Короткая проверка каждого эксперимента: ≤300 шагов, 2 train-файла, 1 seed,
вывод только в `experiments/<папка>/results_smoke/` (в Git не входит).
```bash
SMOKE=1 bash experiments/E010_lr_schedule_study/run.sh smoke
SMOKE=1 bash experiments/E011_long_runs_convergence/run.sh smoke
SMOKE=1 bash experiments/E012_depth_vs_window/run.sh smoke
SMOKE=1 bash experiments/E013_transformer_scaling/run.sh smoke      # самый долгий из smoke (трансформер)
SMOKE=1 bash experiments/E014_kerr_kan_hybrid/run.sh smoke
```
Признаки успеха:
- в конце каждой стадии строка `=== ... exit=0 ===` и `finished: N ok, 0 failed`;
- в `results_smoke/` появились `runs/*.json`, сводные CSV и `figures/*.png`;
- строки `control ... НЕ ПРОШЁЛ` в smoke **нормальны**: 300 шагов на 2 файлах
  дают отрицательное снижение BER (сеть только начала учиться), стадии в smoke
  не блокируются.

Удалить smoke-результаты: `rm -rf experiments/E01*/results_smoke`.

## F. План запусков

Перед каждой стадией посмотрите план; он ничего не считает:
```bash
bash experiments/E010_lr_schedule_study/run.sh plan all
bash experiments/E010_lr_schedule_study/run.sh plan stage1
```
`plan` печатает число прогонов (всего, готово, осталось), шаги, рекомендованные
`JOBS`/`THREADS`, **оценку времени** при рекомендованных и при половинных
настройках, долю времени на валидацию, память и диск. Конкретные часы здесь не
приводятся: мощность сервера заранее неизвестна, их печатает `plan` по
`resources.json`.

Откуда берётся оценка:

    время ≈ Σ_прогонов [ шаги / скорость_модели + замеры × окна_val+test / скорость_оценки ] / JOBS

Скорости измерены probe на эталонных моделях и пересчитаны по числу умножений
(время ∝ (умножения/эталон)^0.7). Это грубая оценка; первые завершённые
прогоны в логе печатают фактическое время и ETA.

| # | стадия | команда | прогонов | шагов на прогон | зависит от |
|---|---|---|---|---|---|
| 1 | E010 контроль | `bash experiments/E010_lr_schedule_study/run.sh baseline-check` | 9 | 20 000 | — |
| 2 | E010 stage1 | `... run.sh stage1` | 54 (45 новых) | 20 000 | контроль E010 прошёл |
| 3 | E010 select | `... run.sh select` | 0 | — | все 54 stage1 |
| 4 | E010 stage2 | `... run.sh stage2` | 72 (60 новых) | 20 000 | select |
| 5 | E010 stage3 | `... run.sh stage3` | 36 (24 новых) | 5 000–20 000 | stage1 + stage2 |
| 6 | E010 сводка | `... run.sh summarize` | 0 | — | — |
| 7 | E012 контроль | `bash experiments/E012_depth_vs_window/run.sh control` | 6 | 20 000 | — |
| 8 | E012 const | `... run.sh const` | 72 (66 новых) | 20 000 и 100 000 | контроль E012 |
| 9 | E012 best | `... run.sh best` | 72 | 20 000 и 100 000 | E010 select |
| 10 | E012 сводка | `... run.sh summarize` | 0 | — | — |
| 11 | E011 контроль | `bash experiments/E011_long_runs_convergence/run.sh control` | 4 | 20 000 | — |
| 12 | E011 const | `... run.sh const` | 24 | 200 000 (`MAX_STEPS`) | контроль E011 |
| 13 | E011 best | `... run.sh best` | 18 | 200 000 | E010 select |
| 14 | E011 сводка | `... run.sh summarize` | 0 | — | — |
| 15 | E013 контроль | `bash experiments/E013_transformer_scaling/run.sh control` | 2 | 20 000 | — |
| 16 | E013 stageA | `... run.sh stageA` | 24 | 20 000 | контроль E013 |
| 17 | E013 stageB | `... run.sh stageB` | 12 | 100 000 | контроль E013 |
| 18 | E013 сводка | `... run.sh summarize` | 0 | — | вердикт требует E011 |
| 19 | E014 контроль | `bash experiments/E014_kerr_kan_hybrid/run.sh control` | 2 | 20 000 | — |
| 20 | E014 main | `... run.sh main` | 30 | 40 000 | контроль E014 |
| 21 | E014 сводка | `... run.sh summarize` и `... run.sh residual` | 0 | — | — |

Объём шагов: E010 ≈ 2.5 млн, E012 ≈ 4.3 млн на ветку, E011 ≈ 8.4 млн, E013 ≈
1.7 млн (но трансформер на шаг в десятки раз дороже комплексной MLP), E014 ≈ 1.2 млн.

Память: около 1 ГБ на процесс (в основном сам torch), то есть примерно
`JOBS × 1 ГБ`. Диск: результаты и чекпоинты — десятки МБ на эксперимент, кэш
данных ~155 МБ.

Стадии `best` в E011/E012 без `experiments/E010_lr_schedule_study/results/stage1_selection.json`
просто пишут «ветка пропущена» и завершаются без ошибки; их можно запустить позже.

## G. Сколько запускать одновременно

- **Один эксперимент за раз** с рекомендованными `JOBS`/`THREADS`. Два
  эксперимента параллельно запускайте только если суммарное число процессов
  не превышает рекомендованного `JOBS` (например, два раза по `JOBS=4` при
  рекомендации 8).
- Перед запуском посмотрите загрузку:
  ```bash
  nproc          # сколько ядер
  uptime         # load average: три числа, загрузка за 1, 5, 15 минут
  top            # выход — клавиша q; htop, если установлен, нагляднее
  ```
  Если load average близок к `nproc`, сервер занят: повторите probe с
  `SHARE=0.25` или задайте `JOBS` вдвое меньше, лучше ночью.
- Вежливый режим встроен: перед каждым новым прогоном раннер смотрит
  `/proc/loadavg` и ждёт (проверка раз в 30 с, до 2 ч, переменная
  `POLITE_MAX_WAIT_S`), если чужая нагрузка выше порога из `resources.json`.
  Все процессы запускаются с `nice -n 10`.
- **Контроль первым.** Стадии `baseline-check`/`control` запускаются первыми.
  Если контроль не прошёл допуск, остальные стадии этого эксперимента
  отказываются запускаться (сообщение «контроль не прошёл допуск»).

Ориентиры (реальные числа выдаёт probe):
- **Малый сервер, 4 ядра, без GPU:** probe порекомендует 1–2 процесса по 1
  потоку. Каждый эксперимент займёт от суток и больше, E013 на CPU — несколько
  суток (один прогон трансформера на 100k шагов идёт больше суток). Имеет смысл
  `SEEDS=17,29`, ночные окна и отказ от веток `best`.
- **Средний, 16 ядер:** обычно 4–8 процессов по 1 потоку; время сокращается
  примерно пропорционально `JOBS`. Самый длинный прогон (200k шагов E011 или
  100k трансформера) задаёт нижнюю границу времени стадии, сколько бы ни было процессов.
- **Большой, 64 ядра и/или GPU:** 16–32 процесса на CPU; трансформер (E013)
  запускать на GPU (раздел L).

## H. Запуск в фоне (чтобы обрыв SSH не убил расчёт)

**tmux** — «виртуальный терминал», который продолжает работать, когда вы отключились.
```bash
tmux new -s e010                                         # открыть сессию с именем e010
bash experiments/E010_lr_schedule_study/run.sh baseline-check
```
- Отключиться, не останавливая: нажать `Ctrl+b`, отпустить, нажать `d`.
- Вернуться: `tmux attach -t e010`.
- Список сессий: `tmux ls`.
- Цепочку стадий можно запустить одной строкой (следующая начнётся, только если
  предыдущая завершилась успешно):
  ```bash
  bash experiments/E010_lr_schedule_study/run.sh baseline-check && \
  bash experiments/E010_lr_schedule_study/run.sh stage1 && \
  bash experiments/E010_lr_schedule_study/run.sh select && \
  bash experiments/E010_lr_schedule_study/run.sh stage2 && \
  bash experiments/E010_lr_schedule_study/run.sh stage3 && \
  bash experiments/E010_lr_schedule_study/run.sh summarize
  ```

Если tmux нет, используйте **nohup**:
```bash
nohup bash experiments/E010_lr_schedule_study/run.sh stage1 > e010_stage1.nohup.log 2>&1 &
echo $!          # номер процесса; запишите его
```
Если SSH оборвался: при tmux просто подключитесь снова и выполните
`tmux attach -t e010`; при nohup процесс продолжает работать, смотрите лог
(раздел I). Если процесс всё же остановился, запустите ту же команду ещё раз:
готовые прогоны пропускаются, длинные продолжаются с чекпоинта.

## I. Мониторинг

```bash
tail -f experiments/E010_lr_schedule_study/logs/stage1.log           # живой лог; выход Ctrl+C (расчёт не останавливает)
ls experiments/E010_lr_schedule_study/results/runs/*.json | wc -l    # сколько прогонов готово
bash experiments/E010_lr_schedule_study/run.sh plan stage1           # всего / готово / осталось + оценка оставшегося времени
uptime                                                               # общая нагрузка
```
Каждая строка `[ok k/N] ...` в логе содержит итог прогона, его время и ETA
всей стадии. Ошибки прогона записываются рядом с результатами:
`results/runs/<run_id>.error.txt`; очередь при этом продолжается.

## J. Остановка и продолжение

- Остановить: в окне tmux нажать `Ctrl+C`. Либо из другого терминала:
  ```bash
  cat experiments/E010_lr_schedule_study/logs/stage1.pid    # PID запуска
  kill <PID>                                                 # подставьте число
  ```
  PID-файл удаляется автоматически при завершении. Если запуск упал
  «грязно» и файл остался, следующий запуск сам заметит, что процесса нет, и
  заменит файл.
- Продолжить: выполнить ту же команду ещё раз. Готовые прогоны
  (`results/runs/<run_id>.json`) пропускаются, прогоны длиннее 20 000 шагов
  продолжаются с последнего чекпоинта (каждые 10 000 шагов, `checkpoints/E0xx/`).
- Пересчитать всё заново: `FORCE=1 bash ... run.sh stage1` (старые JSON будут
  перезаписаны — делайте это только осознанно).

## K. Если что-то пошло не так

| симптом | что делать |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` (или scipy) | раздел C: создать `.venv` и `pip install -e .` |
| `данные split не найдены` / `check_data.py` пишет «Отсутствуют» | раздел B: скопировать файлы по списку |
| `Killed`, `MemoryError`, сервер начал тормозить | уменьшить `JOBS` (например, `JOBS=2 bash ...`) или повторить probe с `SHARE=0.25` |
| `No space left on device` | `df -h .`; удалить `experiments/E01*/results_smoke`, `data/cache` (создастся заново); чекпоинты в `checkpoints/E0xx/*_last.pt` удалять только если прогон не нужно продолжать |
| в JSON `"diverged": true` | это **не ошибка** запуска: прогон разошёлся (NaN), он учитывается в сводке по правилу |
| `КОНТРОЛЬ НЕ ПРОШЁЛ ДОПУСК` | не продолжать. Прислать разработчику `results/control_check.json`, `logs/*.log`, `results/runs/control*.json` (или все `runs/*.json` контрольной стадии), `resources.json` и вывод `git log -1` |
| `стадия ... уже запущена (PID ...)` | этот этап уже идёт; дождитесь его или остановите (раздел J) |
| `stage1_selection.json не найден` | сначала `run.sh select` (E010) |

## L. Если на сервере есть GPU

- `probe` сам увидит CUDA (`torch.cuda.is_available()`), замерит 1, 2 и 4
  процесса на одной карте и порекомендует `DEVICE=cuda`, `THREADS=1`.
- Вручную: `DEVICE=cuda JOBS=2 bash experiments/E013_transformer_scaling/run.sh stageA`.
- При нескольких картах процессы распределяются по картам по кругу.
- Больше всего GPU даёт трансформеру (E013); маленьким комплексным MLP часто
  хватает CPU, и их выгоднее держать на CPU, чтобы не занимать карту.
- Результаты на GPU не совпадают побитово с CPU (другой порядок операций);
  контрольный допуск ±0.7 п.п. при этом должен выполняться. `run.sh`
  выставляет `CUBLAS_WORKSPACE_CONFIG=:4096:8` для детерминированного режима.
- Путь CUDA локально не проверялся: на машине разработки нет GPU. Если что-то
  не работает, запускайте на CPU (`DEVICE=cpu`) и пришлите лог.

---

## Как вернуть результаты

Нужны сводные CSV, JSON выбора/контроля, рисунки и логи. Файлы
`results/runs/*.json` можно не забирать (они восстанавливают сводку, но их много),
`results/curves/*.csv` маленькие и полезны для перепроверки.

| эксперимент | что забрать (внутри `experiments/<папка>/`) |
|---|---|
| E010 | `results/lr_sweep.csv`, `results/lr_sweep_summary.csv`, `results/stage1_selection.json`, `results/stage3_choice.json`, `results/control_check.json`, `results/curves/`, `figures/E010_*.png`, `logs/*.log` |
| E011 | `results/long_runs.csv`, `results/long_runs_curves.csv`, `results/long_runs_summary.csv`, `results/control_check.json`, `figures/E011_long_runs.png`, `logs/*.log` |
| E012 | `results/depth_window.csv`, `results/depth_window_summary.csv`, `results/depth_window_verdicts.csv`, `results/merge_check.csv`, `results/control_check.json`, `figures/E012_depth_window.png`, `logs/*.log` |
| E013 | `results/transformer_runs.csv`, `results/comparison_points.csv`, `results/transformer_verdicts.csv`, `results/control_check.json`, `figures/E013_transformer_vs_macs.png`, `logs/*.log` |
| E014 | `results/kan_hybrid.csv`, `results/kan_hybrid_summary.csv`, `results/kan_hybrid_verdicts.csv`, `results/residual_by_point.csv`, `results/residual_by_amplitude.csv`, `results/control_check.json`, `figures/E014_*.png`, `logs/*.log` |

**Способ 1: git с сервера** (если на сервере настроен доступ к GitHub).
Добавлять только явные пути, никогда `git add -A` или `git add .`:
```bash
E=experiments/E010_lr_schedule_study
git add $E/results/lr_sweep.csv $E/results/lr_sweep_summary.csv $E/results/stage1_selection.json \
        $E/results/stage3_choice.json $E/results/control_check.json $E/results/curves $E/figures/E010_*.png $E/logs/*.log
git status                       # проверьте, что в списке нет data/, checkpoints/, resources.json
git diff --cached --stat
git commit -m "E010: результаты сервера"
git push
```

**Способ 2: rsync/scp в WSL** (выполнять в WSL, из корня репозитория;
`user@server` и путь подставьте свои):
```bash
rsync -av --exclude 'runs/' user@server:~/constellation-denoiser/experiments/E010_lr_schedule_study/results/ experiments/E010_lr_schedule_study/results/
rsync -av user@server:~/constellation-denoiser/experiments/E010_lr_schedule_study/figures/ experiments/E010_lr_schedule_study/figures/
rsync -av user@server:~/constellation-denoiser/experiments/E010_lr_schedule_study/logs/ experiments/E010_lr_schedule_study/logs/
```
Аналогично для остальных папок. Вариант без rsync:
`scp -r user@server:~/constellation-denoiser/experiments/E010_lr_schedule_study/figures experiments/E010_lr_schedule_study/`.

После получения результатов: допишите запись в конец `LOG.md` (числа и ссылки
на файлы); существующие записи не редактируйте.
