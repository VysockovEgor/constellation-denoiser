# Constellation denoiser

Нейросетевые и физически мотивированные эквалайзеры 16-QAM для волоконной линии
(20 × 100 км SSMF, 32 Гбод, один канал): комплексные MLP с активацией Керра (схема
Богданова–Сидельникова, 2021), трансформер, KAN, модели цифрового обратного распространения.
Эксперименты — `experiments/E0NN_*/` (`config.json` — параметры и правило до запуска,
`README.md` — вопрос и команда), журнал запусков — `LOG.md` (только дописывается),
сайт с результатами — `make journal` → `journal/_site/index.html`.

## Установка и запуск

    python3 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"

- Данные: `data/raw/archive/*.csv` (в Git не входят, только чтение); проверка —
  `PYTHONPATH=src python scripts/check_data.py`; список нужных файлов —
  `experiments/E010_E014_required_files.txt` (`rsync --files-from=<список> ./ user@server:<папка>/`).
- Эксперимент: `bash experiments/<E0NN_папка>/run.sh <стадия>`; локальная проверка —
  `SMOKE=1 bash experiments/<E0NN_папка>/run.sh smoke` (≤300 шагов, 2 файла, 1 seed);
  мощность машины и время — стадии `probe`, `estimate`.
- Сервер: `bash scripts/remote.sh go part1|part2|all|<E0NN> <стадия>`, `sync`, `estimate`, `status`,
  `fetch [--to <папка>] [--journal]`, `stop <сессия>`; конфиг `scripts/remote.env`
  (образец `scripts/remote.env.example`), проверка без сервера — `--dry-run`.
- Журнал (Quarto в `~/opt`): `make journal`, `make journal-preview`, `make journal-pptx`,
  `make journal-docx`, `make journal-pdf`; страница нового эксперимента —
  `python scripts/journal_new.py E0NN`. Числа на страницах только из `journal/_auto/`
  (генерирует `scripts/journal_build_auto.py`), раздел «Выводы» пишется руками и только дописывается.

## Данные и метрики

- Строка CSV: `tx_I, tx_Q, rx_I, rx_Q`, 16 384 символа на файл, 1 отсчёт на символ, одна
  поляризация, rx после восстановления фазы; уровни ±1/√10, ±3/√10. Split —
  `experiments/E004_training_file_curve/config.json`: train-пул 210 (N=30 — первые 30),
  validation 45, test 45.
- Снижение BER `(raw − model)/raw · 100 %`; `Q² = 10·log10(10·erfcinv(8·BER/3)²)`; основная
  метрика — тест в точке лучшей валидации. Raw test BER = 0.009551859551859552 (Q² 13.97 дБ).
- Умножения на символ: комплексное = 4, Керр = 7 на нейрон; два слоя с Керром `4h(W+h+1)+14h`
  (W=31, h=48: 16 032), один слой или после слияния `h(4W+11)` (6 480); физика — `model_zoo.macs_phys`.
- Контроль (±0.7 п.п., seeds 17 и 29, 20k шагов, const 1e-3, окно 31, h 48, N=30):
  `complex_both` 61.727, `complex_a` 62.191, `one_hidden` 59.437, cosine `both` 58.763.

## Правила

- Новый эксперимент — новая папка `experiments/E0NN_<имя>/` с номером по порядку; вопрос и правило
  интерпретации записываются в `config.json` (`rule_draft`) до запуска и потом не меняются;
  неизвестное — `unknown`.
- Выбор (чекпоинт, конфигурация) только по валидации, тест — для отчёта. Seeds 17, 29, 43, 57, 71.
  Контроль в каждом эксперименте; вне допуска остальные стадии не запускаются.
- Старые результаты и чекпоинты не перезаписываются; новый запуск — новый E-номер.
- Вывод — только с числом и ссылкой на файл; указывать ограничения, число seeds, файлов и шагов.
  Разброс seeds — размах (max − min). Графики `E0NN_<название>.png`, порядок —
  `paper/figures_manifest.csv`.
- Git: коммиты по явным путям; не коммитить `data/`, `checkpoints/`, `*.pt`, `results_smoke/`,
  `resources.json`, `scripts/remote.env`, `slides/*.pptx`, файлы больше 5 МБ.
- Сервер общий: без `sudo`, без установок вне `~` и репозитория, не трогать чужие процессы,
  сессии и файлы; вежливый режим раннера (nice, `/proc/loadavg`) не отключать.
