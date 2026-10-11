# Constellation denoiser

Исследовательский проект по оценке BER и компенсации остаточных искажений
16-QAM. Исторические эксперименты и артефакты разложены по E-номерам.

## Карта проекта

- `data/raw/` — исходные CSV, только чтение. Архив PR1–PR2000 находится в
  `data/raw/archive/`; исходные PR1–PR10 остаются непосредственно в `data/raw/`.
- `data/manifest/` — pre-migration SHA-256, список уникальных файлов,
  split-конфиги и `data_audit*.csv`.
- `src/constellation_denoiser/` — код экспериментов и централизованные пути
  `paths.py`.
- `experiments/E00N_*/` — протокол, конфигурация, команда, результаты,
  существующие графики и логи одного эксперимента.
- `checkpoints/E00N/` — локальные веса; не добавляются в Git.
- `paper/` — порядок графиков в `figures_manifest.csv` и презентация
  `results.pptx`.
- `notebooks/archive/` — исторические блокноты 01–06.
- `references/` — исходные описания и сохранённая legacy-сводка.
- `LOG.md` — append-only хронология; `FINDINGS.md` — численные факты со ссылками.

Нумерация следует хронологии, установленной по именам, конфигам, журналам и
датам файлов. Дата изменения не считается датой запуска; неизвестные факты
помечены `unknown`.

## Настройка и запуск

Из корня репозитория:

```bash
python -m pip install -e .
```

Перед запуском читать README конкретного эксперимента и его `config.json`.
Каталоги E001–E009 содержат исторические выходы, а их `run.sh` по умолчанию
защищают эти данные от перезаписи. Для нового запуска сначала создать новую
E-папку и отдельные пути вывода, затем записать до запуска вопрос и правило
интерпретации.

Чтобы посмотреть параметры команды без обучения:

```bash
bash experiments/E003_large_scale_pilot/run.sh --help
PYTHONPATH=src python -m constellation_denoiser.large_scale_pilot --help
```

Старый запуск сравнения моделей:

```bash
PYTHONPATH=src python -m constellation_denoiser.experiment --seeds 17 29 43
PYTHONPATH=src python -m constellation_denoiser.baselines
```

Эти исторические команды не запускать поверх существующих результатов.
`data/raw/` не редактировать. Перед staging проверять `git status` и
`git diff --cached --name-only`: датасеты и checkpoint не должны попасть в Git.
