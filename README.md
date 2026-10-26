# Constellation denoiser

Нейросетевые и физически мотивированные эквалайзеры 16-QAM для волоконной линии
(20 × 100 км SSMF, 32 Гбод): комплексные MLP с активацией Керра, трансформер, KAN,
модели цифрового обратного распространения. Эксперименты — `experiments/E0NN_*/`
(`config.json` — параметры и правило до запуска, `README.md` — вопрос и команда),
журнал — `LOG.md` (только дописывается), правила проекта и текущее состояние — `README.md`.

Установка (Python ≥ 3.10, без прав администратора):

    python3 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"

Данные: `data/raw/archive/*.csv` (в Git не входят, только чтение); проверка —
`PYTHONPATH=src python scripts/check_data.py`; нужные файлы перечислены в
`experiments/E010_E014_required_files.txt` (копирование: `rsync --files-from=<список> ./ user@server:<папка>/`).

Запуск эксперимента: `bash experiments/<E0NN_папка>/run.sh <стадия>`; локальная проверка —
`SMOKE=1 bash experiments/<E0NN_папка>/run.sh smoke`; мощность и время на машине — стадии `probe`, `estimate`.

Сервер: `bash scripts/remote.sh go|estimate|status|fetch|stop …` (конфиг `scripts/remote.env`, см. `README.md`).
Презентация из графиков `paper/figures_manifest.csv`: `make deck` → `slides/deck_<дата>.pptx`.
