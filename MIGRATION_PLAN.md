# План реорганизации constellation-denoiser

> Этап 1. Это план, а не миграция: файлы не перемещались, CSV/JSON и изображения не
> изменялись, обучение и построение графиков не запускались. Не начинать этап 2 до
> отдельной команды владельца «ВЫПОЛНЯЙ».

## 0. Исходное состояние и критерий осторожности

- Git-репозиторий существует: ветка `main`, последний коммит `c391a78` (`Calculate
  BER`). Это не чистый снимок проекта: кроме удалённого отслеживаемого
  `qam16_demod_ber.ipynb`, текущие `.gitignore`, README и каталоги проекта не
  отслеживаются Git.
- `.gitignore` не игнорирует ни исходные CSV, ни `.pt`, ни чекпоинты: проверка
  `git check-ignore` для файлов из `data/`, `data/raw/`, `models/` и
  `reports/results/` вернула `NOT IGNORED`.
- Инвентаризация: 2 000 CSV в корне `data/`, 10 CSV в `data/raw/`, 85 файлов
  `.pt`, 38 файлов в `reports/results/`, 7 PNG в `reports/figures/`, 18 файлов
  журналов (не считая `ablation_launch.pid`), 6 пронумерованных блокнотов.
- Даты изменения файлов показывают общую последовательность: исходное сравнение
  и конфиги — 28 сентября; проверки tanh/paper-like — 29 сентября; масштабный
  пилот — 3–4 октября; кривые/свипы/плато — 4 октября; абляция — 11 октября
  2026. Даты mtime — вспомогательное свидетельство, не доказательство времени
  самого запуска.
- При создании pre-migration снимка не добавлять `.venv`, исходные данные или
  чекпоинты. Текущую удалённую запись `qam16_demod_ber.ipynb` не восстанавливать;
  включение её удаления в снимок следует сверить непосредственно перед коммитом.

## 1. Таблица старых и планируемых путей

### Данные и чекпоинты

| Старый путь / маска | Планируемый новый путь | Обоснование и оговорки |
|---|---|---|
| `data/Symbols_1m_1ch_PR_[1-2000].csv` | `data/raw/archive/Symbols_1m_1ch_PR_[1-2000].csv` | Архивная партия PR1–PR2000; использовалась аудитом и масштабными опытами E003+. Подкаталог `archive/` нужен, чтобы не перезаписать уже имеющиеся `data/raw/Symbols_1m_1ch_PR_1.csv`–`PR_10.csv`. Байты, имена и число файлов сохранить. |
| `data/raw/Symbols_1m_1ch_PR_[1-10].csv` | `data/raw/Symbols_1m_1ch_PR_[1-10].csv` (оставить на месте) | Исходная десятка для E001–E002; `data/raw` — только чтение. Те же байты также встречаются в архиве, но не удалять ни одну копию без отдельного одобрения. |
| `models/{mlp,1dcnn,lstm,transformer}.pt` | `checkpoints/E001_pilot_models/{те же имена}` | Первичный пилот моделей, имена совпадают с архитектурами E001. |
| `models/complex_ffe.npz` | `checkpoints/E001_pilot_models/complex_ffe.npz` | Линейный комплексный FFE описан в `models/README.md` как обученный на PR1–PR7. Формат NPZ не менять. |
| `models/tanh_mlp_selected.pt` | `checkpoints/E002_tanh_complex_paper/` | По имени и дате 29 сентября связан с проверкой tanh-MLP; финальную принадлежность сверить по checkpoint metadata. |
| `models/paper_{real,complex}_mlp.pt` | `checkpoints/E002_tanh_complex_paper/` | По именам относятся к paper-like вещественной и комплексной MLP E002. |
| `models/large_{real_tanh,complex}_seed_{17,29,43}.pt` | `checkpoints/E003_large_scale_pilot/` | Масштабный пилот; тип модели и seed есть в имени файла. |
| `models/step_curve/{tanh,complex}_N{7,30,105,210}_{best,last}.pt` | `checkpoints/E005_equal_steps/` | Имена согласуются с кривой числа файлов. В конфигурации отдельных запусков seed не закодирован: не добавлять его по догадке. |
| `models/step_curve/complex_N7_seed{29,43}_{best,last}.pt` | `checkpoints/E006_plateau_seeds/` | Суффиксы seed связывают их с E006; фактическая длительность и полнота по seed сверяются по CSV и логам. |
| `models/step_curve/complex_N7_{best,last}.pt` | `checkpoints/E005_equal_steps/` **?** | Нет явного seed; возможно, это модель шага 20 000 или базовый seed E006. До проверки metadata/log не присваивать безусловно. |
| `models/complex_mlp_sweep/complex_w{31,65,127}_h{32,48}_N30_seed{17,29}_{best,last}.pt` | `checkpoints/E007_window_width_sweep/` | Маска кодирует окно, ширину, число эпох/итераций в метке N и seed; соответствует sweep E007. |
| `models/complex_mlp_lr_sweep/complex_cosine_w{31,65}_h{32,48}_N30_seed{17,29}_{best,last}.pt` | `checkpoints/E008_cosine_lr_coefficients/` | Маска и каталог соответствуют косинусному lr sweep E008. |
| `models/complex_mlp_ablation/ablation_{a,b,c,d}_w31_h48_N30_seed{17,29}_{best,last}.pt` | `checkpoints/E009_kerr_ablation/` | Маска и каталог соответствуют вариантам абляции E009. |
| `models/README.md` | `references/legacy_models_README.md` | Сохранить как исходное описание чекпоинтов и форматов; не удалять/не переписывать фактические утверждения в ходе переноса. |
| `models/` и все файлы `*.pt` | Игнорировать Git | В репозитории хранить только код/документацию; чекпоинты остаются локальными. Добавить явные правила игнорирования до создания снимка. |

**Блокер структуры данных:** целевой перечень называет только `data/raw/`, но
там уже есть имена PR1–PR10, совпадающие с десятью файлами из `data/`. План
предлагает `data/raw/archive/` для 2 000 архивных файлов и сохраняет текущую
десятку в `data/raw/`. Это сохраняет число файлов и устраняет коллизии. Если
владелец не принимает дополнительный подкаталог, до этапа 2 требуется выбрать
другое одобренное место; перезапись или дедупликация не допускается.

### Результаты, графики, журналы, скрипты и прочие файлы

| Старый путь / маска | Планируемый новый путь | E и свидетельство |
|---|---|---|
| `reports/results/experiment_config.json`, `model_comparison.csv`, `model_comparison_summary.csv`, `linear_ffe_baseline.csv`, `linear_ffe_per_run.csv`, `inference_benchmark.csv` | `experiments/E001_pilot_models/{config.json,results/...}` | E001: конфиг PR1–PR10, модели окна 7; таблицы названы по моделям/FFE/benchmark. |
| `reports/figures/model_tradeoff.png` | `experiments/E001_pilot_models/figures/E001_model_tradeoff.png` и ссылка в `paper/figures_manifest.csv` | E001 по содержимому сравнения; подписи задержек требуют отдельной сверки с `inference_benchmark.csv`, ничего не исправлять при переносе. Источник CSV конкретного изображения пока `?`. |
| `reports/results/tanh_mlp_sweep_config.json`, `tanh_mlp_sweep.csv`, `tanh_mlp_sweep_summary.csv`, `tanh_mlp_selected_test.csv`, `paper_real_mlp.csv`, `paper_complex_mlp.csv` | `experiments/E002_tanh_complex_paper/results/` и `config.json` | E002: конфиг sweep прямо задаёт tanh, окна/ширины и PR9–PR10; paper-like CSV названы по двум моделям и описаны в legacy summary. |
| `reports/results/data_audit.csv`, `data_audit_all.csv`, `large_scale_pilot_config.json`, `large_scale_pilot_summary.csv`, `large_scale_pilot_per_file.csv`, `large_scale_pilot_bootstrap.json` | `experiments/E003_large_scale_pilot/{config.json,results/}`; неизменяемые контрольные хэши — `data/manifest/` | E003: аудит и имена pilot; конфиг указывает архив, 1 356 уникальных файлов и split seed. |
| `reports/figures/large_scale_pilot.png` | `experiments/E003_large_scale_pilot/figures/E003_large_scale_pilot.png` | E003, соответствующий CSV — `large_scale_pilot_summary.csv`. |
| `reports/results/learning_curve_config.json`, `learning_curve_runs.csv`, `learning_curve_seed17.csv`, `learning_curve_per_file.csv` | `experiments/E004_training_file_curve/{config.json,results/}` | E004: конфиг задаёт размеры train 7/30/60/120/210 и seed порядка подвыборки; summary фиксирует общий test из 45 файлов. Этот конфиг также используется последующими скриптами, поэтому оригинал оставить как provenance/общую основу и не трактовать как уникальную настройку каждого E. |
| `reports/results/learning_curve_summary.md` | `experiments/E004_training_file_curve/README.md` (факты перенести со ссылками) и `references/legacy_learning_curve_summary.md` (неизменённая копия) | Прямое имя и содержимое указывают на E004. Выводы и ограничения не расширять. |
| `reports/figures/learning_curve_seed17.png` | `experiments/E004_training_file_curve/figures/E004_learning_curve_seed17.png` | E004; исходные данные графика — `learning_curve_seed17.csv`. |
| `reports/results/fixed_step_learning_curve.csv` | `experiments/E005_equal_steps/results/fixed_step_learning_curve.csv` | E005 по имени и значениям N; CSV содержит строку шага 20 000 для каждого из N=7/30/105/210 и обеих моделей. |
| `reports/figures/fixed_step_learning_curve.png` | `experiments/E005_equal_steps/figures/E005_fixed_step_learning_curve.png` | E005; source CSV — `fixed_step_learning_curve.csv`. |
| `reports/logs/fixed_step_complex.log`, `fixed_step_tanh.log` | `experiments/E005_equal_steps/logs/` | E005 по меткам моделей и `fixed_step`; точный seed/фактическое завершение проверять в содержимом логов. |
| `reports/results/plateau_seed_study.csv` | `experiments/E006_plateau_seeds/results/plateau_seed_study.csv` | E006 по имени и колонкам `N,seed,step`; содержит точки до 100 000 для некоторых сочетаний. Покрытие seed неодинаковое — см. раздел 2. |
| `reports/figures/plateau_seed_study.png` | `experiments/E006_plateau_seeds/figures/E006_plateau_seed_study.png` | E006; source CSV — `plateau_seed_study.csv`. Палитру не менять. |
| `reports/logs/plateau_a.log`, `plateau_b.log` | `experiments/E006_plateau_seeds/logs/` | E006 по времени (4 октября) и соседству с plateau CSV/чекпоинтами; содержание до назначения проверить. |
| `reports/results/complex_mlp_sweep.csv`, `complex_mlp_sweep_summary.csv` | `experiments/E007_window_width_sweep/results/` | E007 по sweep именам и чекпоинтам `complex_mlp_sweep/`; summary строки содержат окно/ширину/число параметров и test BER. |
| `reports/figures/complex_mlp_sweep.png` | `experiments/E007_window_width_sweep/figures/E007_complex_mlp_sweep.png` | E007; source CSV — `complex_mlp_sweep_summary.csv`. |
| `reports/logs/complex_mlp_sweep_seed{17,29}.log`, `scripts/run_complex_mlp_sweep.sh` | `experiments/E007_window_width_sweep/{logs/,scripts/}`; создать локальный `run.sh` | E007 по совпадающим именам sweep и seed. Существующий shell-скрипт сохранить без редактирования содержимого, обновить пути отдельно. |
| `reports/results/complex_mlp_lr_sweep.csv`, `complex_mlp_lr_diagnostics.csv`, `complex_mlp_lr_summary.csv`, `complex_mlp_lr_diagnostics_summary.csv` | `experiments/E008_cosine_lr_coefficients/results/` | E008 по `lr_sweep`, диагностике c и именам checkpoint `complex_cosine_*`. |
| `reports/figures/complex_mlp_lr_c_vs_steps.png` | `experiments/E008_cosine_lr_coefficients/figures/E008_complex_mlp_lr_c_vs_steps.png` | E008 по имени. Изображение содержит только график c — не перерисовывать и не дополнять на этом этапе. Source CSV уточнить между diagnostics CSV и его summary: `?`. |
| `reports/logs/complex_mlp_lr_seed{17,29}.log`, `scripts/run_complex_mlp_lr_sweep.sh` | `experiments/E008_cosine_lr_coefficients/{logs/,scripts/}`; создать `run.sh` | E008 по имени sweep, seed и cosine-checkpoints. |
| `reports/results/complex_mlp_ablation.csv`, `complex_mlp_ablation_diagnostics.csv`, `complex_mlp_ablation_merge_check.csv`, `complex_mlp_ablation_summary.csv` | `experiments/E009_kerr_ablation/results/` | E009 по `ablation`, вариантам a–d и merge-check. |
| `reports/results/complex_mlp_ablation_tradeoff.png` | `experiments/E009_kerr_ablation/figures/E009_complex_mlp_ablation_tradeoff.png` | PNG лежит в `results/`, но по типу и имени является графиком E009; source CSV уточнить при чтении метаданных: `?`. Не перерисовывать. |
| `reports/logs/complex_mlp_ablation_{a,b,c,d}_seed{17,29}.log`, `ablation_launch.log`, `ablation_nohup.out` | `experiments/E009_kerr_ablation/logs/` | Первые восемь логов однозначно по именам seed/variant. Два общих лога также относятся к ablation по содержимому/имени, но этап их запуска определить из текста. |
| `reports/logs/complex_mlp_ablation_launch.log`, `complex_mlp_ablation_summarize.log`, если найдены внутри каталогов/при чтении скриптов | `experiments/E009_kerr_ablation/logs/` | Скрипты упоминают эти пути, однако таких отдельных файлов в инвентаризации верхнего уровня `reports/logs/` нет. Не создавать фиктивные логи. |
| `reports/logs/ablation_launch.pid` | `experiments/E009_kerr_ablation/logs/ablation_launch.pid` или оставить в provenance `?` | PID-файл не считать журналом эксперимента; не использовать его для остановки/запуска процессов. Проверить наличие активного процесса только в рамках безопасной процедуры перед переносом. |
| `reports/figures/results.pptx` | `paper/results.pptx` | Презентацию переместить как есть; не переставлять слайды/графики на этом этапе. Порядок будущих графиков задаёт CSV manifest, не имя файла. |
| `reports/results/summary.md` | `references/legacy_summary.md` (неизменённый provenance); ссылки на соответствующие E в `FINDINGS.md` | Документ совмещает E001–E004 и не может без потери контекста быть приписан одному E. Не разрезать и не переписывать выводы автоматически. |
| `reports/results/` прочие файлы, не попавшие в строки выше | Подходящая `experiments/E00N_*/results/` только после сверки имени/содержимого; иначе `references/unassigned/` как временный плановый путь | Любой конкретный неизвестный файл должен быть перечислен отдельно в фактическом отчёте этапа 2, а не оставлен «сиротой». |
| `notebooks/01_baseline.ipynb`–`06_model_comparison.ipynb` | `notebooks/archive/` с исходными именами | В наборе ровно шесть нумерованных блокнотов. Переносить без выполнения/изменения ячеек; `notebooks/README.md` сохранить рядом как `notebooks/archive/README.md` или согласовать отдельное место. |
| `references/**` | `references/**` | Уже соответствует целевому каталогу; оставить на месте. |
| `src/constellation_denoiser/**` | `src/constellation_denoiser/**` | Структуру кода не переносить; менять только пути, необходимые после миграции. |

### Будущий manifest рисунков

Предварительный хронологический порядок (номера непрерывные; E002 не имеет PNG,
поэтому отдельной строки для него нет):

| fig_no | experiment_id | file | caption (черновик) | source_csv |
|---:|---|---|---|---|
| 1 | E001 | `E001_model_tradeoff.png` | Сравнение качества/стоимости моделей; перепроверить подписи задержек | `?` (сверить `inference_benchmark.csv` и генератор) |
| 2 | E003 | `E003_large_scale_pilot.png` | BER на фиксированном масштабном split | `large_scale_pilot_summary.csv` |
| 3 | E004 | `E004_learning_curve_seed17.png` | BER по числу обучающих файлов, seed 17 | `learning_curve_seed17.csv` |
| 4 | E005 | `E005_fixed_step_learning_curve.png` | Результат по числу файлов при фиксированном числе шагов | `fixed_step_learning_curve.csv` |
| 5 | E006 | `E006_plateau_seed_study.png` | Плато/seed study; палитра неоднозначна | `plateau_seed_study.csv` |
| 6 | E007 | `E007_complex_mlp_sweep.png` | Sweep окна/ширины комплексной MLP | `complex_mlp_sweep_summary.csv` |
| 7 | E008 | `E008_complex_mlp_lr_c_vs_steps.png` | Диагностика обучаемого c по шагам; график содержит только c | `?` (проверить diagnostics CSV) |
| 8 | E009 | `E009_complex_mlp_ablation_tradeoff.png` | Trade-off абляции и слияния слоёв | `?` (проверить генератор) |

## 2. Привязка к E-номерам, даты и спорные случаи

| ID | Короткое имя и фактические параметры | Короткое обоснование |
|---|---|---|
| E001 | Пилот MLP/CNN/Bi-LSTM/Transformer/линейный FFE, окно 7, PR1–10; seeds 17/29/43 | `experiment_config.json`, `models/README.md`, `model_comparison*`, `linear_ffe*`, `inference_benchmark.csv`; модельные веса и результаты датированы 28 сентября. |
| E002 | tanh-MLP sweep и paper-like вещественная/комплексная MLP на PR9–PR10 | `tanh_mlp_sweep_config.json` задаёт окна/ширины, seeds и split; `paper_*` CSV/PT и даты 29 сентября. Для paper-like части отдельного полного config не найдено. |
| E003 | Аудит и большой пилот: 1 356 уникальных, выборка 300, 210/45/45 | `large_scale_pilot_config.json`, audit/pilot CSV+JSON, named checkpoints и legacy summary; файлы 3–4 октября. |
| E004 | Кривая по размерам train-набора на фиксированных 45 test-файлах | `learning_curve_config.json`, `learning_curve_*`, summary и график; mtime 4 октября. |
| E005 | Контроль равного числа шагов: N=7/30/105/210, 20 000 шагов | `fixed_step_learning_curve.csv`, `fixed_step_*.log`, `step_curve` checkpoints. CSV действительно содержит строки 20 000 для всех N и обеих моделей. |
| E006 | Плато/разброс по seed, предполагаемый максимум 100 000, seeds 17/29/43 | `plateau_seed_study.csv`, `plateau_a/b.log`, `step_curve` checkpoints, 4 октября. **Неоднозначность:** CSV имеет 102 строки данных. В нём N=7 и N=210 для seed 17 доходят до 100 000; N=7 для seed 29/43 доходит только до 20 000. Не заявлять полные 100 000 шагов для всех seed/N без подтверждения логами. |
| E007 | Sweep окна/ширины complex MLP; окна 31/65/127, ширины 32/48, seeds 17/29 по filenames | `complex_mlp_sweep*`, график, логи и checkpoints; даты 4 октября. |
| E008 | Cosine lr и обучаемые коэффициенты c | `complex_mlp_lr_*`, cosine checkpoints, скрипт/logs и график `c_vs_steps`; даты 4 октября. График — только c, не полноценное сравнение метрик. |
| E009 | Абляция Kerr-активаций и проверка слияния линейных слоёв | `complex_mlp_ablation*`, варианты a–d, checkpoints/logs и merge-check; файлы 11 октября. Параллельные скрипты и текущий статус PID требуют аккуратной проверки. |

### Нераспределённое или требующее проверки

1. `reports/results/summary.md` — составной документ E001–E004; сохранить как
   provenance и разделять утверждения только при наличии явных ссылок на источники.
2. `models/step_curve/complex_N7_best.pt` и `complex_N7_last.pt` — seed не указан;
   проверить metadata/log до окончательного E005/E006.
3. `reports/figures/model_tradeoff.png` — принадлежность E001 вероятна по
   содержимому/дате, но конкретный source CSV и подписи сверить с кодом построения
   и `inference_benchmark.csv`.
4. Источник графиков `complex_mlp_lr_c_vs_steps.png` и
   `complex_mlp_ablation_tradeoff.png` определить по исходному коду; не угадывать.
5. `ablation_launch.pid` и `ablation_nohup.out`: pid-файл не подтверждает, что
   процесс завершился; не останавливать неизвестный процесс и не менять состояние
   обучения.
6. Разница между файлами `launch_ablation_max.sh` и
   `run_complex_mlp_ablation_max_parallel.sh` (оба launcher для E009) требует
   сохранить оба скрипта и проверить содержимое, прежде чем выбирать канонический
   `run.sh`.
7. Отдельные готовые логи для E004 не обнаружены; не создавать их задним числом.

## 3. Жёстко заданные пути и предлагаемый `paths.py`

Поиск выражений `reports/`, `models/`, `data/`, `results/` в `src/` и `scripts/`
обнаружил такие места (контекст до переноса: все пути относительные к
`PROJECT_ROOT`, кроме абсолютных путей в JSON/Markdown):

| Файл | Найденные ссылки/действие на этапе 2 |
|---|---|
| `src/constellation_denoiser/experiment.py` | Ожидает PR1–PR10 в `data/raw/`; сохранить это поведение и путь. |
| `src/constellation_denoiser/large_scale_pilot.py` | `reports/results/data_audit_all.csv`; использовать `DATA_MANIFEST`/результаты E003. |
| `src/constellation_denoiser/learning_curve.py` | `reports/results/data_audit_all.csv`, `large_scale_pilot_config.json`, каталог `reports/results`; задать входы/выход через общие пути и config E004. |
| `src/constellation_denoiser/fixed_step_curve.py` | `models/step_curve`, `reports/results/fixed_step_learning_curve.csv`, shared configs/audit; заменить каталог чекпоинтов на `checkpoints/E005_equal_steps`, результат на каталог E005. |
| `src/constellation_denoiser/complex_mlp_sweep.py` | `models/complex_mlp_sweep`, `reports/results/complex_mlp_sweep.csv`, shared config/audit; E007. |
| `src/constellation_denoiser/complex_mlp_lr_sweep.py` | `models/complex_mlp_lr_sweep`, learning-curve config, `complex_mlp_lr_*` и audit; E008. |
| `src/constellation_denoiser/complex_mlp_ablation_train.py` | `models/complex_mlp_ablation`, config, ablation metrics/diagnostics, audit; E009. |
| `src/constellation_denoiser/summarize_complex_mlp_ablation.py` | Sweep/ablation CSV, каталоги `models/complex_mlp_ablation`, `reports/results`; E009. |
| `src/constellation_denoiser/summarize_learning_curve.py` | `reports/results`, `reports/figures/learning_curve_seed17.png`; E004, filename output должен иметь E004 prefix. |
| `scripts/run_complex_mlp_sweep.sh` | Создаёт/пишет `reports/logs/complex_mlp_sweep_seed*.log`; E007. |
| `scripts/run_complex_mlp_lr_sweep.sh` | Создаёт/пишет `reports/logs/complex_mlp_lr_seed*.log`; E008. |
| `scripts/launch_ablation_max.sh` и `scripts/run_complex_mlp_ablation_max_parallel.sh` | `reports/logs`, `models/complex_mlp_ablation` и E009 metrics; сохранить оба launcher, пути обновить. |
| `reports/results/large_scale_pilot_config.json` | Содержит абсолютные `dataset_root` и `audit_csv`; заменить при миграции на переносимый относительный путь/путь из `paths.py`, не менять остальные числа/ключи. |
| `reports/results/learning_curve_summary.md` | Абсолютная ссылка на старый PNG; при создании README E004 дать относительную ссылку на новый файл. |

В `src/constellation_denoiser/paths.py` предлагаются только `Path`-константы:
`PROJECT_ROOT`, `DATA_RAW`, `DATA_ARCHIVE`, `DATA_MANIFEST`, `EXPERIMENTS`,
`CHECKPOINTS`, `PAPER`, а также helpers/константы для `experiment_results(id)`,
`experiment_logs(id)` и `experiment_figures(id)`. Путь эксперимента выбирается
по E-id, а не по имени результата. Все скрипты и модули используют эти значения;
не вводить дополнительную систему поиска файлов и не перемещать код из `src/`.

## 4. SHA-256 и дубликаты CSV (проверка только чтением)

Сейчас в корне `data/` 2 000 CSV и в `data/raw/` 10 CSV. Хэширование SHA-256
показало 10 групп хэшей, представленных и в архиве, и в `raw/`; все десять
`raw/PR1–PR10` имеют содержательное совпадение с архивом. Есть только 2
один-к-одному совпадения: PR10 и PR5. Остальные совпавшие `raw` файлы имеют
несколько имен в архиве:

| SHA-256 | `data/raw/` | Файлы архива с тем же хэшем |
|---|---|---|
| `b26acb8c5e2faecfc6c492f2626bc6f66334fd40069640e9d884e8b00ecd0ab5` | PR1 | PR1, PR247, PR737, PR917 |
| `ec437bf02224ba60f63b6e4bb7ad488ad51d45275eccaad804a407c6e1917fb9` | PR2 | PR2, PR351 |
| `b0497e23d5237aae0572c26a1582ba969d7bc3358e82e5561bb0ec1ae9076ae4` | PR3 | PR3, PR249 |
| `485cd4c2007dbd6fe91b1d67422713b911b7b0dc7e9a01ac4e14337bd31fea14` | PR4 | PR4, PR250 |
| `9c02c597ebc1e8b88beee407fde601c37c37e3891a686a45864dead0291c10f1` | PR5 | PR5 |
| `0e87638b8ac9dee85a6f872ea51c594d1891571f5aaacffe5cb77e0972dc8a33` | PR6 | PR6, PR105, PR562 |
| `091129b6bee534f0b7f127a2512bf9102af6d750c2c6c2d6e369eaec627d3d1f` | PR7 | PR7, PR775 |
| `71edf7017c8c670e4d5deb47e90f40d3fd7e9f150acd2e2b6cd0d75824671b3b` | PR8 | PR8, PR485, PR763 |
| `0c435a8f099466111ded192677da10c81472a200afda3f2edc1caa8453a4fbb3` | PR9 | PR9, PR486 |
| `669eaf192a40af36d58c0c107229b72142906442c4359b5b8ae32ebc6d5fea58` | PR10 | PR10 |

Пересечение подтверждает точные совпадения, но не является разрешением удалять
копии или выбирать имя PR как уникальную реализацию. В `data/manifest/` на этапе
2 записать SHA-256 каждого исходного файла и полного дерева `results` **до**
переноса; после перемещения сверить SHA-256 по каждому старому/новому пути.
В pre-migration списке учитывать только файлы, существовавшие до создания
manifest; сам manifest и его производные исключить из собственного набора,
чтобы не включать рекурсивный собственный хэш. `data/raw/` читать только.

## 5. Известные проблемы (зафиксировать, не исправлять)

1. `model_tradeoff.png`: сообщено о перепутанных подписях задержек правой панели.
   Сверить числовые значения с `inference_benchmark.csv`; изображение не править.
2. `complex_mlp_lr_c_vs_steps.png`: содержит только график c. Не перерисовывать и
   не дополнять.
3. `plateau_seed_study.png`: один цвет используется одновременно для seed и числа
   файлов. Палитру/легенду на этом этапе не менять.
4. E006: в имеющемся CSV число шагов для seed 29/43 и N=7 не достигает 100 000;
   не утверждать, что все seed сравнивались до максимального шага, до проверки
   логов/чекпоинтов.
5. Потенциальное несоответствие source CSV с графиком E008 и E009; установить по
   исходному коду построения, ничего не перерисовывая.

## 6. Предложение по будущему разделению кода в `src/` (не часть миграции)

После завершения реорганизации отдельно обсудить выделение стабильной библиотеки
(предварительно `data_io.py`, `metrics.py`, `models.py`, `training.py`, `paths.py`)
и тонких модулей/CLI конкретных экспериментов. Сейчас обучение, summarizers,
сравнение моделей и обработка данных перемешаны в одном namespace. На этапе 2
этот рефакторинг не выполнять: оставить `src/constellation_denoiser/` и менять
только файловые пути/входы/выходы.

## 7. Черновики будущих файлов (создаются только на этапе 2)

### `AGENTS.md`

```text
# Правила работы с экспериментами
- Каждый новый запуск — новая папка experiments/E00N_название/; старые результаты
  не перезаписывать.
- До запуска в README эксперимента записать вопрос и правило интерпретации.
- В config.json сохранять все параметры, seed, git-коммит и хэш манифеста данных;
  неизвестные исторические значения помечать unknown, не восстанавливать догадкой.
- Не использовать test для выбора checkpoint.
- После запуска дописать запись в конец LOG.md; LOG.md — append-only.
- Не называть утверждение выводом без числа и ссылки на файл результата.
- Указывать ограничения, число seed, файлов и шагов.
- Данные и checkpoint не редактировать; data/raw/ использовать только для чтения.
- Графики именовать E00N_<название>.png; порядок задавать fig_no в
  paper/figures_manifest.csv.
```

### Корневой `README.md` — карта и запуск

```text
# Constellation denoiser
Карта: data/raw — исходные данные (read-only); data/manifest — SHA-256, уникальные
файлы, split и аудит; src/constellation_denoiser — Python-код; experiments/E00N —
протокол/конфиг/run.sh/results/figures/logs каждого опыта; checkpoints/E00N —
локальные веса, не в Git; paper — manifest графиков и презентация; notebooks/archive
— блокноты 01–06; references — источники и архивные описания.

Окружение и зависимости: согласно pyproject.toml. Запускать только из корня
репозитория; перед фактическим запуском сначала читать README соответствующего E.
Пример: bash experiments/E003_large_scale_pilot/run.sh --help.
Для воспроизводимого запуска использовать параметры из config.json и run.sh;
не запускать обучение при просмотре/проверке файлов.
```

### `FINDINGS.md` — только уже подтверждённые числа с источниками

| № | утверждение (без усиления legacy-текста) | уровень | эксперименты | ограничения |
|---:|---|---|---|---|
| 1 | В пилотной таблице test BER mean для Transformer — 0.009120466; raw — 0.010219502. | предварительно | E001 | PR9–PR10; три инициализации; один фиксированный split. `model_comparison_summary.csv`. |
| 2 | В legacy summary указаны paper-like test BER: tanh 0.008945 и complex 0.009745. | предварительно | E002 | PR9–PR10 ранее использованы; методика статьи воспроизведена не полностью. `paper_real_mlp.csv`, `paper_complex_mlp.csv`, `summary.md`; числа сверить с CSV перед финальным внесением. |
| 3 | Для масштабного пилота в bootstrap JSON: raw 0.00955186, real 0.00670150, complex 0.00355219; complex улучшил 45/45 test-файлов. | предварительно | E003 | 300 файлов из 1 356, один split, три seed; bootstrap по test-файлам не заменяет повторение split. `large_scale_pilot_bootstrap.json`. |
| 4 | В seed-17 таблице E004 при 7 train-файлах complex test BER 0.010160; при 210 — 0.003578. | предварительно | E004 | Один seed для основной кривой, повторно использован fixed test из 45 файлов. `learning_curve_seed17.csv`. |
| 5 | Для E005 CSV содержит строки обеих моделей, N=7/30/105/210 и step=20 000; например complex N=210 test BER 0.00462680. | факт таблицы; не обобщать | E005 | Один фиксированный тестовый split; конфигурацию запуска/seed сверить. `fixed_step_learning_curve.csv`. |
| 6–9 | Выводы не зафиксированы в черновике до ручной проверки полных CSV/logs; не придумывать. | unknown | E006–E009 | Публиковать только после числовой проверки CSV и ограничений. |

При создании итогового FINDINGS числовые значения из summary перепроверить в
указанных CSV; если CSV не подтверждает дословно — не переносить соответствующее
утверждение. Столбцы целевого файла: `№ | утверждение | уровень
(предварительно/подтверждено) | эксперименты | ограничения`.

### `config.json` по экспериментам — черновик распределения источников

Во все девять JSON добавить поля `experiment_id`, `git_commit`, `data_manifest_sha256`
и `date`. Git SHA и хэш заполняются только после создания pre-migration snapshot и
manifest; фактическая историческая дата запуска остаётся `unknown`, если лог/metadata
её не подтверждает. Не приравнивать дату mtime к дате запуска.

| Файл | Известные параметры/источник; остальные поля |
|---|---|
| `E001.../config.json` | Скопировать ключи `window`, `seeds`, `epochs`, `patience`, `batch_size`, `learning_rate`, `weight_decay`, `threads`, `device`, три списка PR split, `selection`, `training_target` из `experiment_config.json`; дописать модельный набор из CSV/summary. `date` и snapshot/hash пока `unknown`. |
| `E002.../config.json` | Sweep: `configs=[[7,64],[15,64],[31,64],[31,128]]`, `seeds=[17,29,43]`, tanh, validation PR8, test PR9–PR10, selection из `tanh_mlp_sweep_config.json`. Paper-like обучения: недостающие параметры отдельно `unknown`, существующие факты брать из legacy summary без обобщения на sweep. |
| `E003.../config.json` | Скопировать `large_scale_pilot_config.json`: `available_unique_files=1356`, `sample_files=300`, `split_seed=20261003`, списки split, batch/epochs/patience/threads/seeds из legacy-описания/логов. Пути сделать относительными в плане миграции; значения данных не менять. |
| `E004.../config.json` | Скопировать `learning_curve_config.json`: размеры, seed и полный pool без изменения значений; test/validation 45 файлов из split E003; модели/optimizer — только из `learning_curve_summary.md`; git/date/hash `unknown` до snapshot. |
| `E005.../config.json` | База — shared `learning_curve_config.json`; `N=[7,30,105,210]`, `steps=20000`, модели tanh/complex подтверждены CSV. Seed и остальные специфичные параметры запусков: `unknown` до логов. |
| `E006.../config.json` | Нет отдельного `*_config.json`; seeds заявлены 17/29/43, целевой шаг 100000 из постановки, но фактическое покрытие неполное. Перечень фактически подтверждённых N/seed/steps записать из CSV; отсутствующие значения — `unknown`, discrepancy не скрывать. |
| `E007.../config.json` | Нет отдельного config; окно/ширина/seeds/N восстановимы из имен чекпоинтов и `complex_mlp_sweep_summary.csv`, и только эти точные combinations перечислить. Остальные настройки — `unknown`. |
| `E008.../config.json` | Нет отдельного config; cosine schedule и обучаемый c следуют из script/output имен. Точные LR/шаги/seed/window/width занести только из исходного скрипта, CSV и checkpoint metadata; пока неизвестные поля `unknown`. |
| `E009.../config.json` | Нет отдельного config; N=30, w=31, h=48, seeds 17/29 и варианты a–d видны по именам checkpoint; значения activation/merge определить по коду/CSV, все отсутствующее `unknown`. |

### `LOG.md` — черновые append-only записи E001–E009

Исторические записи не утверждают, что правила интерпретации были зафиксированы
до запуска, если файл этого не подтверждает. Поэтому неподтверждённое
`Правило интерпретации (до запуска)` и неизвестная команда явно помечены
`unknown`. Значения ниже — только числа из существующих файлов; не дополнять их
пересчитанными итогами без проверки источника.

```text
## E001 — Пилот моделей — 2026-09-28 (mtime; фактическая дата запуска unknown)
Вопрос: Сравнить MLP, CNN, Bi-LSTM, Transformer и линейный FFE на PR1–PR10.
Правило интерпретации (до запуска): unknown.
Что запускали: python -m constellation_denoiser.experiment --seeds 17 29 43
  (команда из корневого README); точная историческая команда unknown.
Результат: model_comparison_summary.csv: Transformer test BER mean 0.009120466
  (3 seed), raw 0.010219502; inference_benchmark.csv содержит benchmark задержки.
Вывод: только legacy-описание; самостоятельный вывод на этом этапе не добавлять.
Ограничения: окно 7; PR1–7 train, PR8 validation, PR9–10 test; три seed.
Что я понял:
Следующий шаг: проверка на групповом split/новых PR — только как записано в legacy summary.

## E002 — tanh-MLP и paper-like complex MLP — 2026-09-29 (mtime; запуск unknown)
Вопрос: Проверить tanh sweep и paper-like tanh/complex MLP на PR9–PR10.
Правило интерпретации (до запуска): unknown.
Что запускали: точная команда unknown; источники — tanh_mlp_sweep_config.json
  и код paper_real_mlp.py/paper_complex_mlp.py.
Результат: legacy summary сообщает paper-like tanh BER 0.008945,
  complex BER 0.009745 при raw 0.010181; таблицы paper_real_mlp.csv и
  paper_complex_mlp.csv подлежат сверке перед переносом этих агрегатов.
Вывод: выводы не переписывать; legacy формулировка и её ограничения в summary.md.
Ограничения: PR9–PR10 уже использовались; три seed в sweep; paper конфиг неполон.
Что я понял:
Следующий шаг: независимый тест — только как указано в legacy summary.

## E003 — Большой пилот и аудит — 2026-10-03/04 (mtime; запуск unknown)
Вопрос: Оценить модели и аудит уникальных последовательностей на архиве.
Правило интерпретации (до запуска): unknown.
Что запускали: PYTHONPATH=src python -m constellation_denoiser.large_scale_pilot
  --sample-files 300 --seeds 17 29 43 --epochs 40 --patience 10
  --batch-size 1024 --threads 6 (команда в legacy summary).
Результат: large_scale_pilot_bootstrap.json: raw mean BER 0.00955186,
  real 0.00670150, complex 0.00355219; complex improved 45/45 test files.
  data_audit_all.csv/summary.md: 2 010 CSV, 1 356 unique by SHA-256.
Вывод: legacy summary называет результат пилотным; новые выводы не добавлять.
Ограничения: 300/1 356 файлов; один split, три seed; bootstrap не повторяет split.
Что я понял:
Следующий шаг: проверить grouped split/cross-validation — см. legacy summary.

## E004 — Кривая числа train-файлов — 2026-10-04 (mtime; запуск unknown)
Вопрос: Сравнить BER при вложенном размере train на фиксированных 45 test-файлах.
Правило интерпретации (до запуска): unknown.
Что запускали: точная команда unknown; параметры — learning_curve_config.json.
Результат: learning_curve_seed17.csv: complex test BER 0.010160 при N=7 и
  0.003578 при N=210; таблица содержит также real и промежуточные N.
Вывод: legacy learning_curve_summary.md; здесь новых обобщений нет.
Ограничения: основной график — seed 17, один ранее использованный holdout из 45 файлов.
Что я понял:
Следующий шаг: дополнительные seed/split и новый test — как указано в summary.

## E005 — Равное число шагов — 2026-10-04 (mtime; запуск unknown)
Вопрос: Сравнить N=7/30/105/210 при фиксированных 20 000 шагах.
Правило интерпретации (до запуска): unknown.
Что запускали: точная команда unknown; существующие журналы fixed_step_tanh.log
  и fixed_step_complex.log.
Результат: fixed_step_learning_curve.csv содержит все 8 сочетаний
  (две модели × четыре N) для step=20000; complex N=210 test_ber 0.0046267987.
Вывод: формулировку не добавлять до проверки полного результата и протокола.
Ограничения: seed/полный split в найденном config не указаны; unknown.
Что я понял:
Следующий шаг: unknown.

## E006 — Плато и seed — 2026-10-04 (mtime; запуск unknown)
Вопрос: Проверить плато и разброс между seed при заявленных 100 000 шагах.
Правило интерпретации (до запуска): unknown.
Что запускали: точная команда unknown; есть plateau_a.log и plateau_b.log.
Результат: plateau_seed_study.csv содержит N=7/210, seed 17 до step=100000;
  N=7, seed 29/43 представлены до step=20000. Пример строки:
  complex,N=7,seed=17,step=20000,test_ber=0.0046549944.
Вывод: не заявлять сравнение всех seed на 100000; сначала сверить логи/checkpoints.
Ограничения: покрытие seed/N/step неполное или неясное; один цвет в графике кодирует
  и seed, и число файлов.
Что я понял:
Следующий шаг: установить из логов, были ли прерваны запуски; не перезапускать здесь.

## E007 — Sweep окна и ширины complex MLP — 2026-10-04 (mtime; запуск unknown)
Вопрос: Сравнить комбинации окна и ширины.
Правило интерпретации (до запуска): unknown.
Что запускали: точная команда unknown; run_complex_mlp_sweep.sh и два seed log.
Результат: complex_mlp_sweep_summary.csv: первая строка window=65, hidden=48,
  mean_test_ber=0.0039756944; остальные комбинации находятся в CSV.
Вывод: новый вывод не записывать без заранее заданного правила.
Ограничения: summary содержит два seed; детали split/условий — сверить с кодом.
Что я понял:
Следующий шаг: unknown.

## E008 — Косинусный lr и коэффициенты c — 2026-10-04 (mtime; запуск unknown)
Вопрос: Проверить cosine lr и динамику обучаемых коэффициентов c.
Правило интерпретации (до запуска): unknown.
Что запускали: точная команда unknown; run_complex_mlp_lr_sweep.sh.
Результат: complex_mlp_lr_summary.csv: для window=31, hidden=32 указаны
  reduction A=51.456362% и B=43.248097%; diagnostics CSV содержит траектории c.
Вывод: не заключать о превосходстве расписания без сверки полного CSV/правила.
Ограничения: график complex_mlp_lr_c_vs_steps.png показывает только c.
Что я понял:
Следующий шаг: unknown.

## E009 — Абляция активаций и merge — 2026-10-11 (mtime; запуск unknown)
Вопрос: Сравнить варианты Kerr-активаций и проверить слияние линейных слоёв.
Правило интерпретации (до запуска): unknown.
Что запускали: точная команда unknown; сохранить и проверить оба launcher script.
Результат: complex_mlp_ablation_summary.csv: baseline mean_improvement_pct
  61.727363%; complex_mlp_ablation_merge_check.csv: variant a, seed 17,
  merge_max_abs_diff=4.549127e-07, 1000 окон, ok=True.
Вывод: новые утверждения не добавлять до проверки всех строк/вариантов.
Ограничения: в логах разные seed/варианты; ablation_launch.pid требует проверки
  статуса, не останавливая процесс.
Что я понял:
Следующий шаг: unknown.
```

## 8. Порядок выполнения этапа 2 и риски

### Последовательность

1. Перед началом зафиксировать, что текущий Git-репозиторий уже содержит
   неотслеживаемые файлы и удалённый `qam16_demod_ber.ipynb`; сверить итоговый
   snapshot scope. Добавить правила `.gitignore` для `data/raw/`, `checkpoints/`,
   `*.pt`, `*.pyc`, `__pycache__/`, `tree.txt`, не игнорируя код и документацию.
   Не помещать CSV/чекпоинты в индекс.
2. Создать pre-migration commit и тег `pre-migration`. Убедиться, что это снимок
   текущего состояния проекта без данных/весов и что удаление старого tracked
   notebook отражено намеренно.
3. Сохранить pre-migration SHA-256 manifest всех файлов `data/` и `reports/results/`
   в `data/manifest/` до любого перемещения; отдельно сохранить число файлов и
   список старый путь → SHA-256. Сырьё читать только.
4. Создать целевые каталоги E001–E009 и `data/manifest`, `paper`,
   `notebooks/archive`; положить только файлы, предусмотренные одобренной таблицей.
   Выполнять перемещения средствами Git там, где файлы отслеживаются; для
   неотслеживаемых файлов сохранять тот же принцип контроля хэшей.
5. Исправить пути в Python/sh/config/Markdown и добавить `paths.py`; кодовую
   структуру `src/` не менять. Для E1–E9 подготовить `README.md`, `config.json`,
   `run.sh`, `results/`, `figures/`, `logs/`; создать корневые AGENTS/LOG/FINDINGS/
   README, `paper/figures_manifest.csv` и не менять сами CSV/JSON-результаты.
6. Перенести блокноты 01–06 без запуска ячеек, презентацию без пересборки.
7. Сверить количество файлов и SHA-256 всех перенесённых данных, JSON/CSV,
   изображений, логов и checkpoint до/после; проверить все модули импорта, `--help`
   всех скриптов, что все пути из figures manifest существуют, и что `git status`
   не показывает исходные данные/веса как добавленные.
8. Выполнить ровно один smoke-запуск с 100 шагами на одном эксперименте только
   после переноса. `fixed_step_curve.py`, `complex_mlp_lr_sweep.py`,
   `complex_mlp_sweep.py` и `complex_mlp_ablation_train.py` имеют параметр `--steps`;
   выбрать один, передать 100 и направить outputs в отдельную временную папку, чтобы
   не перезаписать исторические результаты/checkpoints. Тренировочный вызов сейчас
   не выполнять.
9. Сформировать итоговый отчёт: перемещённые/оставленные файлы, проверки и
   непрошедшие проверки; итоговый коммит, без переписывания pre-migration commit.

### Риски и условия остановки

- Коллизии имён PR1–PR10 при сведении источников; нужен архивный подкаталог или
  отдельное решение владельца. Ни один файл не перезаписывать и не дедуплицировать.
- Неполное/неодинаковое покрытие E006 до 100 000 шагов; не подменять наблюдения
  ожиданиями нумерации.
- Незакоммиченный проект и удалённый отслеживаемый notebook осложняют snapshot;
  не восстанавливать файл и не коммитить без явного согласования состава.
- Сейчас `.gitignore` не защищает CSV/checkpoints от попадания в Git. Проверить
  индекс и staging перед каждым коммитом, а не только наличие ignore-правила.
- JSON-конфиг большого пилота содержит абсолютные пути. Менять только поля путей,
  не числовые значения и не списки файлов.
- Несколько launcher/log вариантов для E009; PID-файл не является разрешением
  останавливать процесс. Если активный запуск использует файлы, необходимые для
  миграции, остановиться и запросить решение, не прерывая его.
- В презентации и старых figures могут быть дефекты подписей. На этапе 2 только
  переименовать/переместить существующие бинарные изображения; исправление возможно
  лишь отдельной задачей после разрешения.
- Наличие CLI `--steps` само по себе не гарантирует, что команда 100 шагов не
  перезапишет основной CSV/чекпоинт. Перед smoke-run проверить поддержку отдельного
  output/checkpoint пути; при отсутствии такой поддержки остановиться и согласовать
  безопасный способ, не менять исторические артефакты.
