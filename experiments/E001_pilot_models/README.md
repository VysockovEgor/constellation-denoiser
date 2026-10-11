# E001 — Первый пилот моделей

## Вопрос
Сравнить MLP, CNN, Bi-LSTM, Transformer и линейный FFE при окне 7.

## Правило интерпретации (до запуска)
`unknown`: сохранившиеся файлы не содержат предварительно записанного правила.

## Протокол
- Данные и split: PR1–PR7 train, PR8 validation, PR9–PR10 test.
- Seed: 17, 29, 43.
- Модель и шаги: окно 7; 30 эпох максимум, patience 6, batch size 1024,
  learning rate 0.001, weight decay 1e-5; CPU, 6 threads.
- Выбор: lowest validation BER, затем lowest validation MSE. Test не используется
  для выбора checkpoint.
- Историческая точная команда: `unknown`; общий модуль запуска —
  `python -m constellation_denoiser.experiment --seeds 17 29 43`.

## Результат
Таблицы находятся в `results/`; график — `figures/E001_model_tradeoff.png`.
`model_comparison_summary.csv` содержит Transformer test BER mean 0.009120466
и raw BER 0.010219502. Подробные метрики для моделей и FFE не пересчитаны.

## Вывод
Новый вывод задним числом не формулируется; численное сравнение см. в CSV.

## Ограничения
Один фиксированный split и три seed. `model_tradeoff` содержит отмеченную в
плане неоднозначность подписей задержек правой панели; график не менялся.
