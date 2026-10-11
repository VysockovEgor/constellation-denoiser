# E007 — Sweep окна и ширины complex MLP

## Вопрос
Сравнить конфигурации окна и hidden width комплексной MLP.

## Правило интерпретации (до запуска)
`unknown`: доступные исходные файлы не содержат предварительного критерия.

## Протокол
- Конфигурации (window, hidden): (65,48), (31,32), (127,48), (65,32),
  (31,48), (127,32).
- Train files: 30; seed: 17, 29; steps: 20 000; checkpoint interval: 1 000;
  learning rate: 0.001.
- Историческая команда: `bash experiments/E007_window_width_sweep/scripts/run_complex_mlp_sweep.sh`;
  безопасный wrapper находится в `run.sh`.

## Результат
`results/complex_mlp_sweep_summary.csv` содержит mean test BER 0.0036557485086896498
для window=31, hidden=48. График: `figures/E007_complex_mlp_sweep.png`.

## Вывод
Новая интерпретация результатов не добавлялась.

## Ограничения
Два seed и фиксированный split; результат — сводка существующих запусков.
