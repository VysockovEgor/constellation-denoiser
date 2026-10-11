# E008 — Cosine learning rate и коэффициенты c

## Вопрос
Сравнить расписания learning rate и сохранить диагностику обучаемых коэффициентов c.

## Правило интерпретации (до запуска)
`unknown`: в сохранённых файлах предварительное правило не найдено.

## Протокол
- Комплексная MLP; конфигурации (window, hidden): (31,32), (31,48), (65,48).
- Train files: 30; seed: 17, 29; 20 000 шагов; checkpoint interval 1 000.
- Начальный learning rate 0.001; schedule `cosine`; диагностика на шагах
  0, 1 000, 5 000, 20 000.
- Историческая команда: `bash experiments/E008_cosine_lr_coefficients/scripts/run_complex_mlp_lr_sweep.sh`;
  безопасный wrapper находится в `run.sh`.

## Результат
`results/complex_mlp_lr_summary.csv`: для window=31, hidden=48 reduction
constant A=61.72736325485471%, cosine B=58.76306991962524%.
График `figures/E008_complex_mlp_lr_c_vs_steps.png` содержит только график c.

## Вывод
Новая интерпретация не добавлялась.

## Ограничения
Два seed и фиксированный split. Известная неполнота графика c не исправлялась.
