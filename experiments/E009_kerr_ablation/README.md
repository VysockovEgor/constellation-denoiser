# E009 — Абляция активаций Kerr и merge check

## Вопрос
Сравнить варианты Kerr-активаций a–d и проверить слияние линейных слоёв.

## Правило интерпретации (до запуска)
`unknown`: предварительное правило в сохранившихся файлах отсутствует.

## Протокол
- Complex MLP; window=31, hidden=48; train files=30; seed 17/29.
- 20 000 шагов; learning rate 0.001; diagnostic steps 0, 5 000, 20 000.
- Сохранены два launcher, отдельные логи вариантов и merge-check CSV.
- Историческая команда: `bash experiments/E009_kerr_ablation/scripts/launch_ablation_max.sh`;
  безопасный wrapper находится в `run.sh`.

## Результат
`results/complex_mlp_ablation_summary.csv`: variant a mean improvement
62.19147876804941%. `results/complex_mlp_ablation_merge_check.csv`:
variant a/seed 17, max abs diff 4.549127083919302e-07, 1000 окон, `ok=True`.
График: `figures/E009_complex_mlp_ablation_tradeoff.png`.

## Вывод
Результаты приведены как факты из CSV; новых выводов не добавлял.

## Ограничения
Два seed. В launch log для c/seed17 есть ошибка, при этом итоговый CSV содержит
строку для этого варианта. Существовавший PID-файл был устаревшим, не признаком
активного процесса. Эти неоднозначности не разрешались догадками.
