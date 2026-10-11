# E002 — tanh sweep и paper-like MLP

## Вопрос
Проверить tanh-MLP sweep и paper-like вещественную и комплексную MLP на PR9–PR10.

## Правило интерпретации (до запуска)
`unknown`: исходные артефакты не содержат предварительно записанного критерия.

## Протокол
- Данные: validation PR8, test PR9–PR10; полный train split paper-like runs
  в сохранившемся config не указан (`unknown`).
- Seed: 17, 29, 43.
- Sweep configs: (window, hidden) = (7,64), (15,64), (31,64), (31,128);
  activation tanh; выбор по mean validation BER, MSE для разрешения равенства.
- Целевая переменная I/Q центрального переданного символа; loss MSE.
- Историческая команда: `unknown`; доступны модули `tanh_sweep`,
  `paper_real_mlp` и `paper_complex_mlp`.

## Результат
CSV сохранены в `results/`. `paper_real_mlp.csv` содержит test BER 0.008690534
для seed 43; `paper_complex_mlp.csv` содержит 0.010211569 для seed 17.

## Вывод
Новая интерпретация задним числом не добавлялась.

## Ограничения
Три seed. PR9–PR10 использовались и в E001, поэтому эти результаты не являются
независимым test set. Исторические параметры paper-like моделей не все
зафиксированы.
