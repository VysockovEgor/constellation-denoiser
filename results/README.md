# Пересчёт E010–E014 в Q² (дБ)

Создано `scripts/recompute_q2.py` (`PYTHONPATH=src python scripts/recompute_q2.py`).
Формула для 16-QAM: `Q2_dB = 10·log10(10·erfcinv(8·BER/3)²)` (`src/constellation_denoiser/q2.py`;
взята из промпта E015–E018 как уравнение (17) Sidelnikov et al., JLT 2021; PDF для сверки не было).

- Источник: `results/runs/*.json` с сервера в Git не попали, поэтому читаются
  поштучные таблицы с теми же полями: `lr_sweep.csv`, `long_runs.csv`,
  `depth_window.csv`, `transformer_runs.csv`, `kan_hybrid.csv`.
- Копии с колонками Q² лежат рядом с оригиналами как `*_q2.csv`; оригиналы
  не изменены.
- `q2_db_raw` — сырой test BER; `q2_db_model`, `q2_gain_db` — test BER на
  последнем шаге; `valbest_q2_*` — тест в точке лучшей валидации, где
  BER = raw·(1 − test_reduction_valbest_select/100). Это точная обратная
  операция к определению reduction при том же raw BER, а не восстановление
  без сырого BER.
- `q2_key_points.csv` — `complex_both`, `complex_a`, `one_hidden` (E012, 20k и
  E011, 200k), `real_tanh` (E011), лучший по валидации трансформер (E013) и
  лучший по валидации гибрид Керр + KAN (E014), с числом умножений после
  слияния (для KAN — разреженная конвенция).
- **Пропуски:** нет. У всех 392 прогонов есть raw BER и test BER, поэтому
  `q2_skipped.json` пуст.
