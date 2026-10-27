# Очередь E024–E033

Запуск: `bash scripts/remote.sh go q all` (сессия `q_all_<дата>`; ждёт окончания сессий `e020_*`, затем p0 → p1 → p2 → p3 → p4 → p5; E032 — отдельная сессия `q_conv_<дата>`, её запускает p1). Статус: `bash scripts/remote.sh status`; результаты: `bash scripts/remote.sh fetch --journal`; остановка: `bash scripts/remote.sh stop <сессия>`.

| часть | эксперименты | прогонов | оценка, ч | что смотреть |
|---|---|---|---|---|
| p0 | E024 аудит, E031 триплеты | 2 (+1 при утечке) + 150 подгонок | ESTIMATE_P0 | `E024…/results/leakage.json` (останов при утечке), `audit_verdicts.csv`, `../../results/noise_floor.json`; `E031…/results/triplet_summary.csv` |
| p1 | E025 настройка | 2 + 806 + 195 + 171 | ESTIMATE_P1 | `code_check.json` (66.253 ± 0.7), `recipes_table.csv`, `../../results/recipes.json`, `../../results/gate_g1.json` |
| p2 | E026 карта, E028 DBP | 278 + 119 | ESTIMATE_P2 | `map_verdicts.csv`, `instability.csv`; `dbp_verdicts.csv`, `b_per_phase.csv` |
| p3 | E027 активации, E029 front-end | 449 + 82 | ESTIMATE_P3 | `activation_verdicts.csv`; `width_verdicts.csv` |
| p4 | E030 потери | 226 | ESTIMATE_P4 | `loss_verdicts.csv` |
| conv | E032 1M шагов (JOBS 16, параллельно p2–p4) | 20 | ESTIMATE_CONV | `horizons.csv`, `convergence_verdicts.csv` |
| p5 | E033 Парето | 2 + ≤ 50 | ESTIMATE_P5 | `level_best.md`, `dominance.csv`, `pareto_verdicts.csv` |
| всего | | | ESTIMATE_TOTAL | |

Числа прогонов зависимых стадий (S2, S3, confirm, a_long, *_fixed) — по отбору-заглушке; реальный отбор делается по валидации предыдущей стадии. Оценка часов — `bash scripts/remote.sh estimate q <часть>` (короткий замер скорости на сервере, коэффициент конкуренции из `resources.json`).

Автостоп (`results/gate_status.json` → `halt: true`): утечка данных (E024); контроль вне ±0.7 п.п. (каждый эксперимент) или кодовая проверка E025; больше 15 % упавших или разошедшихся прогонов стадии (в стадиях поиска — только упавших); свободной памяти меньше 16 ГБ дольше 30 минут. Чужая нагрузка только уменьшает JOBS (пересчёт перед каждой стадией: min(40, ядра − чужие потоки − 2), при работающей `q_conv` — 24 и 16). Снять останов после разбора причины: удалить или исправить `results/gate_status.json` на сервере (делает владелец) и запустить нужную часть снова (`go q <часть>`; готовые прогоны пропускаются).
