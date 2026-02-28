# QuantLab Pro - Backend Architecture (MVP)

## Objetivo MVP de backend
Construir un motor local cliente-servidor que:
1. ingiera CSV 1-min con formato real (`date,time,open,high,low,close,volume`, con o sin header),
2. parsee `timestamp_utc` con `%Y.%m.%d %H:%M`,
3. cachee en Parquet,
4. calcule indicadores fuera del hot loop,
5. ejecute backtest en Numba (`@njit`) con fricción realista,
6. entregue métricas y trades compactos para capas superiores (Optuna/API/UI).

## Componentes

### 1) Ingesta y calidad de datos (Polars)
- `load_and_prepare_csv(...)`:
  - autodetección header/no-header,
  - parseo UTC,
  - orden ascendente,
  - deduplicación por timestamp (keep first),
  - detección de gaps `> 1 min` sin rellenar velas,
  - export Parquet.
- Salida: ruta de cache + métricas de calidad (`duplicate_count`, `gaps_over_1m`, etc.).

### 2) Feature/indicadores (NumPy)
- Calculados **antes** de Numba:
  - EMA, MACD, ADX, RSI, Bollinger, Donchian, ATR.
- Resultado: arrays NumPy listos para pasar al hot loop.

### 3) Núcleo de simulación (Numba)
- `_run_backtest_numba(...)`:
  - vela-a-vela, solo arrays numéricos,
  - toggles enteros (0/1),
  - señal en `close[t]`, entrada en `open[t+1]`,
  - spread estocástico por sesión UTC,
  - slippage dependiente de ATR,
  - resolución intrabar pesimista (si SL y TP en misma vela → STOP primero),
  - gestión de riesgo por `% equity` y distancia a SL,
  - trailing stop opcional,
  - export de equity curve + arrays compactos de trades.

### 4) Métricas y score
- `compute_metrics(...)`: Net Profit abs/%; trades; PF; Max DD%; Sharpe/Sortino/Calmar.
- `score_trial(...)`:
  - `trades < 50 => 0`,
  - `max_dd > 30% => 0`,
  - penalización suave por pocos trades y penalización sigmoide para `PF < 1.1`.

## Reproducibilidad
- `run_backtest(..., seed=...)` usa `np.random.default_rng(seed)` para ruido de spread.
- Evita random global no controlado.

## Escalabilidad y continuidad (siguiente fase)
- `optimizer.py`: Optuna + SQLite `storage` reanudable.
- `api.py`: control de ciclo de optimización y WebSocket de telemetría.
- Frontend: dashboard + leaderboard + chart de velas/trades.

## Nota de rendimiento
Para evitar oversubscription cuando se combine Optuna `n_jobs=-1` con kernels compilados:
- fijar `NUMBA_NUM_THREADS` y/o usar `n_jobs` menor al total de cores físicos,
- evitar paralelismo anidado en múltiples capas al mismo tiempo.
