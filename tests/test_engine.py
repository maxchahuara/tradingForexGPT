import numpy as np

from engine import build_indicators, run_backtest, _run_backtest_numba


def _base_arrays(n: int = 300):
    close = np.linspace(1.0, 1.3, n)
    open_ = close.copy()
    high = close + 0.001
    low = close - 0.001
    ts = np.array([np.datetime64('2024-01-01T00:00') + np.timedelta64(i, 'm') for i in range(n)])
    return {
        "timestamp_utc": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.zeros(n),
    }


def _params():
    return {
        "use_ema_cross": 1,
        "use_macd": 0,
        "use_adx_filter": 0,
        "adx_threshold": 20.0,
        "use_rsi": 0,
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0,
        "use_bollinger": 0,
        "bb_mode": 0,
        "use_donchian_breakout": 0,
        "risk_percent": 0.01,
        "sl_atr_multiplier": 2.0,
        "tp_atr_multiplier": 3.0,
        "use_trailing_stop": 0,
        "trailing_atr_distance": 2.0,
    }


def _indicators(arrays):
    return build_indicators(
        arrays["open"], arrays["high"], arrays["low"], arrays["close"],
        ema_fast_len=10,
        ema_slow_len=30,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        adx_len=14,
        rsi_len=14,
        bb_len=20,
        bb_std=2.0,
        donchian_len=20,
        atr_len=14,
    )


def test_seed_reproducibility():
    arrays = _base_arrays()
    indicators = _indicators(arrays)
    params = _params()

    r1 = run_backtest(arrays, indicators, params, seed=42)
    r2 = run_backtest(arrays, indicators, params, seed=42)

    assert np.allclose(r1.equity_curve, r2.equity_curve)
    assert np.allclose(r1.trades["pnl"], r2.trades["pnl"])


def test_intrabar_worst_case_stop_first():
    n = 5
    open_ = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    high = np.array([1.0, 1.05, 1.2, 1.0, 1.0])
    low = np.array([1.0, 0.95, 0.8, 1.0, 1.0])
    close = np.array([1.0, 1.01, 1.0, 1.0, 1.0])

    zeros = np.zeros(n)
    ones = np.ones(n)
    hour = np.zeros(n, dtype=np.int64)

    equity, eidx, xidx, side, pnl = _run_backtest_numba(
        open_, high, low, close, hour,
        ones * 2, ones, zeros, zeros, ones * 50, zeros,
        zeros, ones * 2, -ones * 2,
        ones * 2, -ones * 2,
        ones * 0.1, zeros,
        0.0,
        1, 0, 0, 20.0, 0, 30.0, 70.0, 0, 0, 0,
        0.01, 1.0, 1.0, 0, 1.0, 10000.0,
    )

    assert len(pnl) >= 1
    assert pnl[0] < 0


def test_no_lookahead_entry_next_open():
    arrays = _base_arrays(100)
    arrays["open"][10] = 2.0
    arrays["close"][9] = 1.5
    indicators = _indicators(arrays)
    params = _params()

    result = run_backtest(arrays, indicators, params, seed=7)
    if len(result.trades["entry_idx"]) > 0:
        first_entry = int(result.trades["entry_idx"][0])
        assert first_entry >= 1
