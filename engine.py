from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import polars as pl
from numba import njit


@dataclass(frozen=True)
class DatasetArtifacts:
    parquet_path: Path
    quality: Dict[str, float]


@dataclass(frozen=True)
class BacktestOutput:
    metrics: Dict[str, float]
    equity_curve: np.ndarray
    trades: Dict[str, np.ndarray]


def _csv_has_header(csv_path: Path) -> bool:
    first_line = csv_path.read_text(encoding="utf-8").splitlines()[0].strip().lower()
    return first_line.startswith("date,")


def load_and_prepare_csv(csv_path: str | Path, parquet_cache_path: str | Path) -> DatasetArtifacts:
    """
    Parse CSV with shape: date,time,open,high,low,close,volume.
    Supports with/without header. Builds timestamp_utc from date+time.
    Sorts ascending. Drops duplicate timestamps keeping first seen row.
    """
    csv_path = Path(csv_path)
    parquet_cache_path = Path(parquet_cache_path)
    has_header = _csv_has_header(csv_path)

    cols = ["date", "time", "open", "high", "low", "close", "volume"]
    df = pl.read_csv(
        csv_path,
        has_header=has_header,
        new_columns=None if has_header else cols,
        try_parse_dates=False,
    )
    if has_header:
        df = df.rename({c: c.strip().lower() for c in df.columns})
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = (
        df.select(cols)
        .with_columns(
            [
                (pl.col("date") + pl.lit(" ") + pl.col("time"))
                .str.strptime(pl.Datetime, format="%Y.%m.%d %H:%M", strict=True)
                .alias("timestamp_utc"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            ]
        )
        .drop(["date", "time"])
        .sort("timestamp_utc")
    )

    total_rows = df.height
    duplicate_count = int(df.select(pl.col("timestamp_utc").is_duplicated().sum()).item())
    if duplicate_count > 0:
        df = df.unique(subset=["timestamp_utc"], keep="first", maintain_order=True)

    ts = df["timestamp_utc"]
    diffs = ts.diff().drop_nulls().dt.total_minutes()
    gaps_over_1m = int((diffs > 1).sum()) if diffs.len() > 0 else 0

    quality = {
        "total_rows": float(total_rows),
        "rows_after_dedup": float(df.height),
        "duplicate_count": float(duplicate_count),
        "gaps_over_1m": float(gaps_over_1m),
        "has_gaps": float(gaps_over_1m > 0),
    }

    parquet_cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_cache_path)
    return DatasetArtifacts(parquet_path=parquet_cache_path, quality=quality)


def load_cached_arrays(parquet_path: str | Path) -> Dict[str, np.ndarray]:
    df = pl.read_parquet(parquet_path)
    arrays = {
        "timestamp_utc": df["timestamp_utc"].to_numpy(),
        "open": df["open"].to_numpy(),
        "high": df["high"].to_numpy(),
        "low": df["low"].to_numpy(),
        "close": df["close"].to_numpy(),
        "volume": df["volume"].to_numpy(),
    }
    return arrays


def _ema(values: np.ndarray, length: int) -> np.ndarray:
    out = np.empty_like(values)
    alpha = 2.0 / (length + 1.0)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rsi(close: np.ndarray, length: int) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ema(gain, length)
    avg_loss = _ema(loss, length)
    rs = np.divide(avg_gain, avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return _ema(tr, length)


def _rolling_mean_std(values: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(values)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    csum = np.cumsum(values)
    csum2 = np.cumsum(values * values)
    for i in range(length - 1, n):
        s = csum[i] - (csum[i - length] if i >= length else 0.0)
        s2 = csum2[i] - (csum2[i - length] if i >= length else 0.0)
        m = s / length
        var = max(s2 / length - m * m, 0.0)
        mean[i] = m
        std[i] = np.sqrt(var)
    mean[: length - 1] = mean[length - 1]
    std[: length - 1] = std[length - 1]
    return mean, std


def _donchian(high: np.ndarray, low: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(high)
    upper = np.empty(n)
    lower = np.empty(n)
    for i in range(n):
        start = max(0, i - length + 1)
        upper[i] = np.max(high[start : i + 1])
        lower[i] = np.min(low[start : i + 1])
    return upper, lower


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    up = np.diff(high, prepend=high[0])
    down = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _atr(high, low, close, length)
    plus_di = 100 * _ema(plus_dm, length) / (atr + 1e-12)
    minus_di = 100 * _ema(minus_dm, length) / (atr + 1e-12)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
    return _ema(dx, length)


def build_indicators(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    ema_fast_len: int,
    ema_slow_len: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    adx_len: int,
    rsi_len: int,
    bb_len: int,
    bb_std: float,
    donchian_len: int,
    atr_len: int,
) -> Dict[str, np.ndarray]:
    ema_fast = _ema(close, ema_fast_len)
    ema_slow = _ema(close, ema_slow_len)

    macd_line = _ema(close, macd_fast) - _ema(close, macd_slow)
    macd_sig = _ema(macd_line, macd_signal)

    adx = _adx(high, low, close, adx_len)
    rsi = _rsi(close, rsi_len)

    bb_mid, bb_sigma = _rolling_mean_std(close, bb_len)
    bb_up = bb_mid + bb_std * bb_sigma
    bb_low = bb_mid - bb_std * bb_sigma

    donch_up, donch_low = _donchian(high, low, donchian_len)
    atr = _atr(high, low, close, atr_len)

    return {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "macd_line": macd_line,
        "macd_signal": macd_sig,
        "adx": adx,
        "rsi": rsi,
        "bb_mid": bb_mid,
        "bb_up": bb_up,
        "bb_low": bb_low,
        "donch_up": donch_up,
        "donch_low": donch_low,
        "atr": atr,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }


@njit(cache=True)
def _run_backtest_numba(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    hour_utc: np.ndarray,
    ema_fast: np.ndarray,
    ema_slow: np.ndarray,
    macd_line: np.ndarray,
    macd_signal: np.ndarray,
    adx: np.ndarray,
    rsi: np.ndarray,
    bb_mid: np.ndarray,
    bb_up: np.ndarray,
    bb_low: np.ndarray,
    donch_up: np.ndarray,
    donch_low: np.ndarray,
    atr: np.ndarray,
    spread_noise: np.ndarray,
    seed_bias: float,
    use_ema_cross: int,
    use_macd: int,
    use_adx_filter: int,
    adx_threshold: float,
    use_rsi: int,
    rsi_oversold: float,
    rsi_overbought: float,
    use_bollinger: int,
    bb_mode: int,
    use_donchian_breakout: int,
    risk_percent: float,
    sl_atr_multiplier: float,
    tp_atr_multiplier: float,
    use_trailing_stop: int,
    trailing_atr_distance: float,
    initial_equity: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    equity_curve = np.empty(n)
    equity = initial_equity

    entry_idx = np.full(n, -1)
    exit_idx = np.full(n, -1)
    side_arr = np.zeros(n)
    pnl_arr = np.zeros(n)

    in_pos = 0
    side = 0
    entry = 0.0
    stop = 0.0
    take = 0.0
    size = 0.0
    eidx = 0
    trade_count = 0

    for t in range(n - 1):
        h = hour_utc[t]
        sess = 0.00008
        if 7 <= h < 13:
            sess = 0.00005
        elif 13 <= h < 21:
            sess = 0.00006

        spread = sess * (1.0 + 0.15 * spread_noise[t] + 0.01 * seed_bias)
        if spread < 0.0:
            spread = sess

        slip = (atr[t] / (close[t] + 1e-12)) * 0.25

        if in_pos == 1:
            if use_trailing_stop == 1:
                trail = close[t] - trailing_atr_distance * atr[t] if side == 1 else close[t] + trailing_atr_distance * atr[t]
                if side == 1 and trail > stop:
                    stop = trail
                if side == -1 and trail < stop:
                    stop = trail

            hit_stop = 0
            hit_take = 0
            if side == 1:
                hit_stop = 1 if low[t] <= stop else 0
                hit_take = 1 if high[t] >= take else 0
            else:
                hit_stop = 1 if high[t] >= stop else 0
                hit_take = 1 if low[t] <= take else 0

            if hit_stop == 1 or hit_take == 1:
                exit_price = stop  # worst-case intrabar rule: stop first if both
                gross = (exit_price - entry) * side
                pnl = gross * size
                equity += pnl
                exit_idx[trade_count] = t
                pnl_arr[trade_count] = pnl
                in_pos = 0
                trade_count += 1

        bullish = 0
        bearish = 0

        if use_ema_cross == 1:
            bullish += 1 if ema_fast[t] > ema_slow[t] else 0
            bearish += 1 if ema_fast[t] < ema_slow[t] else 0

        if use_macd == 1:
            bullish += 1 if macd_line[t] > macd_signal[t] else 0
            bearish += 1 if macd_line[t] < macd_signal[t] else 0

        if use_rsi == 1:
            bullish += 1 if rsi[t] <= rsi_oversold else 0
            bearish += 1 if rsi[t] >= rsi_overbought else 0

        if use_bollinger == 1:
            if bb_mode == 0:
                bullish += 1 if close[t] < bb_low[t] else 0
                bearish += 1 if close[t] > bb_up[t] else 0
            else:
                bullish += 1 if close[t] > bb_up[t] else 0
                bearish += 1 if close[t] < bb_low[t] else 0

        if use_donchian_breakout == 1:
            bullish += 1 if close[t] >= donch_up[t] else 0
            bearish += 1 if close[t] <= donch_low[t] else 0

        adx_ok = 1
        if use_adx_filter == 1:
            adx_ok = 1 if adx[t] >= adx_threshold else 0

        long_signal = 1 if bullish > 0 and bearish == 0 and adx_ok == 1 else 0
        short_signal = 1 if bearish > 0 and bullish == 0 and adx_ok == 1 else 0

        if in_pos == 0 and (long_signal == 1 or short_signal == 1):
            side = 1 if long_signal == 1 else -1
            px = open_[t + 1]
            entry = px + (spread / 2.0 + slip) if side == 1 else px - (spread / 2.0 + slip)
            sl_dist = max(sl_atr_multiplier * atr[t], 1e-8)
            tp_dist = tp_atr_multiplier * atr[t]
            stop = entry - sl_dist if side == 1 else entry + sl_dist
            take = entry + tp_dist if side == 1 else entry - tp_dist

            risk_dollars = equity * risk_percent
            size = risk_dollars / sl_dist
            if size < 0:
                size = 0.0

            in_pos = 1
            eidx = t + 1
            entry_idx[trade_count] = eidx
            side_arr[trade_count] = side

        equity_curve[t] = equity

    equity_curve[n - 1] = equity
    return equity_curve, entry_idx[:trade_count], exit_idx[:trade_count], side_arr[:trade_count], pnl_arr[:trade_count]


def compute_metrics(equity_curve: np.ndarray, pnl: np.ndarray, initial_equity: float) -> Dict[str, float]:
    net_profit_abs = float(equity_curve[-1] - initial_equity)
    net_profit_pct = (net_profit_abs / initial_equity) * 100.0
    trade_count = len(pnl)

    gross_profit = float(np.sum(np.where(pnl > 0, pnl, 0.0)))
    gross_loss = float(np.abs(np.sum(np.where(pnl < 0, pnl, 0.0))))
    pf = gross_profit / (gross_loss + 1e-12)

    peaks = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peaks) / (peaks + 1e-12)
    max_dd_pct = float(np.min(dd) * 100.0)

    rets = np.diff(equity_curve) / (equity_curve[:-1] + 1e-12)
    mu = float(np.mean(rets)) if rets.size else 0.0
    sigma = float(np.std(rets)) if rets.size else 0.0
    downside = rets[rets < 0]
    downside_std = float(np.std(downside)) if downside.size else 0.0

    annualizer = np.sqrt(365.0 * 24.0 * 60.0)
    sharpe = (mu / (sigma + 1e-12)) * annualizer
    sortino = (mu / (downside_std + 1e-12)) * annualizer
    calmar = (net_profit_pct / (abs(max_dd_pct) + 1e-12))

    return {
        "net_profit_abs": net_profit_abs,
        "net_profit_pct": float(net_profit_pct),
        "trade_count": float(trade_count),
        "profit_factor": float(pf),
        "max_drawdown_pct": float(abs(max_dd_pct)),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
    }


def score_trial(metrics: Dict[str, float]) -> float:
    trades = metrics["trade_count"]
    pf = metrics["profit_factor"]
    dd = metrics["max_drawdown_pct"]
    if trades < 50:
        return 0.0
    if dd > 30.0:
        return 0.0

    penalty_trades = 1.0 - np.exp(-(trades - 50.0) / 100.0)
    penalty_pf = 1.0 / (1.0 + np.exp(-12.0 * (pf - 1.1)))
    return float(metrics["net_profit_pct"] * penalty_trades * penalty_pf)


def run_backtest(
    arrays: Dict[str, np.ndarray],
    indicators: Dict[str, np.ndarray],
    params: Dict[str, float],
    seed: int,
    initial_equity: float = 10000.0,
) -> BacktestOutput:
    ts = arrays["timestamp_utc"].astype("datetime64[m]")
    minutes = (ts - ts.astype("datetime64[D]")).astype("timedelta64[m]").astype(np.int64)
    hour_utc = (minutes // 60).astype(np.int64)

    rng = np.random.default_rng(seed)
    spread_noise = rng.standard_normal(len(arrays["close"]))

    out = _run_backtest_numba(
        arrays["open"],
        arrays["high"],
        arrays["low"],
        arrays["close"],
        hour_utc,
        indicators["ema_fast"],
        indicators["ema_slow"],
        indicators["macd_line"],
        indicators["macd_signal"],
        indicators["adx"],
        indicators["rsi"],
        indicators["bb_mid"],
        indicators["bb_up"],
        indicators["bb_low"],
        indicators["donch_up"],
        indicators["donch_low"],
        indicators["atr"],
        spread_noise,
        float(seed % 997),
        int(params["use_ema_cross"]),
        int(params["use_macd"]),
        int(params["use_adx_filter"]),
        float(params["adx_threshold"]),
        int(params["use_rsi"]),
        float(params["rsi_oversold"]),
        float(params["rsi_overbought"]),
        int(params["use_bollinger"]),
        int(params["bb_mode"]),
        int(params["use_donchian_breakout"]),
        float(params["risk_percent"]),
        float(params["sl_atr_multiplier"]),
        float(params["tp_atr_multiplier"]),
        int(params["use_trailing_stop"]),
        float(params["trailing_atr_distance"]),
        initial_equity,
    )

    equity_curve, entry_idx, exit_idx, side, pnl = out
    metrics = compute_metrics(equity_curve, pnl, initial_equity)
    metrics["score"] = score_trial(metrics)

    trades = {
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "side": side,
        "pnl": pnl,
    }
    return BacktestOutput(metrics=metrics, equity_curve=equity_curve, trades=trades)
