"""ATR and RSI on a daily price table. No network and no other indicators.

ATR(14) is the simple average of the last 14 true ranges (project.md §6.2).
RSI(14) is Wilder's smoothed RSI. See DECISIONS.md.
"""

from __future__ import annotations

import pandas as pd

_ATR_COLUMNS = ("High", "Low", "Close")


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average of the last ``period`` true ranges, aligned to ``frame``."""
    _check_period(period)
    prices = _require_prices(frame, _ATR_COLUMNS)
    if prices.empty:
        return _empty_series(prices.index, "ATR")
    tr = _true_range(prices["High"], prices["Low"], prices["Close"])
    averaged = tr.rolling(window=period, min_periods=period).mean()
    averaged.name = "ATR"
    return averaged


def rsi(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder RSI. The first value sits on the bar after ``period`` close changes."""
    _check_period(period)
    prices = _require_prices(frame, ("Close",))
    if prices.empty:
        return _empty_series(prices.index, "RSI")

    close = prices["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    values = [float("nan")] * len(close)
    if len(close) > period:
        avg_gain = float(gain.iloc[1 : period + 1].mean())
        avg_loss = float(loss.iloc[1 : period + 1].mean())
        values[period] = _rsi_value(avg_gain, avg_loss)
        for i in range(period + 1, len(close)):
            avg_gain = (avg_gain * (period - 1) + float(gain.iloc[i])) / period
            avg_loss = (avg_loss * (period - 1) + float(loss.iloc[i])) / period
            values[i] = _rsi_value(avg_gain, avg_loss)
    return pd.Series(values, index=close.index, name="RSI", dtype="float64")


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous = close.shift(1)
    parts = pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()],
        axis=1,
    )
    tr = parts.max(axis=1, skipna=True)
    tr.iloc[0] = float(high.iloc[0] - low.iloc[0])
    tr.name = "TR"
    return tr


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _require_prices(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Indicators need a price table.")
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Price table is missing {', '.join(missing)}.")
    data = pd.DataFrame(
        {name: pd.to_numeric(frame[name], errors="coerce") for name in columns},
        index=frame.index,
    )
    if data.isna().any().any():
        raise ValueError("Price columns must be numbers with no gaps.")
    return data


def _check_period(period: int) -> None:
    if isinstance(period, bool) or not isinstance(period, int) or period < 1:
        raise ValueError("period must be a positive integer.")


def _empty_series(index: pd.Index, name: str) -> pd.Series:
    return pd.Series(dtype="float64", index=index, name=name)
