"""Latest-bar trend, RSI bucket, and ATR percent. No network.

Trend compares the last close with the 50-day and 200-day simple averages.
ATR percent is percentage points (1.52 means 1.52% of price). See DECISIONS.md.
"""

from __future__ import annotations

import pandas as pd

from analytics.indicators import atr, rsi

FAST_DMA = 50
SLOW_DMA = 200
_ATR_PERIOD = 14
_RSI_PERIOD = 14


def regime(frame: pd.DataFrame) -> dict:
    """Classify the last row of ``frame``."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Regime needs a price table.")
    if frame.empty:
        return _blank()
    averaged = atr(frame, _ATR_PERIOD)
    strength = rsi(frame, _RSI_PERIOD)
    close = pd.to_numeric(frame["Close"], errors="coerce")
    sma_fast = close.rolling(FAST_DMA, min_periods=FAST_DMA).mean()
    sma_slow = close.rolling(SLOW_DMA, min_periods=SLOW_DMA).mean()

    last_close = float(close.iloc[-1])
    fast = _last_number(sma_fast)
    slow = _last_number(sma_slow)
    above_fast = None if fast is None else last_close > fast
    above_slow = None if slow is None else last_close > slow
    below_fast = None if fast is None else last_close < fast
    below_slow = None if slow is None else last_close < slow
    last_atr = _last_number(averaged)
    last_rsi = _last_number(strength)
    atr_pct = None if last_atr is None else (last_atr / last_close) * 100.0
    atr_pct_series = (averaged / close) * 100.0
    percentile = _atr_percentile(atr_pct_series)

    return {
        "trend": _trend(above_fast, above_slow, below_fast, below_slow),
        "above_50dma": above_fast,
        "above_200dma": above_slow,
        "sma50": fast,
        "sma200": slow,
        "rsi14": last_rsi,
        "rsi_bucket": None if last_rsi is None else rsi_bucket(last_rsi),
        "atr": last_atr,
        "atr_pct": atr_pct,
        "atr_percentile": percentile,
        "atr_bucket": volatility_bucket(percentile),
    }


def rsi_bucket(value: float) -> str:
    """Map RSI to <30, 30–50, 50–70, or >70.

    30 belongs to 30–50. 50 and 70 belong to 50–70.
    """
    if value < 30.0:
        return "below_30"
    if value < 50.0:
        return "30_50"
    if value <= 70.0:
        return "50_70"
    return "above_70"


def volatility_bucket(percentile: float | None) -> str | None:
    """Tercile of an ATR% percentile in the 0–100 range."""
    if percentile is None:
        return None
    if percentile < 100.0 / 3.0:
        return "low"
    if percentile < 200.0 / 3.0:
        return "mid"
    return "high"


def _trend(
    above_fast: bool | None,
    above_slow: bool | None,
    below_fast: bool | None,
    below_slow: bool | None,
) -> str | None:
    if above_fast is None or above_slow is None:
        return None
    if above_fast and above_slow:
        return "up"
    if below_fast and below_slow:
        return "down"
    return "range"


def _atr_percentile(atr_pct: pd.Series) -> float | None:
    """Percent of other ATR% readings strictly below the latest one."""
    clean = atr_pct.dropna()
    if len(clean) < 2:
        return None
    latest = float(clean.iloc[-1])
    below = float((clean < latest).sum())
    return 100.0 * below / (len(clean) - 1)


def _last_number(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = series.iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def _blank() -> dict:
    return {
        "trend": None,
        "above_50dma": None,
        "above_200dma": None,
        "sma50": None,
        "sma200": None,
        "rsi14": None,
        "rsi_bucket": None,
        "atr": None,
        "atr_pct": None,
        "atr_percentile": None,
        "atr_bucket": None,
    }
