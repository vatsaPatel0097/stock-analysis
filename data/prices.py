"""Daily NSE prices from yfinance, with a parquet cache. No analytics."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
OHLCV = ("Open", "High", "Low", "Close", "Volume")
_FIELD_NAMES = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adjclose": "Adj Close",
    "volume": "Volume",
}


class PriceError(Exception):
    """Price data was missing or had an unexpected shape."""


def fetch_daily(
    ticker: str,
    *,
    period: str = "5y",
    as_of: date | None = None,
    cache_dir: Path | str | None = None,
    downloader=None,
) -> pd.DataFrame:
    """Return daily OHLCV for one ``SYMBOL.NS`` ticker.

    The first successful fetch on a given IST date is stored as parquet and
    reused for the rest of that date. ``downloader`` defaults to yfinance
    with ``auto_adjust=True``. Tests pass a fixture callable instead.
    """
    symbol = _require_nse_ticker(ticker)
    day = _as_of_date(as_of)
    path = _cache_path(cache_dir, symbol, day)
    if path.is_file():
        return _read_cache(path)

    source = downloader if downloader is not None else download_daily
    raw = source(symbol, period)
    frame = normalize_ohlcv(raw, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame


def download_daily(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download one ticker from yfinance. Not used by tests."""
    import yfinance as yf

    symbol = _require_nse_ticker(ticker)
    try:
        frame = yf.download(
            tickers=symbol,
            period=period,
            interval="1d",
            auto_adjust=True,
            group_by="column",
            progress=False,
            threads=False,
            keepna=True,
            multi_level_index=True,
        )
    except Exception as exc:
        raise PriceError(f"Could not download prices for {symbol}.") from exc
    if not isinstance(frame, pd.DataFrame):
        raise PriceError(f"Could not download prices for {symbol}.")
    return frame


def normalize_ohlcv(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Turn a one-ticker or many-ticker yfinance frame into OHLCV.

    Holiday rows (NaN prices) are dropped. The index is the NSE session
    date stored as UTC midnight.
    """
    symbol = _require_nse_ticker(ticker)
    if not isinstance(frame, pd.DataFrame):
        raise PriceError("Price data must be a table.")
    extracted = _extract_ticker(frame, symbol)
    renamed = _rename_columns(extracted)
    missing = [name for name in OHLCV if name not in renamed.columns]
    if missing:
        raise PriceError(f"Price table is missing {', '.join(missing)}.")

    out = renamed.loc[:, list(OHLCV)].copy()
    for column in OHLCV:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"], how="any")
    out["Volume"] = out["Volume"].fillna(0)
    if out.empty:
        raise PriceError(f"No daily prices for {symbol}.")

    out.index = _utc_session_index(out.index)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    out.index.name = "Date"
    return out


def _extract_ticker(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    columns = frame.columns
    if not isinstance(columns, pd.MultiIndex):
        return frame
    if columns.nlevels != 2:
        raise PriceError("Price table has an unexpected column layout.")

    level0 = _field_map(columns.get_level_values(0))
    level1 = _field_map(columns.get_level_values(1))
    if _has_ohlc(level0):
        field_level, ticker_level = 0, 1
    elif _has_ohlc(level1):
        field_level, ticker_level = 1, 0
    else:
        raise PriceError("Price table is missing Open and Close columns.")

    tickers = list(dict.fromkeys(columns.get_level_values(ticker_level)))
    match = _match_ticker(tickers, ticker)
    return frame.xs(match, axis=1, level=ticker_level)


def _rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for column in frame.columns:
        field = _canon_field(column)
        if field is not None and field not in mapping.values():
            mapping[column] = field
    renamed = frame.rename(columns=mapping)
    if "Close" not in renamed.columns and "Adj Close" in renamed.columns:
        renamed = renamed.rename(columns={"Adj Close": "Close"})
    if "Adj Close" in renamed.columns:
        renamed = renamed.drop(columns=["Adj Close"])
    return renamed


def _field_map(values) -> dict[object, str]:
    mapping = {}
    for value in dict.fromkeys(values):
        field = _canon_field(value)
        if field is not None:
            mapping[value] = field
    return mapping


def _canon_field(label: object) -> str | None:
    key = "".join(ch for ch in str(label).strip().lower() if ch.isalnum())
    return _FIELD_NAMES.get(key)


def _has_ohlc(mapping: dict[object, str]) -> bool:
    fields = set(mapping.values())
    return "Open" in fields and "Close" in fields


def _match_ticker(tickers: list[object], ticker: str) -> object:
    wanted = ticker.upper()
    for candidate in tickers:
        if str(candidate).strip().upper() == wanted:
            return candidate
    raise PriceError(f"No price columns for {ticker}.")


def _utc_session_index(index) -> pd.DatetimeIndex:
    parsed = pd.DatetimeIndex(pd.to_datetime(index))
    if parsed.tz is None:
        session_dates = parsed.normalize()
    else:
        local = parsed.tz_convert(IST).tz_localize(None)
        session_dates = pd.DatetimeIndex(local).normalize()
    return session_dates.tz_localize("UTC")


def _require_nse_ticker(ticker: object) -> str:
    if not isinstance(ticker, str):
        raise PriceError("Ticker must be an NSE symbol ending in .NS.")
    symbol = ticker.strip().upper()
    if not symbol.endswith(".NS") or symbol == ".NS":
        raise PriceError("Ticker must be an NSE symbol ending in .NS.")
    return symbol


def _as_of_date(value: date | None) -> date:
    if value is None:
        return datetime.now(IST).date()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.date()
        return value.astimezone(IST).date()
    if isinstance(value, date):
        return value
    raise PriceError("as_of must be a date.")


def _cache_path(cache_dir: Path | str | None, ticker: str, day: date) -> Path:
    root = Path(cache_dir) if cache_dir is not None else Path("cache") / "prices"
    safe = "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in ticker)
    return root / safe / f"{day.isoformat()}.parquet"


def _read_cache(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not isinstance(frame, pd.DataFrame):
        raise PriceError("Cached prices are not a table.")
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame.index.name = "Date"
    return frame
