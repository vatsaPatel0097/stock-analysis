"""Price shape, cache, and XNSE calendar tests. Fixtures only; no network."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from data.calendar import CalendarError, is_trading_day, trading_days
from data.prices import PriceError, fetch_daily, normalize_ohlcv

IST = ZoneInfo("Asia/Kolkata")


def _flat_frame() -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17"])
    return pd.DataFrame(
        {
            "Open": [100.5, None, 106.0],
            "High": [110.25, None, 112.5],
            "Low": [99.0, None, 101.25],
            "Close": [105.75, None, 108.5],
            "Adj Close": [105.75, None, 108.5],
            "Volume": [1000, None, 1500],
        },
        index=index,
    )


def _column_grouped_frame() -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["RELIANCE.NS", "TCS.NS"]],
        names=["Price", "Ticker"],
    )
    index = pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17"])
    rows = [
        [100.5, 999.0, 110.25, 1000.0, 99.0, 990.0, 105.75, 888.0, 1000, 5000],
        [None, 201.0, None, 211.0, None, 191.0, None, 206.0, None, 2100],
        [106.0, 202.0, 112.5, 212.0, 101.25, 192.0, 108.5, 208.0, 1500, 2200],
    ]
    return pd.DataFrame(rows, index=index, columns=columns)


def _ticker_grouped_frame() -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-15", "2024-01-17"])
    data = {
        ("RELIANCE.NS", "Open"): [100.5, 106.0],
        ("RELIANCE.NS", "High"): [110.25, 112.5],
        ("RELIANCE.NS", "Low"): [99.0, 101.25],
        ("RELIANCE.NS", "Close"): [105.75, 108.5],
        ("RELIANCE.NS", "Volume"): [1000, 1500],
        ("TCS.NS", "Close"): [888.0, 208.0],
        ("TCS.NS", "Open"): [999.0, 202.0],
        ("TCS.NS", "High"): [1000.0, 212.0],
        ("TCS.NS", "Low"): [990.0, 192.0],
        ("TCS.NS", "Volume"): [5000, 2200],
    }
    frame = pd.DataFrame(data, index=index)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["Ticker", "Price"])
    return frame


class NormalizeTests(unittest.TestCase):
    def test_flat_frame_drops_holiday_and_keeps_ohlcv(self):
        out = normalize_ohlcv(_flat_frame(), "RELIANCE.NS")

        self.assertEqual(list(out.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(len(out), 2)
        self.assertEqual(out["Close"].tolist(), [105.75, 108.5])
        self.assertEqual(out["High"].iloc[0], 110.25)
        self.assertEqual(out["Low"].iloc[1], 101.25)
        self.assertEqual(out["Volume"].tolist(), [1000.0, 1500.0])
        self.assertEqual(str(out.index.tz), "UTC")
        self.assertEqual(out.index[0].date(), date(2024, 1, 15))
        self.assertEqual(out.index[1].date(), date(2024, 1, 17))

    def test_one_ticker_multiindex_matches_flat_values(self):
        flat = _flat_frame().drop(columns=["Adj Close"])
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["RELIANCE.NS"]]
        )
        wide = flat.copy()
        wide.columns = columns

        out = normalize_ohlcv(wide, "reliance.ns")

        self.assertEqual(out["Close"].tolist(), [105.75, 108.5])
        self.assertEqual(out["Open"].tolist(), [100.5, 106.0])

    def test_many_tickers_column_grouped_extracts_requested_symbol(self):
        out = normalize_ohlcv(_column_grouped_frame(), "RELIANCE.NS")

        self.assertEqual(out["Close"].tolist(), [105.75, 108.5])
        self.assertNotIn(888.0, out["Close"].tolist())
        self.assertNotIn(999.0, out["Open"].tolist())
        self.assertEqual(len(out), 2)

    def test_many_tickers_ticker_grouped_extracts_requested_symbol(self):
        out = normalize_ohlcv(_ticker_grouped_frame(), "RELIANCE.NS")

        self.assertEqual(out["Close"].tolist(), [105.75, 108.5])
        self.assertEqual(out["Volume"].tolist(), [1000.0, 1500.0])

    def test_ist_timestamp_stores_that_session_date_in_utc(self):
        index = pd.to_datetime(["2024-01-15 15:30"]).tz_localize(IST)
        frame = pd.DataFrame(
            {"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5], "Volume": [50]},
            index=index,
        )

        out = normalize_ohlcv(frame, "TCS.NS")

        self.assertEqual(out.index[0], pd.Timestamp("2024-01-15", tz="UTC"))
        self.assertEqual(out["Close"].iloc[0], 10.5)

    def test_rejects_non_table_missing_columns_and_unknown_ticker(self):
        with self.assertRaises(PriceError):
            normalize_ohlcv([[1, 2, 3]], "RELIANCE.NS")  # type: ignore[arg-type]
        with self.assertRaises(PriceError):
            normalize_ohlcv(pd.DataFrame({"Close": [1.0]}), "RELIANCE.NS")
        with self.assertRaises(PriceError):
            normalize_ohlcv(_column_grouped_frame(), "INFY.NS")
        with self.assertRaises(PriceError):
            normalize_ohlcv(_flat_frame(), "RELIANCE")

    def test_all_nan_rows_raise(self):
        frame = _flat_frame().iloc[1:2]
        with self.assertRaises(PriceError):
            normalize_ohlcv(frame, "RELIANCE.NS")


class FetchCacheTests(unittest.TestCase):
    def test_second_call_same_day_reads_parquet_and_skips_downloader(self):
        calls = {"n": 0}

        def downloader(ticker: str, period: str) -> pd.DataFrame:
            calls["n"] += 1
            self.assertEqual(ticker, "RELIANCE.NS")
            self.assertEqual(period, "5y")
            return _flat_frame()

        with tempfile.TemporaryDirectory() as tmp:
            kwargs = {
                "as_of": date(2026, 9, 27),
                "cache_dir": Path(tmp),
                "downloader": downloader,
            }
            first = fetch_daily("RELIANCE.NS", **kwargs)
            cached = next(Path(tmp).rglob("*.parquet"))
            self.assertTrue(cached.is_file())
            self.assertEqual(cached.parent.name, "RELIANCE.NS")
            self.assertEqual(cached.name, "2026-09-27.parquet")

            def fail_if_called(ticker: str, period: str) -> pd.DataFrame:
                raise AssertionError("cache hit must not download")

            second = fetch_daily(
                "RELIANCE.NS",
                as_of=date(2026, 9, 27),
                cache_dir=Path(tmp),
                downloader=fail_if_called,
            )

        self.assertEqual(calls["n"], 1)
        self.assertEqual(first["Close"].tolist(), second["Close"].tolist())
        self.assertEqual(str(second.index.tz), "UTC")

    def test_next_day_fetches_again(self):
        calls = {"n": 0}

        def downloader(ticker: str, period: str) -> pd.DataFrame:
            calls["n"] += 1
            return _flat_frame()

        with tempfile.TemporaryDirectory() as tmp:
            fetch_daily("TCS.NS", as_of=date(2026, 9, 27), cache_dir=tmp, downloader=downloader)
            fetch_daily("TCS.NS", as_of=date(2026, 9, 28), cache_dir=tmp, downloader=downloader)

        self.assertEqual(calls["n"], 2)

    def test_default_downloader_requests_adjusted_daily_bars(self):
        import yfinance as yf

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(yf, "download", return_value=_flat_frame()) as download:
                fetch_daily("RELIANCE.NS", as_of=date(2026, 9, 27), cache_dir=tmp)
        kwargs = download.call_args.kwargs
        self.assertTrue(kwargs["auto_adjust"])
        self.assertTrue(kwargs["keepna"])
        self.assertEqual(kwargs["interval"], "1d")
        self.assertEqual(kwargs["tickers"], "RELIANCE.NS")


class CalendarTests(unittest.TestCase):
    def test_republic_day_week_uses_exchange_calendar(self):
        # 2024-01-26 is Republic Day (Friday). 27–28 are the weekend.
        days = trading_days("2024-01-25", "2024-01-29")

        self.assertEqual(
            [stamp.date() for stamp in days],
            [date(2024, 1, 25), date(2024, 1, 29)],
        )
        self.assertEqual(str(days.tz), "UTC")
        self.assertFalse(is_trading_day(date(2024, 1, 26)))
        self.assertFalse(is_trading_day("2024-01-27"))
        self.assertTrue(is_trading_day(date(2024, 1, 29)))

    def test_end_before_start_raises(self):
        with self.assertRaises(CalendarError):
            trading_days(date(2024, 1, 29), date(2024, 1, 25))


if __name__ == "__main__":
    unittest.main()
