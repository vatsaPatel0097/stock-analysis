"""analyze() against fixture prices and a fake resolver. No network."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from analyze import AmbiguousQuery, _SECTION3_KEYS, analyze, format_card, format_pick_list
from data.prices import PriceError
from data.resolve import Candidate, ResolveError
from store.calls import list_calls

_AS_OF = date(2026, 9, 24)


def _forbid_search(query: str):
    raise AssertionError(f"search must not run for {query!r}")


def _ohlcv(n: int, *, start: float = 100.0, step: float = 0.25, spread: float = 0.01, volume: float = 2_000_000.0) -> pd.DataFrame:
    """Gentle climb with a controlled daily range and high ADV."""
    rows = []
    close = start
    for i in range(n):
        high = close * (1.0 + spread)
        low = close * (1.0 - spread)
        rows.append((close, high, low, volume))
        close = close + step
    index = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "Open": [row[0] for row in rows],
            "High": [row[1] for row in rows],
            "Low": [row[2] for row in rows],
            "Close": [row[0] for row in rows],
            "Volume": [row[3] for row in rows],
        },
        index=index,
    )


def _loader(frame: pd.DataFrame):
    def load(ticker: str, *, as_of=None):
        return frame.copy()

    return load


class AnalyzeBuyTests(unittest.TestCase):
    def test_buy_card_has_section3_keys_and_null_llm_fields(self):
        frame = _ohlcv(80)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            payload = analyze(
                "RELIANCE",
                as_of=_AS_OF,
                price_loader=_loader(frame),
                searcher=_forbid_search,
                calls_path=path,
            )

        self.assertEqual(list(payload.keys()), list(_SECTION3_KEYS))
        self.assertEqual(payload["ticker"], "RELIANCE.NS")
        self.assertIsNone(payload["news"])
        self.assertIsNone(payload["explanation"])
        self.assertEqual(payload["plan"]["action"], "BUY")
        self.assertIsNone(payload["skip_reason"])
        entry = payload["plan"]["entry"]
        self.assertEqual(payload["price"]["current"], entry)
        self.assertEqual(payload["plan"]["stop_loss"], round(entry * 0.98, 2))
        self.assertEqual(payload["plan"]["stop_pct"], 2.0)
        self.assertGreaterEqual(payload["confidence_basis"]["sample_size"], 50)
        self.assertIsNotNone(payload["plan"]["recommended_target"])
        self.assertIsNotNone(payload["plan"]["sell_by_day"])
        self.assertIsNotNone(payload["plan"]["sell_by_date"])
        self.assertEqual(len(payload["targets"]), 5)
        card = format_card(payload)
        self.assertIn("BUY", card)
        self.assertIn("Educational research only", card)
        self.assertIn("gap-down", card)


class AnalyzeSkipTests(unittest.TestCase):
    def test_short_history_skips_on_sample_size(self):
        frame = _ohlcv(30)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            payload = analyze(
                "INFY",
                as_of=_AS_OF,
                price_loader=_loader(frame),
                searcher=_forbid_search,
                calls_path=path,
            )
        self.assertEqual(payload["plan"]["action"], "SKIP")
        self.assertIn("50", payload["skip_reason"])
        self.assertEqual(payload["price"]["current"], round(float(frame["Close"].iloc[-1]), 2))


class AnalyzeTargetTests(unittest.TestCase):
    def test_rupee_target_becomes_user_target_percent(self):
        frame = _ohlcv(80)
        entry = float(frame["Close"].iloc[-1])
        wished = round(entry * 1.08, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            payload = analyze(
                "RELIANCE",
                user_target=wished,
                as_of=_AS_OF,
                price_loader=_loader(frame),
                searcher=_forbid_search,
                calls_path=path,
            )
        self.assertIsNotNone(payload["user_target"])
        self.assertEqual(payload["user_target"]["price"], wished)
        self.assertAlmostEqual(payload["user_target"]["pct"], 8.0, places=1)
        self.assertEqual(len(payload["targets"]), 5)


class AnalyzeErrorTests(unittest.TestCase):
    def test_ambiguous_name_raises_and_does_not_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            with self.assertRaises(AmbiguousQuery) as raised:
                analyze(
                    "Tata",
                    as_of=_AS_OF,
                    price_loader=_loader(_ohlcv(80)),
                    searcher=_forbid_search,
                    calls_path=path,
                )
            self.assertGreaterEqual(len(raised.exception.matches), 2)
            self.assertEqual(list_calls(path), [])

    def test_unknown_name_raises(self):
        def empty_search(query: str):
            return []

        with self.assertRaises(ResolveError):
            analyze(
                "DefinitelyNotAStockXYZ",
                as_of=_AS_OF,
                price_loader=_loader(_ohlcv(80)),
                searcher=empty_search,
            )

    def test_missing_prices_raise(self):
        def bad_loader(ticker: str, *, as_of=None):
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        with self.assertRaises(PriceError):
            analyze(
                "RELIANCE",
                as_of=_AS_OF,
                price_loader=bad_loader,
                searcher=_forbid_search,
            )


class AnalyzeLogTests(unittest.TestCase):
    def test_two_analyzes_append_two_rows(self):
        frame = _ohlcv(80)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            first = analyze(
                "RELIANCE",
                as_of=_AS_OF,
                price_loader=_loader(frame),
                searcher=_forbid_search,
                calls_path=path,
            )
            second = analyze(
                "INFY",
                as_of=_AS_OF,
                price_loader=_loader(frame),
                searcher=_forbid_search,
                calls_path=path,
            )
            rows = list_calls(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["ticker"], "RELIANCE.NS")
        self.assertEqual(rows[1]["ticker"], "INFY.NS")
        self.assertEqual(rows[0]["payload"]["plan"]["action"], first["plan"]["action"])
        self.assertEqual(rows[1]["payload"]["plan"]["entry"], second["plan"]["entry"])


class CliFormatTests(unittest.TestCase):
    def test_pick_list_text_lists_tickers(self):
        text = format_pick_list(
            "Tata",
            (
                Candidate("Tata Steel", "TATASTEEL.NS"),
                Candidate("Tata Power", "TATAPOWER.NS"),
            ),
        )
        self.assertIn("Several matches", text)
        self.assertIn("TATASTEEL.NS", text)
        self.assertIn("TATAPOWER.NS", text)


if __name__ == "__main__":
    unittest.main()
