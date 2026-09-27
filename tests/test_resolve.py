"""Name and ticker resolution. Fixtures only; no network."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from data.resolve import (
    NSE_LISTINGS,
    Candidate,
    Listing,
    ResolveError,
    candidates_from_quotes,
    resolve,
)

_TATA = {
    "TATACHEM.NS",
    "TATACONSUM.NS",
    "TATAPOWER.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TMCV.NS",
    "TMPV.NS",
}


def _forbid(query: str):
    raise AssertionError(f"search should not run for {query}")


class MapTests(unittest.TestCase):
    def test_listings_are_unique_nse_symbols(self):
        tickers = [row.ticker for row in NSE_LISTINGS]
        self.assertEqual(len(tickers), len(set(tickers)))
        self.assertNotIn("TATAMOTORS.NS", tickers)
        for row in NSE_LISTINGS:
            self.assertTrue(row.name.strip())
            self.assertTrue(row.ticker.endswith(".NS"))
            self.assertGreater(len(row.ticker), 3)


class LocalResolveTests(unittest.TestCase):
    def test_alias_symbol_and_suffix_are_reliance(self):
        for query in ("Reliance", "RIL", "RELIANCE", "reliance.ns", "  RELIANCE.NS  "):
            result = resolve(query, searcher=_forbid)
            self.assertEqual(result.ticker, "RELIANCE.NS", query)
            self.assertEqual(result.query, query.strip())
            self.assertEqual(len(result.matches), 1)
            self.assertEqual(result.matches[0].name, "Reliance Industries")
            self.assertEqual(result.matches[0].ticker, "RELIANCE.NS")

    def test_full_name_is_unique(self):
        result = resolve("Reliance Industries", searcher=_forbid)
        self.assertEqual(result.ticker, "RELIANCE.NS")

    def test_exact_alias_beats_a_longer_prefix(self):
        result = resolve("SBI", searcher=_forbid)
        self.assertEqual(result.ticker, "SBIN.NS")
        self.assertEqual(result.matches[0].name, "State Bank of India")
        self.assertEqual(len(result.matches), 1)

    def test_tata_is_a_pick_list(self):
        result = resolve("Tata", searcher=_forbid)
        self.assertIsNone(result.ticker)
        self.assertEqual({match.ticker for match in result.matches}, _TATA)
        self.assertEqual(
            [match.ticker for match in result.matches],
            sorted(match.ticker for match in result.matches),
        )
        for match in result.matches:
            self.assertTrue(match.name)
            self.assertTrue(match.ticker.endswith(".NS"))

    def test_tata_motors_does_not_pick_one_company(self):
        result = resolve("Tata Motors", searcher=_forbid)
        self.assertIsNone(result.ticker)
        self.assertEqual(
            {match.ticker for match in result.matches},
            {"TMCV.NS", "TMPV.NS"},
        )

    def test_shared_prefix_stays_ambiguous(self):
        cases = {
            "HDFC": {"HDFCBANK.NS", "HDFCLIFE.NS"},
            "Tata Cons": {"TCS.NS", "TATACONSUM.NS"},
            "Mahindra": {"M&M.NS", "KOTAKBANK.NS"},
            "Bajaj": {"BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS"},
        }
        for query, expected in cases.items():
            result = resolve(query, searcher=_forbid)
            self.assertIsNone(result.ticker, query)
            self.assertEqual({match.ticker for match in result.matches}, expected, query)

    def test_tied_alias_returns_a_pick_list(self):
        symbols = (
            Listing("Tata Motors Limited", "TMCV.NS", ("Tata",)),
            Listing("Tata Steel", "TATASTEEL.NS", ("Tata",)),
        )
        result = resolve("tata", symbols=symbols, searcher=_forbid)
        self.assertIsNone(result.ticker)
        self.assertEqual(
            [match.ticker for match in result.matches],
            ["TATASTEEL.NS", "TMCV.NS"],
        )

    def test_ampersand_alias_and_suffix(self):
        self.assertEqual(resolve("L&T", searcher=_forbid).ticker, "LT.NS")
        self.assertEqual(resolve("M&M.NS", searcher=_forbid).ticker, "M&M.NS")
        self.assertEqual(resolve("BAJAJ-AUTO", searcher=_forbid).ticker, "BAJAJ-AUTO.NS")

    def test_local_hit_does_not_write_a_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolve("Reliance", searcher=_forbid, cache_dir=tmp)
            self.assertEqual(list(Path(tmp).rglob("*")), [])

    def test_blank_query_raises_and_does_not_search(self):
        calls = []

        def searcher(query: str):
            calls.append(query)
            return []

        for bad in ("", "   ", None, 12):
            with self.assertRaises(ResolveError) as caught:
                resolve(bad, searcher=searcher)  # type: ignore[arg-type]
            self.assertEqual(str(caught.exception), "Type a stock name or ticker.")
        self.assertEqual(calls, [])


class SearchFallbackTests(unittest.TestCase):
    def test_unknown_name_raises_without_a_ticker(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ResolveError) as caught:
                resolve(
                    "Not A Real Company",
                    searcher=lambda query: [],
                    cache_dir=tmp,
                    as_of=date(2026, 9, 27),
                )
        message = str(caught.exception)
        self.assertIn("Not A Real Company", message)
        self.assertNotIn(".NS", message)

    def test_unique_search_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = resolve(
                "Zydus",
                symbols=(),
                searcher=lambda query: [Candidate("Zydus Lifesciences", "ZYDUSLIFE.NS")],
                cache_dir=tmp,
                as_of=date(2026, 9, 27),
            )
        self.assertEqual(result.ticker, "ZYDUSLIFE.NS")
        self.assertEqual(result.matches[0].name, "Zydus Lifesciences")

    def test_two_search_hits_are_a_pick_list(self):
        found = [
            Candidate("Zinc Co", "ZINC.NS"),
            Candidate("Hindustan Zinc", "HINDZINC.NS"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result = resolve(
                "Zinc",
                symbols=(),
                searcher=lambda query: found,
                cache_dir=tmp,
            )
        self.assertIsNone(result.ticker)
        self.assertEqual(
            [match.ticker for match in result.matches],
            ["HINDZINC.NS", "ZINC.NS"],
        )

    def test_non_nse_hit_is_not_a_ticker(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ResolveError) as caught:
                resolve(
                    "Foo",
                    symbols=(),
                    searcher=lambda query: [Candidate("Foo", "FOO.BO")],
                    cache_dir=tmp,
                )
        self.assertNotIn("FOO.BO", str(caught.exception))
        self.assertNotIn(".NS", str(caught.exception))

    def test_typed_suffix_is_not_replaced_with_another_company(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ResolveError) as caught:
                resolve(
                    "ZZZZ.NS",
                    symbols=(),
                    searcher=lambda query: [Candidate("Real Co", "REAL.NS")],
                    cache_dir=tmp,
                )
        self.assertNotIn("REAL.NS", str(caught.exception))

    def test_typed_suffix_accepts_that_ticker(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = resolve(
                "zyduslife.ns",
                symbols=(),
                searcher=lambda query: [Candidate("Zydus Lifesciences", "ZYDUSLIFE.NS")],
                cache_dir=tmp,
            )
        self.assertEqual(result.ticker, "ZYDUSLIFE.NS")

    def test_second_lookup_same_day_does_not_search_again(self):
        calls = {"n": 0}

        def searcher(query: str):
            calls["n"] += 1
            self.assertEqual(query, "Zydus")
            return [Candidate("Zydus Lifesciences", "ZYDUSLIFE.NS")]

        with tempfile.TemporaryDirectory() as tmp:
            kwargs = {
                "symbols": (),
                "cache_dir": tmp,
                "as_of": date(2026, 9, 27),
            }
            first = resolve("Zydus", searcher=searcher, **kwargs)
            cached = next(Path(tmp).rglob("*.json"))
            self.assertEqual(cached.parent.name, "ZYDUS")
            self.assertEqual(cached.name, "2026-09-27.json")
            second = resolve("  zydus  ", searcher=_forbid, **kwargs)

        self.assertEqual(calls["n"], 1)
        self.assertEqual(first.ticker, second.ticker)
        self.assertEqual(second.query, "zydus")

    def test_unknown_lookup_is_cached(self):
        calls = {"n": 0}

        def searcher(query: str):
            calls["n"] += 1
            return []

        with tempfile.TemporaryDirectory() as tmp:
            kwargs = {"symbols": (), "cache_dir": tmp, "as_of": date(2026, 9, 27)}
            with self.assertRaises(ResolveError):
                resolve("Not A Real Company", searcher=searcher, **kwargs)
            with self.assertRaises(ResolveError):
                resolve("Not A Real Company", searcher=_forbid, **kwargs)
        self.assertEqual(calls["n"], 1)

    def test_next_day_searches_again(self):
        calls = {"n": 0}

        def searcher(query: str):
            calls["n"] += 1
            return [Candidate("Zydus Lifesciences", "ZYDUSLIFE.NS")]

        with tempfile.TemporaryDirectory() as tmp:
            resolve("Zydus", symbols=(), searcher=searcher, cache_dir=tmp, as_of=date(2026, 9, 27))
            resolve("Zydus", symbols=(), searcher=searcher, cache_dir=tmp, as_of=date(2026, 9, 28))
        self.assertEqual(calls["n"], 2)

    def test_search_failure_stays_a_resolve_error(self):
        def searcher(query: str):
            raise OSError("offline")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ResolveError) as caught:
                resolve("Zydus", symbols=(), searcher=searcher, cache_dir=tmp)
        self.assertNotIn("offline", str(caught.exception))
        self.assertNotIn("Traceback", str(caught.exception))
        self.assertIn("look up", str(caught.exception))

    def test_corrupt_cache_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolve(
                "Zydus",
                symbols=(),
                searcher=lambda query: [Candidate("Zydus Lifesciences", "ZYDUSLIFE.NS")],
                cache_dir=tmp,
                as_of=date(2026, 9, 27),
            )
            cached = next(Path(tmp).rglob("*.json"))
            cached.write_text("{", encoding="utf-8")
            with self.assertRaises(ResolveError) as caught:
                resolve(
                    "Zydus",
                    symbols=(),
                    searcher=_forbid,
                    cache_dir=tmp,
                    as_of=date(2026, 9, 27),
                )
        self.assertIn("unreadable", str(caught.exception))

    def test_default_search_keeps_nse_equities(self):
        import yfinance as yf

        quotes = [
            {
                "symbol": "ZYDUSLIFE.NS",
                "longname": "Zydus Lifesciences Limited",
                "quoteType": "EQUITY",
                "exchange": "NSI",
            },
            {
                "symbol": "ZYDUSLIFE.BO",
                "shortname": "Zydus BSE",
                "quoteType": "EQUITY",
                "exchange": "BSE",
            },
            {
                "symbol": "NIFTYBEES.NS",
                "shortname": "Nippon ETF",
                "quoteType": "ETF",
            },
        ]

        class _Quotes:
            def __init__(self):
                self.quotes = quotes

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(yf, "Search", return_value=_Quotes()) as search:
                result = resolve(
                    "Zydus Life",
                    symbols=(),
                    cache_dir=tmp,
                    as_of=date(2026, 9, 27),
                )
        self.assertEqual(result.ticker, "ZYDUSLIFE.NS")
        self.assertEqual(result.matches[0].name, "Zydus Lifesciences Limited")
        args, kwargs = search.call_args
        self.assertEqual(args[0], "Zydus Life")
        self.assertEqual(kwargs["max_results"], 8)
        self.assertEqual(kwargs["news_count"], 0)
        self.assertFalse(kwargs["enable_fuzzy_query"])


class QuoteFilterTests(unittest.TestCase):
    def test_quotes_keep_nse_equities_only(self):
        quotes = [
            {
                "symbol": "RELIANCE.NS",
                "longname": "Reliance Industries Limited",
                "quoteType": "EQUITY",
            },
            {"symbol": "RELIANCE.BO", "shortname": "Reliance BSE", "quoteType": "EQUITY"},
            {
                "symbol": "INFY",
                "longname": "Infosys Limited",
                "quoteType": "EQUITY",
                "exchange": "NSI",
            },
            {"symbol": "RELIANCE.NS", "shortname": "Duplicate", "quoteType": "EQUITY"},
            {"symbol": "NIFTYBEES.NS", "quoteType": "ETF", "shortname": "Nippon ETF"},
            {"symbol": "^NSEI", "quoteType": "INDEX", "shortname": "Nifty 50"},
            "bad",
        ]
        found = candidates_from_quotes(quotes)
        self.assertEqual(
            [(item.ticker, item.name) for item in found],
            [
                ("RELIANCE.NS", "Reliance Industries Limited"),
                ("INFY.NS", "Infosys Limited"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
