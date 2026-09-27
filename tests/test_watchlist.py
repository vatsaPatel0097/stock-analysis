"""Append-only watchlist snapshots. No network."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import store.watchlist as watchlist_mod
from store.watchlist import add_snapshot, list_snapshots


def _payload(
    ticker: str,
    action: str,
    *,
    current: float = 100.0,
    entry: float = 100.0,
    stop: float = 98.0,
    target: float | None = 106.0,
    sell_by_day: int | None = 10,
    sell_by_date: str | None = "2026-10-08",
) -> dict:
    return {
        "query": ticker.replace(".NS", ""),
        "ticker": ticker,
        "as_of": "2026-09-24T15:30:00+05:30",
        "price": {
            "current": current,
            "day_high": current + 1.0,
            "day_low": current - 1.0,
            "prev_close": current - 0.5,
        },
        "plan": {
            "action": action,
            "entry": entry,
            "stop_loss": stop,
            "stop_pct": 2.0,
            "window_days": 15,
            "recommended_target": target,
            "recommended_target_pct": None if target is None else 6.0,
            "sell_by_day": sell_by_day,
            "sell_by_date": sell_by_date,
        },
        "disclaimer": "Educational research only. Not investment advice.",
    }


class WatchlistTests(unittest.TestCase):
    def test_two_adds_are_two_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            first = _payload("RELIANCE.NS", "BUY")
            second = _payload("INFY.NS", "SKIP", target=None, sell_by_day=None, sell_by_date=None)
            add_snapshot(first, path)
            add_snapshot(second, path)
            rows = list_snapshots(path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["ticker"], "RELIANCE.NS")
        self.assertEqual(rows[0]["action"], "BUY")
        self.assertEqual(rows[0]["price_at_add"], 100.0)
        self.assertEqual(rows[0]["entry"], 100.0)
        self.assertEqual(rows[0]["stop_loss"], 98.0)
        self.assertEqual(rows[0]["recommended_target"], 106.0)
        self.assertEqual(rows[0]["sell_by_day"], 10)
        self.assertEqual(rows[0]["sell_by_date"], "2026-10-08")
        self.assertEqual(rows[0]["payload"]["plan"]["entry"], 100.0)
        self.assertEqual(rows[1]["ticker"], "INFY.NS")
        self.assertEqual(rows[1]["action"], "SKIP")
        self.assertIsNone(rows[1]["recommended_target"])

    def test_same_ticker_twice_is_two_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            first = _payload("RELIANCE.NS", "BUY", current=100.0)
            second = _payload("RELIANCE.NS", "WAIT", current=105.0)
            add_snapshot(first, path)
            add_snapshot(second, path)
            rows = list_snapshots(path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["price_at_add"], 100.0)
        self.assertEqual(rows[1]["price_at_add"], 105.0)
        self.assertEqual(rows[0]["payload"]["price"]["current"], 100.0)
        self.assertEqual(rows[1]["payload"]["price"]["current"], 105.0)

    def test_payload_fields_match_card_at_add(self):
        card = _payload(
            "TCS.NS",
            "BUY",
            current=3200.0,
            entry=3200.0,
            stop=3136.0,
            target=3392.0,
            sell_by_day=12,
            sell_by_date="2026-10-10",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            add_snapshot(card, path)
            row = list_snapshots(path)[0]

        self.assertEqual(row["payload"], card)
        self.assertEqual(row["price_at_add"], 3200.0)
        self.assertEqual(row["entry"], 3200.0)
        self.assertEqual(row["stop_loss"], 3136.0)
        self.assertEqual(row["recommended_target"], 3392.0)
        self.assertEqual(row["sell_by_day"], 12)
        self.assertEqual(row["sell_by_date"], "2026-10-10")

    def test_module_has_no_update_or_delete_api(self):
        names = [name for name in dir(watchlist_mod) if not name.startswith("_")]
        forbidden = [
            name for name in names if "update" in name.lower() or "delete" in name.lower()
        ]
        self.assertEqual(forbidden, [])
        self.assertIn("add_snapshot", names)
        self.assertIn("list_snapshots", names)

    def test_missing_price_current_raises(self):
        bad = _payload("RELIANCE.NS", "BUY")
        del bad["price"]["current"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            with self.assertRaises(ValueError):
                add_snapshot(bad, path)


if __name__ == "__main__":
    unittest.main()
