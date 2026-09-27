"""Watchlist marks on hand-built OHLC fixtures. No network."""

from __future__ import annotations

import copy
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from eval.resolve import resolve_path
from eval.watchlist import list_memory, refresh_marks
from store.watchlist import add_snapshot, list_snapshots


def _frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    index = pd.date_range("2025-01-02", periods=len(rows), freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "Open": [row[0] for row in rows],
            "High": [row[1] for row in rows],
            "Low": [row[2] for row in rows],
            "Close": [row[3] for row in rows],
        },
        index=index,
    )


def _payload(
    *,
    entry: float = 100.0,
    stop: float = 98.0,
    target: float | None = 106.0,
    window: int = 15,
    as_of: str = "2025-01-02T15:30:00+05:30",
    action: str = "BUY",
    ticker: str = "TEST.NS",
    current: float | None = None,
) -> dict:
    price = entry if current is None else current
    return {
        "query": "Test",
        "ticker": ticker,
        "as_of": as_of,
        "price": {
            "current": price,
            "day_high": price + 1.0,
            "day_low": price - 1.0,
            "prev_close": price - 0.5,
        },
        "plan": {
            "action": action,
            "entry": entry,
            "stop_loss": stop,
            "stop_pct": 2.0,
            "window_days": window,
            "recommended_target": target,
            "recommended_target_pct": None if target is None else 6.0,
            "sell_by_day": 10,
            "sell_by_date": "2025-01-16",
        },
        "disclaimer": "Educational research only. Not investment advice.",
    }


class WatchlistMarksTests(unittest.TestCase):
    def test_incomplete_window_has_price_no_outcome(self):
        # Entry bar + 5 forward bars: not enough for a 15-day outcome.
        rows = [(100, 101, 99, 100)]
        for i in range(5):
            close = 102.0 + i
            rows.append((close, close + 1.0, close - 1.0, close))
        frame = _frame(rows)
        payload = _payload()
        original = copy.deepcopy(payload)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            add_snapshot(payload, path)
            refresh_marks(
                path,
                as_of=date(2025, 1, 10),
                price_loader=lambda _ticker: frame,
            )
            memory = list_memory(path)
            snap = list_snapshots(path)[0]

        self.assertEqual(len(memory), 1)
        row = memory[0]
        self.assertEqual(row["price_at_add"], 100.0)
        self.assertEqual(row["latest_price"], 106.0)
        self.assertEqual(row["change_pct"], 6.0)
        self.assertIsNone(row["outcome"])
        self.assertEqual(snap["payload"], original)

    def test_complete_window_stores_target_outcome(self):
        rows = [(100, 101, 99, 100)]
        rows.append((100, 101, 99, 100))
        rows.append((100, 107, 99, 105))
        for _ in range(13):
            rows.append((100, 101, 99, 100))
        frame = _frame(rows)
        payload = _payload()
        expected = resolve_path(payload, frame)
        assert expected is not None

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            add_snapshot(payload, path)
            refresh_marks(
                path,
                as_of=date(2025, 2, 1),
                price_loader=lambda _ticker: frame,
            )
            memory = list_memory(path)

        row = memory[0]
        self.assertEqual(row["outcome"], "target")
        self.assertEqual(row["outcome"], expected["outcome"])
        self.assertEqual(row["exit_day"], expected["exit_day"])
        self.assertEqual(row["fill_price"], expected["fill_price"])
        self.assertAlmostEqual(row["realised_pct"], expected["realised_pct"], places=10)
        self.assertEqual(row["latest_price"], 100.0)
        self.assertEqual(row["change_pct"], 0.0)

    def test_second_refresh_keeps_outcome_updates_price(self):
        rows = [(100, 101, 99, 100)]
        rows.append((100, 107, 99, 105))
        for _ in range(14):
            rows.append((100, 101, 99, 100))
        # Extra bar after the window so a later refresh can show a new close.
        rows.append((110, 111, 109, 110))
        frame = _frame(rows)
        payload = _payload()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            add_snapshot(payload, path)
            refresh_marks(
                path,
                as_of=date(2025, 1, 24),
                price_loader=lambda _ticker: frame,
            )
            first = list_memory(path)[0]
            self.assertEqual(first["outcome"], "target")
            first_outcome = first["outcome"]
            first_exit = first["exit_day"]
            first_fill = first["fill_price"]

            refresh_marks(
                path,
                as_of=date(2025, 1, 27),
                price_loader=lambda _ticker: frame,
            )
            second = list_memory(path)[0]
            snap_payload = list_snapshots(path)[0]["payload"]

        self.assertEqual(second["outcome"], first_outcome)
        self.assertEqual(second["exit_day"], first_exit)
        self.assertEqual(second["fill_price"], first_fill)
        self.assertEqual(second["latest_price"], 110.0)
        self.assertEqual(second["change_pct"], 10.0)
        self.assertEqual(snap_payload, payload)

    def test_list_memory_null_marks_before_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            add_snapshot(_payload(), path)
            memory = list_memory(path)

        self.assertEqual(len(memory), 1)
        self.assertIsNone(memory[0]["latest_price"])
        self.assertIsNone(memory[0]["change_pct"])
        self.assertIsNone(memory[0]["outcome"])
        self.assertEqual(memory[0]["action"], "BUY")


if __name__ == "__main__":
    unittest.main()
