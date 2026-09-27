"""Outcome resolver on hand-built OHLC fixtures. No network."""

from __future__ import annotations

import copy
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from eval.resolve import list_outcomes, resolve_due, resolve_path
from store.calls import append_call, list_calls


def _frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """rows are (Open, High, Low, Close)."""
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
) -> dict:
    return {
        "query": "Test",
        "ticker": ticker,
        "as_of": as_of,
        "plan": {
            "action": action,
            "entry": entry,
            "stop_loss": stop,
            "stop_pct": 2.0,
            "window_days": window,
            "recommended_target": target,
            "recommended_target_pct": None if target is None else 6.0,
            "sell_by_day": 10,
            "sell_by_date": None,
        },
        "targets": [
            {
                "price": target or entry,
                "pct": 6.0,
                "prob": 0.58,
                "days_median": 7,
                "days_p25_p75": [5, 10],
                "verdict": "recommended",
            }
        ],
        "user_target": None,
        "disclaimer": "Educational research only. Not investment advice.",
    }


class ResolvePathTests(unittest.TestCase):
    def test_target_first(self):
        # Entry on 2025-01-02 (index 0). Day 1: quiet. Day 2: high hits 106.
        rows = [(100, 101, 99, 100)]  # entry bar
        rows.append((100, 101, 99, 100))  # day 1
        rows.append((100, 107, 99, 105))  # day 2 target
        for _ in range(13):
            rows.append((100, 101, 99, 100))
        result = resolve_path(_payload(window=15), _frame(rows))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["outcome"], "target")
        self.assertEqual(result["exit_day"], 2)
        self.assertEqual(result["fill_price"], 106.0)
        self.assertAlmostEqual(result["realised_pct"], 0.06, places=10)

    def test_stop_first(self):
        rows = [(100, 101, 99, 100)]
        rows.append((100, 101, 97, 99))  # day 1 stop
        for _ in range(14):
            rows.append((100, 110, 99, 105))
        result = resolve_path(_payload(window=15), _frame(rows))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["outcome"], "stop")
        self.assertEqual(result["exit_day"], 1)
        self.assertEqual(result["fill_price"], 98.0)
        self.assertAlmostEqual(result["realised_pct"], -0.02, places=10)

    def test_neither(self):
        rows = [(100, 101, 99, 100)]
        for _ in range(15):
            rows.append((100, 103, 99, 101))
        result = resolve_path(_payload(window=15, target=106.0), _frame(rows))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["outcome"], "neither")
        self.assertEqual(result["exit_day"], 15)
        self.assertEqual(result["fill_price"], 101.0)
        self.assertAlmostEqual(result["realised_pct"], 0.01, places=10)

    def test_same_bar_stop_and_target_counts_as_stop(self):
        rows = [(100, 101, 99, 100)]
        rows.append((100, 120, 90, 100))
        for _ in range(14):
            rows.append((100, 101, 99, 100))
        result = resolve_path(_payload(window=15), _frame(rows))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["outcome"], "stop")
        self.assertEqual(result["exit_day"], 1)
        self.assertEqual(result["fill_price"], 98.0)

    def test_gap_through_stop_fills_at_open(self):
        rows = [(100, 101, 99, 100)]
        rows.append((95.0, 96.0, 94.0, 95.5))  # open already through stop
        for _ in range(14):
            rows.append((100, 101, 99, 100))
        result = resolve_path(_payload(window=15), _frame(rows))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["outcome"], "stop")
        self.assertEqual(result["fill_price"], 95.0)
        self.assertAlmostEqual(result["realised_pct"], -0.05, places=10)
        self.assertAlmostEqual(result["worst_adverse_pct"], 0.06, places=10)

    def test_incomplete_window_returns_none(self):
        rows = [(100, 101, 99, 100)]
        for _ in range(10):
            rows.append((100, 101, 99, 100))
        result = resolve_path(_payload(window=15), _frame(rows))
        self.assertIsNone(result)

    def test_no_recommended_target_can_only_stop_or_neither(self):
        rows = [(100, 101, 99, 100)]
        rows.append((100, 120, 99, 110))  # would be a target if one existed
        for _ in range(14):
            rows.append((100, 101, 99, 100))
        result = resolve_path(_payload(window=15, target=None), _frame(rows))

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["outcome"], "neither")


class ResolveDueTests(unittest.TestCase):
    def test_appends_outcome_without_rewriting_call(self):
        rows = [(100, 101, 99, 100)]
        rows.append((100, 107, 99, 105))
        for _ in range(14):
            rows.append((100, 101, 99, 100))
        frame = _frame(rows)
        payload = _payload(window=15)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            original = copy.deepcopy(payload)
            append_call(payload, path)
            before = list_calls(path)[0]["payload"]

            inserted = resolve_due(
                path,
                as_of=date(2025, 2, 1),
                price_loader=lambda _ticker: frame,
            )
            after = list_calls(path)[0]["payload"]
            outcomes = list_outcomes(path)

            self.assertEqual(len(inserted), 1)
            self.assertEqual(len(outcomes), 1)
            self.assertEqual(outcomes[0]["outcome"], "target")
            self.assertEqual(before, after)
            self.assertEqual(after, original)

            again = resolve_due(
                path,
                as_of=date(2025, 2, 1),
                price_loader=lambda _ticker: frame,
            )
            self.assertEqual(again, [])
            self.assertEqual(len(list_outcomes(path)), 1)

    def test_incomplete_forward_path_writes_nothing(self):
        rows = [(100, 101, 99, 100)]
        for _ in range(10):
            rows.append((100, 101, 99, 100))
        frame = _frame(rows)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            append_call(_payload(window=15), path)
            inserted = resolve_due(
                path,
                as_of=date(2025, 2, 1),
                price_loader=lambda _ticker: frame,
            )
            self.assertEqual(inserted, [])
            self.assertEqual(list_outcomes(path), [])


if __name__ == "__main__":
    unittest.main()
