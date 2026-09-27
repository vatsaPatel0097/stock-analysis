"""Pure-code backtest on a short fixture. No network."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from eval.backtest import run_backtest
from eval.resolve import list_outcomes
from eval.universe import UNIVERSE
from store.calls import list_calls


def _ohlcv(
    n: int,
    *,
    start: float = 100.0,
    step: float = 0.25,
    spread: float = 0.01,
    volume: float = 2_000_000.0,
    start_day: str = "2023-01-02",
) -> pd.DataFrame:
    rows = []
    close = start
    for _ in range(n):
        high = close * (1.0 + spread)
        low = close * (1.0 - spread)
        rows.append((close, high, low, close, volume))
        close = close + step
    index = pd.date_range(start_day, periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "Open": [row[0] for row in rows],
            "High": [row[1] for row in rows],
            "Low": [row[2] for row in rows],
            "Close": [row[3] for row in rows],
            "Volume": [row[4] for row in rows],
        },
        index=index,
    )


def _session(value) -> date:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC")
    return stamp.date()


class UniverseTests(unittest.TestCase):
    def test_universe_has_one_hundred_unique_ns_tickers(self):
        self.assertEqual(len(UNIVERSE), 100)
        self.assertEqual(len(set(UNIVERSE)), 100)
        for ticker in UNIVERSE:
            self.assertTrue(ticker.endswith(".NS"))


class BacktestTests(unittest.TestCase):
    def test_writes_report_resolves_outcome_and_ignores_future_bars(self):
        # Long gentle climb, then a gap-down stop the day after entry_a.
        history = _ohlcv(600, start_day="2023-01-02")
        sessions = [_session(stamp) for stamp in history.index]
        # Pick two consecutive sessions in 2025 with room for a 15-day window.
        candidates = [day for day in sessions if day >= date(2025, 1, 2)]
        self.assertGreaterEqual(len(candidates), 40)
        entry_a = candidates[0]
        entry_b = candidates[1]

        entry_close = float(
            history.loc[[_session(stamp) == entry_a for stamp in history.index], "Close"].iloc[0]
        )

        # Frame = history through entry_a, then gap bar on entry_b, then fillers,
        # then a wild spike that must not change the entry_a card.
        through_a = history.loc[[_session(stamp) <= entry_a for stamp in history.index]].copy()
        gap = pd.DataFrame(
            {
                "Open": [90.0],
                "High": [91.0],
                "Low": [89.0],
                "Close": [90.0],
                "Volume": [2_000_000.0],
            },
            index=pd.DatetimeIndex([pd.Timestamp(entry_b, tz="UTC")], name="Date"),
        )
        fillers = []
        for _ in range(20):
            px = entry_close
            fillers.append((px, px * 1.01, px * 0.99, px, 2_000_000.0))
        fill_index = pd.date_range(
            pd.Timestamp(entry_b, tz="UTC") + pd.Timedelta(days=1),
            periods=len(fillers),
            freq="B",
            tz="UTC",
        )
        fill_frame = pd.DataFrame(
            {
                "Open": [row[0] for row in fillers],
                "High": [row[1] for row in fillers],
                "Low": [row[2] for row in fillers],
                "Close": [row[3] for row in fillers],
                "Volume": [row[4] for row in fillers],
            },
            index=fill_index,
        )
        spike = pd.DataFrame(
            {
                "Open": [entry_close * 3],
                "High": [entry_close * 3.1],
                "Low": [entry_close * 2.9],
                "Close": [entry_close * 3],
                "Volume": [2_000_000.0],
            },
            index=pd.DatetimeIndex([fill_index[-1] + pd.Timedelta(days=3)], name="Date"),
        )
        frame = pd.concat([through_a, gap, fill_frame, spike]).sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]

        live_db = Path("calls.db")
        live_before = live_db.read_bytes() if live_db.exists() else None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scratch = root / "scratch.db"
            report_path = root / "report.json"

            report = run_backtest(
                ["RELIANCE.NS"],
                lambda _ticker: frame.copy(),
                start=entry_a,
                report_path=report_path,
                calls_path=scratch,
                entry_dates=[entry_a, entry_b],
            )

            self.assertTrue(report_path.is_file())
            written = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(written["universe_size"], 1)
            self.assertTrue(written["no_rules_changed_on_this_period"])
            self.assertGreaterEqual(written["analyzed"], 1)

            calls = list_calls(scratch)
            self.assertGreaterEqual(len(calls), 1)
            first = next(
                row
                for row in calls
                if row["payload"]["as_of"].startswith(entry_a.isoformat())
            )
            self.assertAlmostEqual(first["payload"]["plan"]["entry"], round(entry_close, 2), places=2)
            self.assertLess(first["payload"]["plan"]["entry"], entry_close * 2)

            outcomes = list_outcomes(scratch)
            stop_rows = [row for row in outcomes if row["outcome"] == "stop"]
            self.assertGreaterEqual(len(stop_rows), 1)
            hit = next(
                row
                for row in stop_rows
                if abs(float(row["fill_price"]) - 90.0) < 1e-9
            )
            self.assertAlmostEqual(
                hit["realised_pct"],
                (90.0 - round(entry_close, 2)) / round(entry_close, 2),
                places=4,
            )

            self.assertIn("survivorship_note", report)
            self.assertIn("Educational research only", report["disclaimer"])

        if live_before is None:
            self.assertFalse(live_db.exists())
        else:
            self.assertEqual(live_db.read_bytes(), live_before)


if __name__ == "__main__":
    unittest.main()
