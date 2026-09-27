"""Regime on hand-built closes. No network."""

from __future__ import annotations

import unittest

import pandas as pd

from analytics.regime import regime, rsi_bucket, volatility_bucket


def _flat(n: int, close: float, *, high_add: float = 1.0, low_sub: float = 1.0) -> list[tuple[float, float, float]]:
    return [(close, close + high_add, close - low_sub) for _ in range(n)]


def _frame(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Close": [row[0] for row in rows],
            "High": [row[1] for row in rows],
            "Low": [row[2] for row in rows],
        }
    )


class RegimeTests(unittest.TestCase):
    def test_up_trend_dma_rsi_and_atr_percentile(self):
        # 199 closes at 100, then 110. Last true range is 12; the prior 13 are 2.
        rows = _flat(199, 100)
        rows.append((110, 112, 108))
        result = regime(_frame(rows))

        self.assertAlmostEqual(result["sma50"], 5010 / 50, places=10)
        self.assertAlmostEqual(result["sma200"], 20010 / 200, places=10)
        self.assertEqual(result["trend"], "up")
        self.assertTrue(result["above_50dma"])
        self.assertTrue(result["above_200dma"])
        self.assertEqual(result["rsi14"], 100.0)
        self.assertEqual(result["rsi_bucket"], "above_70")
        self.assertAlmostEqual(result["atr"], 38 / 14, places=10)
        self.assertAlmostEqual(result["atr_pct"], (38 / 14) / 110 * 100, places=10)
        self.assertEqual(result["atr_percentile"], 100.0)
        self.assertEqual(result["atr_bucket"], "high")

    def test_down_trend_when_close_is_below_both_averages(self):
        rows = _flat(199, 100)
        rows.append((90, 92, 88))
        result = regime(_frame(rows))

        self.assertAlmostEqual(result["sma50"], 4990 / 50, places=10)
        self.assertAlmostEqual(result["sma200"], 19990 / 200, places=10)
        self.assertEqual(result["trend"], "down")
        self.assertFalse(result["above_50dma"])
        self.assertFalse(result["above_200dma"])
        self.assertEqual(result["rsi14"], 0.0)
        self.assertEqual(result["rsi_bucket"], "below_30")
        self.assertAlmostEqual(result["atr"], 38 / 14, places=10)
        self.assertAlmostEqual(result["atr_pct"], (38 / 14) / 90 * 100, places=10)

    def test_range_when_fast_and_slow_averages_disagree(self):
        # 150 closes at 200, 49 closes at 100, last close 101.
        # SMA50 = 100.02 (above). SMA200 = 175.005 (below). Last 14 true ranges are 2.
        rows = _flat(150, 200)
        rows.extend(_flat(49, 100))
        rows.append((101, 102, 100))
        result = regime(_frame(rows))

        self.assertAlmostEqual(result["sma50"], 5001 / 50, places=10)
        self.assertAlmostEqual(result["sma200"], 35001 / 200, places=10)
        self.assertEqual(result["trend"], "range")
        self.assertTrue(result["above_50dma"])
        self.assertFalse(result["above_200dma"])
        self.assertAlmostEqual(result["atr"], 2.0, places=10)
        self.assertAlmostEqual(result["atr_pct"], 200 / 101, places=10)
        # Shock of -100 at the break, then 48 flat bars, then +1.
        # avg_loss = (100/14) * (13/14)**49, avg_gain = 1/14.
        avg_loss = (100 / 14) * (13 / 14) ** 49
        avg_gain = 1 / 14
        expected_rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        self.assertAlmostEqual(result["rsi14"], expected_rsi, places=8)

    def test_short_history_leaves_the_slow_average_unset(self):
        result = regime(_frame(_flat(60, 100)))

        self.assertIsNone(result["sma200"])
        self.assertIsNone(result["above_200dma"])
        self.assertIsNone(result["trend"])
        self.assertAlmostEqual(result["sma50"], 100.0, places=10)
        self.assertFalse(result["above_50dma"])
        self.assertEqual(result["rsi14"], 50.0)
        self.assertEqual(result["rsi_bucket"], "50_70")
        self.assertEqual(result["atr_percentile"], 0.0)
        self.assertEqual(result["atr_bucket"], "low")

    def test_touching_both_averages_is_range(self):
        result = regime(_frame(_flat(200, 100)))

        self.assertEqual(result["sma50"], 100.0)
        self.assertEqual(result["sma200"], 100.0)
        self.assertFalse(result["above_50dma"])
        self.assertFalse(result["above_200dma"])
        self.assertEqual(result["trend"], "range")

    def test_rsi_buckets_follow_the_spec_edges(self):
        self.assertEqual(rsi_bucket(29.9), "below_30")
        self.assertEqual(rsi_bucket(30.0), "30_50")
        self.assertEqual(rsi_bucket(49.9), "30_50")
        self.assertEqual(rsi_bucket(50.0), "50_70")
        self.assertEqual(rsi_bucket(70.0), "50_70")
        self.assertEqual(rsi_bucket(70.1), "above_70")

    def test_atr_buckets_follow_terciles(self):
        self.assertEqual(volatility_bucket(0.0), "low")
        self.assertEqual(volatility_bucket(100.0 / 3.0 - 0.01), "low")
        self.assertEqual(volatility_bucket(100.0 / 3.0), "mid")
        self.assertEqual(volatility_bucket(50.0), "mid")
        self.assertEqual(volatility_bucket(200.0 / 3.0), "high")
        self.assertEqual(volatility_bucket(100.0), "high")
        self.assertIsNone(volatility_bucket(None))

    def test_empty_and_bad_input(self):
        empty = regime(pd.DataFrame(columns=["High", "Low", "Close"]))
        self.assertIsNone(empty["trend"])
        self.assertIsNone(empty["atr"])
        with self.assertRaises(ValueError):
            regime(_frame(_flat(20, 100)).drop(columns=["High"]))


if __name__ == "__main__":
    unittest.main()
