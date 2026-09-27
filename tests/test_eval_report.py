"""Calibration report on a tiny fixture database. No network."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eval.report import build_report
from eval.resolve import _append_outcome
from store.calls import append_call


def _payload(*, action: str, prob: float, ticker: str = "TEST.NS") -> dict:
    return {
        "query": ticker.replace(".NS", ""),
        "ticker": ticker,
        "as_of": "2025-01-02T15:30:00+05:30",
        "plan": {
            "action": action,
            "entry": 100.0,
            "stop_loss": 98.0,
            "stop_pct": 2.0,
            "window_days": 15,
            "recommended_target": 106.0,
            "recommended_target_pct": 6.0,
            "sell_by_day": 10,
            "sell_by_date": None,
        },
        "targets": [
            {
                "price": 106.0,
                "pct": 6.0,
                "prob": prob,
                "days_median": 7,
                "days_p25_p75": [5, 10],
                "verdict": "recommended",
            }
        ],
        "user_target": None,
        "disclaimer": "Educational research only. Not investment advice.",
    }


def _seed_outcome(path: Path, call_id: int, *, outcome: str, realised: float, worst: float) -> None:
    _append_outcome(
        call_id,
        {
            "outcome": outcome,
            "exit_day": 3,
            "fill_price": 100.0 * (1.0 + realised),
            "realised_pct": realised,
            "worst_adverse_pct": worst,
        },
        path,
    )


class ReportTests(unittest.TestCase):
    def test_calibration_bucket_and_loss_shares(self):
        # Four resolved calls in the 0.35–0.45 bucket (predicted 0.40).
        # Wins: 1 of 4 → actual 0.25. Abs error = |0.40 − 0.25| = 0.15.
        # Stops (3): realised −0.02, −0.022, −0.04 → shares 1/3, 1/3, 1/3.
        # Worst worse than fill on the −0.02 stop (worst 0.05 > 0.02) → 1/3.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            ids = []
            for action in ("BUY", "BUY", "BUY", "SKIP"):
                ids.append(append_call(_payload(action=action, prob=0.40), path))

            _seed_outcome(path, ids[0], outcome="target", realised=0.06, worst=0.01)
            _seed_outcome(path, ids[1], outcome="stop", realised=-0.02, worst=0.05)
            _seed_outcome(path, ids[2], outcome="stop", realised=-0.022, worst=0.022)
            _seed_outcome(path, ids[3], outcome="stop", realised=-0.04, worst=0.04)

            report = build_report(path)

        self.assertEqual(report["resolved"], 4)
        self.assertEqual(report["counts"], {"target": 1, "stop": 3, "neither": 0})
        self.assertEqual(report["buy"]["n"], 3)
        self.assertAlmostEqual(report["buy"]["win_rate"], 1 / 3, places=10)

        bucket = next(row for row in report["calibration"]["buckets"] if row["lo"] == 0.35)
        self.assertEqual(bucket["n"], 4)
        self.assertAlmostEqual(bucket["predicted_mean"], 0.40, places=10)
        self.assertAlmostEqual(bucket["actual_win_rate"], 0.25, places=10)
        self.assertAlmostEqual(bucket["abs_error"], 0.15, places=10)
        self.assertAlmostEqual(report["calibration"]["mean_abs_error"], 0.15, places=10)

        loss = report["loss_distribution"]
        self.assertEqual(loss["n"], 3)
        self.assertAlmostEqual(loss["at_or_better_than_2pct"], 1 / 3, places=10)
        self.assertAlmostEqual(loss["between_2_and_2_5pct"], 1 / 3, places=10)
        self.assertAlmostEqual(loss["worse_than_2_5pct"], 1 / 3, places=10)
        self.assertAlmostEqual(loss["worst_worse_than_fill"], 1 / 3, places=10)
        self.assertIn("Educational research only", report["disclaimer"])


if __name__ == "__main__":
    unittest.main()
