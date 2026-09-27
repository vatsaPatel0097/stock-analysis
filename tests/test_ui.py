"""UI formatters render analyze() JSON only. No network."""

from __future__ import annotations

import unittest

from ui import view


def _buy_payload() -> dict:
    return {
        "query": "Reliance",
        "ticker": "RELIANCE.NS",
        "as_of": "2026-09-24T15:30:00+05:30",
        "price": {
            "current": 100.0,
            "day_high": 101.0,
            "day_low": 99.0,
            "prev_close": 99.5,
        },
        "plan": {
            "action": "BUY",
            "entry": 100.0,
            "stop_loss": 98.0,
            "stop_pct": 2.0,
            "window_days": 15,
            "recommended_target": 106.0,
            "recommended_target_pct": 6.0,
            "sell_by_day": 10,
            "sell_by_date": "2026-10-08",
        },
        "targets": [
            {
                "price": 103.0,
                "pct": 3.0,
                "prob": 0.71,
                "days_median": 3,
                "days_p25_p75": [2, 4],
                "verdict": "possible",
            },
            {
                "price": 106.0,
                "pct": 6.0,
                "prob": 0.58,
                "days_median": 7,
                "days_p25_p75": [5, 10],
                "verdict": "recommended",
            },
        ],
        "user_target": None,
        "regime": {
            "trend": "up",
            "rsi14": 58.3,
            "atr_pct": 1.5,
            "atr_percentile": 44,
            "above_200dma": True,
        },
        "news": None,
        "explanation": None,
        "skip_reason": None,
        "disclaimer": view.DISCLAIMER,
    }


def _skip_payload() -> dict:
    payload = _buy_payload()
    payload["plan"] = {
        "action": "SKIP",
        "entry": 100.0,
        "stop_loss": 98.0,
        "stop_pct": 2.0,
        "window_days": 15,
        "recommended_target": None,
        "recommended_target_pct": None,
        "sell_by_day": None,
        "sell_by_date": None,
    }
    payload["skip_reason"] = "Fewer than 50 similar days."
    return payload


class ViewStateTests(unittest.TestCase):
    def test_empty_copy(self):
        self.assertIn("Type a name", view.empty_message())

    def test_error_sentence(self):
        self.assertEqual(view.error_message("  No NSE stock matches 'X'.  "), "No NSE stock matches 'X'.")
        self.assertEqual(view.error_message(""), view.EMPTY_ERROR)

    def test_pick_list_has_no_prices(self):
        rows = view.pick_list_rows(
            [
                {"name": "Tata Steel", "ticker": "TATASTEEL.NS"},
                {"name": "Tata Power", "ticker": "TATAPOWER.NS"},
            ]
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows[0].keys()), {"name", "ticker"})


class ViewCardTests(unittest.TestCase):
    def test_buy_card_stop_matches_json(self):
        payload = _buy_payload()
        cells = view.plan_cells(payload)
        self.assertIn("₹98.00", cells["Stop"])
        self.assertEqual(cells["Stop"], "₹98.00 (2.0%)")
        banner = view.action_banner(payload)
        self.assertEqual(banner["action"], "BUY")
        self.assertEqual(banner["text"], "BUY")
        rows = view.target_table_rows(payload)
        self.assertEqual(rows[1]["Verdict"], "recommended")
        self.assertIn("gap-down", view.card_gap_risk())
        self.assertIn("Educational research only", view.card_disclaimer())

    def test_skip_card_keeps_prices_and_dash_targets(self):
        payload = _skip_payload()
        cells = view.plan_cells(payload)
        self.assertEqual(cells["Buy at"], "₹100.00")
        self.assertEqual(cells["Recommended target"], "—")
        self.assertEqual(cells["Sell by"], "—")
        banner = view.action_banner(payload)
        self.assertEqual(banner["action"], "SKIP")
        self.assertIn("Fewer than 50", banner["text"])
        self.assertEqual(view.format_rupees(payload["plan"]["stop_loss"]), "₹98.00")
        self.assertEqual(view.format_rupees(payload["plan"]["stop_loss"]), "₹98.00")


if __name__ == "__main__":
    unittest.main()
