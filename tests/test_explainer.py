"""Explainer digit check. No network."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from agents.explainer_agent import digits_already_in_json, run_explainer

_AS_OF = date(2026, 9, 24)


def _card() -> dict:
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
        "confidence_basis": {
            "sample_size": 65,
            "note": "65 complete windows",
            "breakeven_win_rate": 0.29,
        },
        "regime": {
            "trend": "up",
            "rsi14": 58.3,
            "atr_pct": 1.5,
            "atr_percentile": 44,
            "above_200dma": True,
        },
        "risk_flags": [],
        "news": {"sentiment": "neutral", "summary": "Quiet tape.", "sources": []},
        "explanation": None,
        "skip_reason": None,
        "disclaimer": "Educational research only.",
    }


class DigitCheckTests(unittest.TestCase):
    def test_sentence_repeating_json_numbers_is_kept(self):
        text = (
            "Entry is 100.0 with a 2.0 percent stop at 98.0. "
            "The 6.0 percent target at 106.0 has probability 0.58 "
            "and a sell-by date of 2026-10-08 within the 15 day window."
        )
        self.assertTrue(digits_already_in_json(text, _card()))

    def test_invented_number_is_rejected(self):
        text = "The stock could reach 99 tomorrow."
        self.assertFalse(digits_already_in_json(text, _card()))

    def test_invented_date_is_rejected(self):
        text = "Results land on 2026-11-01."
        self.assertFalse(digits_already_in_json(text, _card()))


class ExplainerAgentTests(unittest.TestCase):
    def test_run_explainer_keeps_clean_prose(self):
        def completer(ticker, day, step, messages, schema, **kwargs):
            self.assertEqual(step, "explain")
            return {
                "text": (
                    "Action is BUY at 100.0. Stop 98.0 is 2.0 percent. "
                    "Recommended target 106.0 (+6.0) by 2026-10-08."
                )
            }

        with tempfile.TemporaryDirectory() as tmp:
            text = run_explainer(
                _card(),
                as_of=_AS_OF,
                completer=completer,
                cache_dir=Path(tmp),
            )
        self.assertIsNotNone(text)
        self.assertIn("BUY", text or "")

    def test_run_explainer_discards_invented_digits(self):
        def completer(ticker, day, step, messages, schema, **kwargs):
            return {"text": "Expect a move to 199 next week."}

        text = run_explainer(_card(), as_of=_AS_OF, completer=completer)
        self.assertIsNone(text)


if __name__ == "__main__":
    unittest.main()
