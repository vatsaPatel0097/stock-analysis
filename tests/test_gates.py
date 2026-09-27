"""One failing input and one passing input for each R2 rule. No network."""

from __future__ import annotations

import unittest
from datetime import date, datetime

from rules.gates import (
    GAP_RISK_NOTE,
    ScheduledEvent,
    gap_risk_note,
    gate_atr_too_high,
    gate_atr_too_low,
    gate_chase,
    gate_events,
    gate_liquidity,
    gate_min_probability,
    gate_sample_size,
    gate_stop_width,
)

_START = date(2026, 9, 24)
_END = date(2026, 10, 15)


class GateTests(unittest.TestCase):
    def test_r2_1_wider_stop_skips(self):
        failed = gate_stop_width(0.03)
        self.assertEqual(failed.action, "SKIP")
        self.assertEqual(failed.rule, "R2.1")
        self.assertIn("2%", failed.reason)

    def test_r2_1_two_percent_stop_passes(self):
        passed = gate_stop_width(0.02)
        self.assertIsNone(passed.action)
        self.assertIsNone(passed.reason)
        tighter = gate_stop_width(0.01)
        self.assertIsNone(tighter.action)

    def test_r2_2_atr_above_two_percent_skips(self):
        failed = gate_atr_too_high(2.01)
        self.assertEqual(failed.action, "SKIP")
        self.assertEqual(failed.rule, "R2.2")

    def test_r2_2_atr_at_two_percent_passes(self):
        self.assertIsNone(gate_atr_too_high(2.0).action)
        self.assertIsNone(gate_atr_too_high(1.52).action)

    def test_r2_3_atr_below_0_4_percent_skips(self):
        failed = gate_atr_too_low(0.39)
        self.assertEqual(failed.action, "SKIP")
        self.assertEqual(failed.rule, "R2.3")

    def test_r2_3_atr_at_0_4_percent_passes(self):
        self.assertIsNone(gate_atr_too_low(0.4).action)
        self.assertIsNone(gate_atr_too_low(1.52).action)

    def test_r2_4_event_inside_the_window_skips(self):
        for kind in ("earnings", "results", "AGM", "dividend record"):
            failed = gate_events(
                [ScheduledEvent(kind, date(2026, 10, 3))],
                window_start=_START,
                window_end=_END,
            )
            self.assertEqual(failed.action, "SKIP", kind)
            self.assertEqual(failed.rule, "R2.4")
            self.assertIn("2026-10-03", failed.reason)

    def test_r2_4_event_outside_the_window_passes(self):
        outside = gate_events(
            [ScheduledEvent("earnings", date(2026, 10, 16))],
            window_start=_START,
            window_end=_END,
        )
        self.assertIsNone(outside.action)
        edges = gate_events(
            [ScheduledEvent("dividend", _START), ScheduledEvent("results", _END)],
            window_start=_START,
            window_end=_END,
        )
        self.assertEqual(edges.action, "SKIP")
        self.assertIn("2026-09-24", edges.reason)
        empty = gate_events([], window_start=_START, window_end=_END)
        self.assertIsNone(empty.action)
        none_events = gate_events(None, window_start=datetime(2026, 9, 24), window_end=_END)
        self.assertIsNone(none_events.action)

    def test_r2_5_under_five_crore_skips(self):
        failed = gate_liquidity(49_999_999)
        self.assertEqual(failed.action, "SKIP")
        self.assertEqual(failed.rule, "R2.5")
        self.assertIn("5 crore", failed.reason)

    def test_r2_5_five_crore_passes(self):
        self.assertIsNone(gate_liquidity(50_000_000).action)
        self.assertIsNone(gate_liquidity(50_000_001).action)

    def test_r2_6_no_rung_at_35_percent_skips(self):
        failed = gate_min_probability([0.34, 0.10, 0.0])
        self.assertEqual(failed.action, "SKIP")
        self.assertEqual(failed.rule, "R2.6")
        empty = gate_min_probability([])
        self.assertEqual(empty.action, "SKIP")

    def test_r2_6_one_rung_at_35_percent_passes(self):
        passed = gate_min_probability([0.34, 0.35], user_target_prob=0.20)
        self.assertIsNone(passed.action)
        self.assertIsNone(passed.reason)
        self.assertIn("35%", passed.note)
        clear = gate_min_probability([0.58], user_target_prob=0.40)
        self.assertIsNone(clear.action)
        self.assertIsNone(clear.note)

    def test_r2_7_sample_under_50_skips(self):
        failed = gate_sample_size(49)
        self.assertEqual(failed.action, "SKIP")
        self.assertEqual(failed.rule, "R2.7")
        self.assertEqual(gate_sample_size(0).action, "SKIP")

    def test_r2_7_sample_of_50_passes(self):
        self.assertIsNone(gate_sample_size(50).action)
        self.assertIsNone(gate_sample_size(187).action)

    def test_r2_8_chase_above_1_5_percent_waits(self):
        failed = gate_chase(101.6, 100)
        self.assertEqual(failed.action, "WAIT")
        self.assertEqual(failed.rule, "R2.8")
        self.assertIsNone(failed.note)

    def test_r2_8_price_at_or_below_1_5_percent_passes(self):
        self.assertIsNone(gate_chase(101.5, 100).action)
        self.assertIsNone(gate_chase(100, 100).action)
        self.assertIsNone(gate_chase(99, 100).action)

    def test_r2_9_gap_risk_sentence_is_fixed(self):
        self.assertEqual(
            gap_risk_note(),
            "A 2% stop does not cap the loss at 2%. A gap-down can fill at −4% or worse.",
        )
        self.assertEqual(gap_risk_note(), GAP_RISK_NOTE)

    def test_bad_input_raises(self):
        with self.assertRaises(ValueError):
            gate_stop_width(-0.01)
        with self.assertRaises(ValueError):
            gate_atr_too_high(float("nan"))
        with self.assertRaises(ValueError):
            gate_events(
                [ScheduledEvent("split", _START)],
                window_start=_START,
                window_end=_END,
            )
        with self.assertRaises(ValueError):
            gate_sample_size(True)
        with self.assertRaises(ValueError):
            gate_min_probability([1.2])


if __name__ == "__main__":
    unittest.main()
