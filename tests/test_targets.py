"""target_stats and ladder against a hand-walked OHLC fixture. No network."""

from __future__ import annotations

import unittest

import pandas as pd

from analytics.targets import RUNGS, ladder, target_stats
from config import STOP_PCT, WINDOW_DAYS

# Twelve sessions. max_days=5 leaves seven entry bars (indexes 0..6).
# Windows, stop 2% checked before the target:
#   +3% and +6%: wins on days [2, 1, 1, 2, 1], two losses, no neither.
#   +10%: one win (day 1), four losses, two neither. 1/7 < 0.15.
#   +15% and +20%: no wins, four losses, three neither.
_ROWS = [
    (100, 101, 99.5),
    (101, 102, 100),
    (102, 109, 101),
    (100, 101, 96),
    (100, 112, 99),
    (100, 101, 99),
    (105, 107, 104),
    (100, 100.5, 99.5),
    (100, 100.5, 99.5),
    (100, 100.5, 99.5),
    (100, 100.5, 99.5),
    (100, 100.5, 99.5),
]
_MAX_DAYS = 5
_N = 7


def _frame(rows: list[tuple[float, float, float]] | None = None) -> pd.DataFrame:
    data = _ROWS if rows is None else rows
    return pd.DataFrame(
        {
            "Close": [row[0] for row in data],
            "High": [row[1] for row in data],
            "Low": [row[2] for row in data],
        }
    )


class TargetStatsTests(unittest.TestCase):
    def test_six_percent_matches_the_hand_count(self):
        stats = target_stats(_frame(), target=0.06, stop=0.02, max_days=_MAX_DAYS)

        self.assertEqual(stats["n"], _N)
        self.assertAlmostEqual(stats["prob"], 5 / 7, places=10)
        self.assertAlmostEqual(stats["loss"], 2 / 7, places=10)
        self.assertEqual(stats["days_median"], 1)
        self.assertEqual(stats["days_p25_p75"], [1, 2])

    def test_ten_percent_is_one_win_in_seven(self):
        stats = target_stats(_frame(), target=0.10, stop=0.02, max_days=_MAX_DAYS)

        self.assertEqual(stats["n"], _N)
        self.assertAlmostEqual(stats["prob"], 1 / 7, places=10)
        self.assertAlmostEqual(stats["loss"], 4 / 7, places=10)
        self.assertEqual(stats["days_median"], 1)
        self.assertEqual(stats["days_p25_p75"], [1, 1])

    def test_twenty_percent_never_hits(self):
        stats = target_stats(_frame(), target=0.20, stop=0.02, max_days=_MAX_DAYS)

        self.assertEqual(stats["n"], _N)
        self.assertEqual(stats["prob"], 0.0)
        self.assertAlmostEqual(stats["loss"], 4 / 7, places=10)
        self.assertIsNone(stats["days_median"])
        self.assertIsNone(stats["days_p25_p75"])

    def test_same_bar_stop_and_target_counts_as_a_loss(self):
        frame = _frame([(100, 101, 99), (100, 120, 90)])
        stats = target_stats(frame, target=0.06, stop=0.02, max_days=1)

        self.assertEqual(stats["n"], 1)
        self.assertEqual(stats["prob"], 0.0)
        self.assertEqual(stats["loss"], 1.0)
        self.assertIsNone(stats["days_median"])

    def test_stop_on_an_earlier_bar_beats_a_later_target(self):
        frame = _frame(
            [
                (100, 101, 99),
                (100, 101, 97),
                (100, 120, 99),
            ]
        )
        stats = target_stats(frame, target=0.06, stop=0.02, max_days=2)

        self.assertEqual(stats["prob"], 0.0)
        self.assertEqual(stats["loss"], 1.0)
        self.assertIsNone(stats["days_median"])

    def test_target_on_day_one_ignores_a_later_stop(self):
        frame = _frame(
            [
                (100, 101, 99),
                (100, 110, 99),
                (100, 101, 90),
            ]
        )
        stats = target_stats(frame, target=0.06, stop=0.02, max_days=2)

        self.assertEqual(stats["prob"], 1.0)
        self.assertEqual(stats["loss"], 0.0)
        self.assertEqual(stats["days_median"], 1)
        self.assertEqual(stats["days_p25_p75"], [1, 1])

    def test_short_history_has_no_windows(self):
        stats = target_stats(_frame(), target=0.06, stop=0.02, max_days=15)

        self.assertEqual(stats["n"], 0)
        self.assertEqual(stats["prob"], 0.0)
        self.assertEqual(stats["loss"], 0.0)
        self.assertIsNone(stats["days_median"])

    def test_defaults_are_the_config_stop_and_window(self):
        self.assertEqual(target_stats.__defaults__[-2:], (STOP_PCT, WINDOW_DAYS))


class LadderTests(unittest.TestCase):
    def test_recommended_rung_is_the_best_eligible_ev(self):
        rows = ladder(_frame(), stop=0.02, max_days=_MAX_DAYS)

        self.assertEqual([row["pct"] for row in rows], list(RUNGS))
        by_pct = {row["pct"]: row for row in rows}
        self.assertEqual(by_pct[0.03]["verdict"], "possible")
        self.assertEqual(by_pct[0.06]["verdict"], "recommended")
        self.assertEqual(by_pct[0.10]["verdict"], "unrealistic")
        self.assertEqual(by_pct[0.15]["verdict"], "unrealistic")
        self.assertEqual(by_pct[0.20]["verdict"], "unrealistic")
        self.assertAlmostEqual(by_pct[0.03]["ev"], 0.11 / 7, places=10)
        self.assertAlmostEqual(by_pct[0.06]["ev"], 0.26 / 7, places=10)
        self.assertAlmostEqual(by_pct[0.10]["ev"], 0.02 / 7, places=10)
        self.assertAlmostEqual(by_pct[0.20]["ev"], -0.08 / 7, places=10)
        self.assertEqual(by_pct[0.06]["days_p25_p75"], [1, 2])

    def test_user_target_is_appended_and_stays_low_odds(self):
        rows = ladder(_frame(), user_target_pct=0.08, stop=0.02, max_days=_MAX_DAYS)

        self.assertEqual([row["pct"] for row in rows], [*RUNGS, 0.08])
        wished = rows[-1]
        self.assertAlmostEqual(wished["prob"], 2 / 7, places=10)
        self.assertAlmostEqual(wished["loss"], 3 / 7, places=10)
        self.assertEqual(wished["days_median"], 1)
        self.assertEqual(wished["days_p25_p75"], [1, 1])
        self.assertEqual(wished["verdict"], "low_odds")
        self.assertAlmostEqual(wished["ev"], 0.10 / 7, places=10)
        self.assertEqual(rows[1]["verdict"], "recommended")

    def test_user_target_equal_to_a_rung_is_not_duplicated(self):
        rows = ladder(_frame(), user_target_pct=0.06, stop=0.02, max_days=_MAX_DAYS)
        self.assertEqual([row["pct"] for row in rows], list(RUNGS))

    def test_probability_of_0_35_stays_eligible(self):
        # 21 bars, one-day windows: entries 0..6 hit +6%, entries 7..19 do not.
        # 7/20 = 0.35, which is eligible. 0.15 is the unrealistic line, tested
        # by a three-win frame (3/20) versus a two-win frame (2/20).
        eligible = ladder(_cutoff_frame(7), stop=0.02, max_days=1)
        six = next(row for row in eligible if row["pct"] == 0.06)
        three = next(row for row in eligible if row["pct"] == 0.03)
        self.assertAlmostEqual(six["prob"], 0.35, places=10)
        self.assertEqual(six["verdict"], "recommended")
        self.assertEqual(three["verdict"], "possible")

        low = ladder(_cutoff_frame(3), stop=0.02, max_days=1)
        self.assertEqual(next(row for row in low if row["pct"] == 0.06)["verdict"], "low_odds")
        unrealistic = ladder(_cutoff_frame(2), stop=0.02, max_days=1)
        self.assertEqual(
            next(row for row in unrealistic if row["pct"] == 0.06)["verdict"],
            "unrealistic",
        )

    def test_bad_input_raises(self):
        with self.assertRaises(ValueError):
            target_stats(_frame().drop(columns=["High"]))
        with self.assertRaises(ValueError):
            target_stats(_frame(), target=0)
        with self.assertRaises(ValueError):
            target_stats(_frame(), max_days=0)
        with self.assertRaises(ValueError):
            ladder(_frame(), user_target_pct=-0.05)


def _cutoff_frame(wins: int) -> pd.DataFrame:
    rows = [(100, 101, 99)]
    for i in range(1, 21):
        if i <= wins:
            rows.append((100, 107, 99))
        else:
            rows.append((100, 101, 99))
    return _frame(rows)


if __name__ == "__main__":
    unittest.main()
