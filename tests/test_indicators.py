"""ATR(14) and RSI(14) against hand math on a 20-row fixture. No network."""

from __future__ import annotations

import unittest

import pandas as pd

from analytics.indicators import atr, rsi

# Twenty sessions. High and low are offsets from that day's close so each
# true range is fixed by hand.
_CLOSES = [
    100, 102, 101, 104, 103, 106, 108, 107, 110, 109,
    112, 115, 113, 116, 118, 117, 120, 119, 122, 124,
]
_HIGH_ADD = [2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1]
_LOW_SUB = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

# True ranges, bar by bar: first bar is high-low; later bars are
# max(high-low, |high-prev close|, |low-prev close|).
# [3, 3, 4, 4, 3, 4, 5, 3, 5, 3, 6, 4, 3, 4, 5, 3, 5, 3, 6, 3]
#
# First ATR uses the first 14 ranges. Their sum is 54, so ATR = 54/14 = 27/7.
# Last ATR uses the last 14 ranges. Their sum is 58, so ATR = 58/14 = 29/7.
_ATR_FIRST = 54 / 14
_ATR_LAST = 58 / 14
_ATR_DEFINED = [
    54 / 14,  # bars 0-13
    4.0,  # 56/14
    4.0,  # 56/14
    57 / 14,
    4.0,  # 56/14
    59 / 14,
    58 / 14,
]

# First 14 close-to-close changes: gains sum to 24, losses sum to 6.
# RS = (24/14) / (6/14) = 4. RSI = 100 - 100/5 = 80.
# Later values use Wilder: (previous average * 13 + this bar) / 14.
_RSI_DEFINED = [
    80.0,
    77.22772277227723,
    79.52054794520548,
    76.746669378623,
    79.10183850669667,
    80.51845943884354,
]


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "High": [close + add for close, add in zip(_CLOSES, _HIGH_ADD)],
            "Low": [close - sub for close, sub in zip(_CLOSES, _LOW_SUB)],
            "Close": _CLOSES,
        },
        index=pd.date_range("2024-01-02", periods=20, freq="B"),
    )


class IndicatorTests(unittest.TestCase):
    def test_atr14_matches_hand_average_of_true_range(self):
        frame = _fixture()
        result = atr(frame)

        self.assertEqual(list(result.index), list(frame.index))
        self.assertTrue(result.iloc[:13].isna().all())
        for got, expected in zip(result.iloc[13:].tolist(), _ATR_DEFINED):
            self.assertAlmostEqual(got, expected, places=10)
        self.assertAlmostEqual(result.iloc[13], _ATR_FIRST, places=10)
        self.assertAlmostEqual(result.iloc[-1], _ATR_LAST, places=10)
        self.assertEqual(list(frame["Close"]), _CLOSES)

    def test_rsi14_matches_wilder_hand_values(self):
        frame = _fixture()
        result = rsi(frame)

        self.assertEqual(list(result.index), list(frame.index))
        self.assertTrue(result.iloc[:14].isna().all())
        self.assertEqual(result.iloc[14], 80.0)
        for got, expected in zip(result.iloc[14:].tolist(), _RSI_DEFINED):
            self.assertAlmostEqual(got, expected, places=10)

    def test_fourteen_bars_have_one_atr_and_no_rsi(self):
        frame = _fixture().iloc[:14]
        averaged = atr(frame)
        strength = rsi(frame)

        self.assertTrue(averaged.iloc[:13].isna().all())
        self.assertAlmostEqual(averaged.iloc[-1], _ATR_FIRST, places=10)
        self.assertTrue(strength.isna().all())

    def test_short_and_empty_tables_stay_undefined(self):
        short = atr(_fixture().iloc[:10])
        self.assertTrue(short.isna().all())
        self.assertEqual(len(short), 10)

        empty = pd.DataFrame(columns=["High", "Low", "Close"])
        self.assertEqual(len(atr(empty)), 0)
        self.assertEqual(len(rsi(empty)), 0)

    def test_bad_input_raises(self):
        frame = _fixture().drop(columns=["High"])
        with self.assertRaises(ValueError):
            atr(frame)
        with self.assertRaises(ValueError):
            rsi(pd.DataFrame({"Close": [1.0, None, 2.0]}))
        with self.assertRaises(ValueError):
            atr(_fixture(), period=0)


if __name__ == "__main__":
    unittest.main()
