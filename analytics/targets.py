"""Target hit-rate and the fixed percent ladder. No network.

v1 counts every complete window in the frame (project.md §6.3).
Stop is checked before target on the same bar.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import MIN_TARGET_PROB, STOP_PCT, WINDOW_DAYS

RUNGS = (0.03, 0.06, 0.10, 0.15, 0.20)
# §6.3b: below this probability the rung is labeled unrealistic.
UNREALISTIC_PROB = 0.15

_COLUMNS = ("High", "Low", "Close")


def target_stats(
    frame: pd.DataFrame,
    target: float = 0.06,
    stop: float = STOP_PCT,
    max_days: int = WINDOW_DAYS,
) -> dict:
    """Hit-rate of ``target`` before ``stop`` inside ``max_days`` sessions.

    ``target`` and ``stop`` are fractions (0.06 means +6%). Days are counted
    from the entry bar. A bar that touches both levels is a loss.
    """
    prices = _require_prices(frame)
    _check_target(target)
    _check_stop(stop)
    _check_days(max_days)

    high = prices["High"].to_numpy(dtype=float)
    low = prices["Low"].to_numpy(dtype=float)
    close = prices["Close"].to_numpy(dtype=float)
    wins: list[int] = []
    losses = 0
    neither = 0
    last_start = len(close) - max_days
    for i in range(max(last_start, 0)):
        entry = close[i]
        stop_px = entry * (1.0 - stop)
        target_px = entry * (1.0 + target)
        for j in range(i + 1, i + 1 + max_days):
            if low[j] <= stop_px:
                losses += 1
                break
            if high[j] >= target_px:
                wins.append(j - i)
                break
        else:
            neither += 1

    n = len(wins) + losses + neither
    if n == 0:
        return {
            "prob": 0.0,
            "loss": 0.0,
            "n": 0,
            "days_median": None,
            "days_p25_p75": None,
        }
    days = np.asarray(wins, dtype=float)
    return {
        "prob": len(wins) / n,
        "loss": losses / n,
        "n": n,
        "days_median": int(np.median(days)) if len(days) else None,
        "days_p25_p75": (
            [int(np.percentile(days, 25)), int(np.percentile(days, 75))]
            if len(days)
            else None
        ),
    }


def ladder(
    frame: pd.DataFrame,
    user_target_pct: float | None = None,
    stop: float = STOP_PCT,
    max_days: int = WINDOW_DAYS,
) -> list[dict]:
    """Rungs +3 / +6 / +10 / +15 / +20, plus an optional wished percent.

    The recommended row is the eligible rung (probability >= 0.35) with the
    highest expected value. A wished percent that matches a rung is not copied.
    """
    pcts = list(RUNGS)
    if user_target_pct is not None:
        _check_target(user_target_pct)
        if not any(math.isclose(user_target_pct, rung, rel_tol=0.0, abs_tol=1e-9) for rung in pcts):
            pcts.append(user_target_pct)

    rows: list[dict] = []
    for pct in pcts:
        row = {"pct": pct, **target_stats(frame, pct, stop, max_days)}
        row["ev"] = row["pct"] * row["prob"] - stop * row["loss"]
        row["verdict"] = _verdict(row["prob"])
        rows.append(row)

    eligible = [row for row in rows if row["prob"] >= MIN_TARGET_PROB]
    if eligible:
        best = max(eligible, key=lambda row: row["ev"])
        best["verdict"] = "recommended"
    return rows


def _verdict(prob: float) -> str:
    if prob < UNREALISTIC_PROB:
        return "unrealistic"
    if prob < MIN_TARGET_PROB:
        return "low_odds"
    return "possible"


def _require_prices(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Targets need a price table.")
    missing = [name for name in _COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Price table is missing {', '.join(missing)}.")
    data = pd.DataFrame(
        {name: pd.to_numeric(frame[name], errors="coerce") for name in _COLUMNS},
        index=frame.index,
    )
    if data.isna().any().any():
        raise ValueError("Price columns must be numbers with no gaps.")
    return data


def _check_target(target: float) -> None:
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        raise ValueError("target percent must be a positive number.")
    if not math.isfinite(float(target)) or float(target) <= 0.0:
        raise ValueError("target percent must be a positive number.")


def _check_stop(stop: float) -> None:
    if isinstance(stop, bool) or not isinstance(stop, (int, float)):
        raise ValueError("stop percent must be between 0 and 1.")
    if not math.isfinite(float(stop)) or float(stop) <= 0.0 or float(stop) >= 1.0:
        raise ValueError("stop percent must be between 0 and 1.")


def _check_days(max_days: int) -> None:
    if isinstance(max_days, bool) or not isinstance(max_days, int) or max_days < 1:
        raise ValueError("max_days must be a positive integer.")
