"""Orchestrator: resolve → prices → math → gates → §3 JSON. CLI entry point."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from analytics.indicators import atr
from analytics.regime import regime as compute_regime
from analytics.targets import RUNGS, ladder
from config import STOP_PCT, WINDOW_DAYS
from data.calendar import trading_days
from data.prices import PriceError, fetch_daily
from data.resolve import Candidate, ResolveError, resolve
from rules.gates import (
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
from store.calls import append_call

IST = ZoneInfo("Asia/Kolkata")
DISCLAIMER = (
    "Educational research only. Not investment advice. "
    "Past patterns do not predict future results."
)
BREAKEVEN_WIN_RATE = 0.29
_LIQUIDITY_DAYS = 20
_SECTION3_KEYS = (
    "query",
    "ticker",
    "as_of",
    "price",
    "expected_range_today",
    "plan",
    "targets",
    "user_target",
    "confidence_basis",
    "regime",
    "risk_flags",
    "news",
    "explanation",
    "skip_reason",
    "disclaimer",
)


class AmbiguousQuery(Exception):
    """Several listings match. The caller must show a pick list."""

    def __init__(self, query: str, matches: tuple[Candidate, ...]):
        self.query = query
        self.matches = matches
        names = ", ".join(f"{row.name} ({row.ticker})" for row in matches)
        super().__init__(f"Several matches for {query!r}: {names}.")


def analyze(
    query: str,
    user_target: float | None = None,
    *,
    as_of: date | None = None,
    price_loader=None,
    searcher=None,
    calls_path: Path | str | None = None,
    events: list[ScheduledEvent] | None = None,
) -> dict:
    """Return the project.md §3 trade card for one resolved NSE stock.

    ``user_target`` is a wished price in rupees. ``news`` and ``explanation``
    are null until Phase 2. Ambiguous names raise ``AmbiguousQuery`` and are
    not logged. Finished BUY / WAIT / SKIP rows are appended to the call log.
    """
    shown = _require_query(query)
    wished = _optional_rupees(user_target, "user_target")
    day = _as_of_date(as_of)

    resolved = resolve(shown, searcher=searcher, as_of=day)
    if resolved.ticker is None:
        raise AmbiguousQuery(resolved.query, resolved.matches)

    ticker = resolved.ticker
    frame = _load_prices(ticker, day, price_loader)
    if len(frame) < 2:
        raise PriceError(f"Need at least two daily bars for {ticker}.")

    entry = float(frame["Close"].iloc[-1])
    day_high = float(frame["High"].iloc[-1])
    day_low = float(frame["Low"].iloc[-1])
    prev_close = float(frame["Close"].iloc[-2])
    if not all(math.isfinite(value) for value in (entry, day_high, day_low, prev_close)):
        raise PriceError(f"Price bars for {ticker} are not usable.")
    if entry <= 0.0 or prev_close <= 0.0:
        raise PriceError(f"Price bars for {ticker} are not usable.")

    user_pct = None if wished is None else (wished / entry) - 1.0
    if user_pct is not None and user_pct <= 0.0:
        raise ValueError("user_target must be above the entry price.")

    rows = ladder(frame, user_target_pct=user_pct)
    fixed = [row for row in rows if _is_rung(row["pct"])]
    wished_row = None if user_pct is None else _row_for_pct(rows, user_pct)

    reg = compute_regime(frame)
    atr_series = atr(frame, 14)
    last_atr = float(atr_series.iloc[-1]) if len(atr_series) and math.isfinite(float(atr_series.iloc[-1])) else None
    atr_pct = reg.get("atr_pct")
    if atr_pct is None:
        raise PriceError(f"Not enough history to measure ATR for {ticker}.")

    sample_size = int(fixed[0]["n"]) if fixed else 0
    avg_traded = _avg_traded_value(frame)
    stop_loss = entry * (1.0 - STOP_PCT)
    session = _session_date(frame.index[-1])
    window_end = _nth_trading_day_after(session, WINDOW_DAYS)

    action, skip_reason, gate_note = _decide_action(
        stop_pct=STOP_PCT,
        atr_pct=float(atr_pct),
        events=events if events is not None else [],
        window_start=session,
        window_end=window_end,
        avg_traded_value=avg_traded,
        probs=[float(row["prob"]) for row in fixed],
        user_target_prob=None if wished_row is None else float(wished_row["prob"]),
        sample_size=sample_size,
        price=entry,
        prev_close=prev_close,
    )

    recommended = next((row for row in rows if row["verdict"] == "recommended"), None)
    sell_by_day = None
    sell_by_date = None
    if recommended is not None and recommended.get("days_p25_p75"):
        sell_by_day = int(recommended["days_p25_p75"][1])
        sell_by_date = _nth_trading_day_after(session, sell_by_day).isoformat()

    loss_rate = float(recommended["loss"]) if recommended is not None else (
        float(fixed[0]["loss"]) if fixed else 0.0
    )
    note = (
        f"{sample_size} complete windows in the history; "
        f"stop hit first on {round(loss_rate * 100):.0f}% of them"
    )
    if gate_note:
        note = f"{note}. {gate_note}"

    now = datetime.now(IST)
    if as_of is not None:
        now = datetime(day.year, day.month, day.day, 15, 30, tzinfo=IST)

    payload = {
        "query": shown,
        "ticker": ticker,
        "as_of": now.isoformat(),
        "price": {
            "current": _money(entry),
            "day_high": _money(day_high),
            "day_low": _money(day_low),
            "prev_close": _money(prev_close),
        },
        "expected_range_today": {
            "low": _money(entry - last_atr) if last_atr is not None else None,
            "high": _money(entry + last_atr) if last_atr is not None else None,
            "basis": f"ATR14 = {_pct_points(float(atr_pct))}%",
        },
        "plan": {
            "action": action,
            "entry": _money(entry),
            "stop_loss": _money(stop_loss),
            "stop_pct": _pct_points(STOP_PCT * 100.0),
            "window_days": WINDOW_DAYS,
            "recommended_target": (
                None if recommended is None else _money(entry * (1.0 + float(recommended["pct"])))
            ),
            "recommended_target_pct": (
                None if recommended is None else _pct_points(float(recommended["pct"]) * 100.0)
            ),
            "sell_by_day": sell_by_day,
            "sell_by_date": sell_by_date,
        },
        "targets": [_target_row(entry, row) for row in fixed],
        "user_target": None if wished_row is None else _user_target_row(entry, wished_row, wished),
        "confidence_basis": {
            "sample_size": sample_size,
            "note": note,
            "breakeven_win_rate": BREAKEVEN_WIN_RATE,
        },
        "regime": {
            "trend": reg.get("trend"),
            "rsi14": None if reg.get("rsi14") is None else round(float(reg["rsi14"]), 1),
            "atr_pct": _pct_points(float(atr_pct)),
            "atr_percentile": (
                None
                if reg.get("atr_percentile") is None
                else int(round(float(reg["atr_percentile"])))
            ),
            "above_200dma": reg.get("above_200dma"),
        },
        "risk_flags": [],
        "news": None,
        "explanation": None,
        "skip_reason": skip_reason,
        "disclaimer": DISCLAIMER,
    }
    _assert_section3(payload)
    append_call(payload, calls_path)
    return payload


def _decide_action(
    *,
    stop_pct: float,
    atr_pct: float,
    events: list[ScheduledEvent],
    window_start: date,
    window_end: date,
    avg_traded_value: float,
    probs: list[float],
    user_target_prob: float | None,
    sample_size: int,
    price: float,
    prev_close: float,
) -> tuple[str, str | None, str | None]:
    """First SKIP in R2 order wins; else WAIT; else BUY."""
    results = [
        gate_stop_width(stop_pct),
        gate_atr_too_high(atr_pct),
        gate_atr_too_low(atr_pct),
        gate_events(events, window_start=window_start, window_end=window_end),
        gate_liquidity(avg_traded_value),
        gate_min_probability(probs, user_target_prob),
        gate_sample_size(sample_size),
        gate_chase(price, prev_close),
    ]
    note = next((item.note for item in results if item.note), None)
    for item in results:
        if item.action == "SKIP":
            return "SKIP", item.reason, note
    for item in results:
        if item.action == "WAIT":
            return "WAIT", None, note
    return "BUY", None, note


def _load_prices(ticker: str, day: date, price_loader) -> pd.DataFrame:
    if price_loader is not None:
        frame = price_loader(ticker, as_of=day)
    else:
        frame = fetch_daily(ticker, as_of=day)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise PriceError(f"No daily prices for {ticker}.")
    return frame


def _avg_traded_value(frame: pd.DataFrame) -> float:
    tail = frame.tail(_LIQUIDITY_DAYS)
    values = pd.to_numeric(tail["Close"], errors="coerce") * pd.to_numeric(
        tail["Volume"], errors="coerce"
    )
    values = values.dropna()
    if values.empty:
        return 0.0
    return float(values.mean())


def _target_row(entry: float, row: dict) -> dict:
    days = row.get("days_p25_p75")
    return {
        "price": _money(entry * (1.0 + float(row["pct"]))),
        "pct": _pct_points(float(row["pct"]) * 100.0),
        "prob": _prob(float(row["prob"])),
        "days_median": row.get("days_median"),
        "days_p25_p75": None if days is None else [int(days[0]), int(days[1])],
        "verdict": row["verdict"],
    }


def _user_target_row(entry: float, row: dict, wished: float | None) -> dict:
    price = _money(wished) if wished is not None else _money(entry * (1.0 + float(row["pct"])))
    return {
        "price": price,
        "pct": _pct_points(float(row["pct"]) * 100.0),
        "prob": _prob(float(row["prob"])),
        "days_median": row.get("days_median"),
        "verdict": row["verdict"],
    }


def _is_rung(pct: float) -> bool:
    return any(math.isclose(pct, rung, rel_tol=0.0, abs_tol=1e-9) for rung in RUNGS)


def _row_for_pct(rows: list[dict], pct: float) -> dict:
    for row in rows:
        if math.isclose(float(row["pct"]), pct, rel_tol=0.0, abs_tol=1e-9):
            return row
    # Matching a rung was not duplicated; look it up from the fixed set.
    for row in rows:
        if _is_rung(row["pct"]) and math.isclose(float(row["pct"]), pct, rel_tol=0.0, abs_tol=1e-9):
            return row
    raise ValueError("Wished target row is missing from the ladder.")


def _nth_trading_day_after(start: date, n: int) -> date:
    if n < 1:
        raise ValueError("n must be a positive integer.")
    # Look ahead far enough for holidays and weekends.
    end = start + timedelta(days=max(n * 3, 40))
    sessions = trading_days(start + timedelta(days=1), end)
    if len(sessions) < n:
        end = start + timedelta(days=n * 6)
        sessions = trading_days(start + timedelta(days=1), end)
    if len(sessions) < n:
        raise PriceError("Could not find enough NSE trading days for the sell-by date.")
    stamp = sessions[n - 1]
    return stamp.date() if hasattr(stamp, "date") else stamp.tz_convert("UTC").date()


def _session_date(value) -> date:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC")
    return stamp.date()


def _as_of_date(value: date | None) -> date:
    if value is None:
        return datetime.now(IST).date()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.date()
        return value.astimezone(IST).date()
    if isinstance(value, date):
        return value
    raise ValueError("as_of must be a date.")


def _require_query(query: object) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Type a stock name or ticker.")
    return query.strip()


def _optional_rupees(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive number of rupees.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a positive number of rupees.")
    return number


def _money(value: float) -> float:
    return round(float(value), 2)


def _pct_points(value: float) -> float:
    return round(float(value), 1)


def _prob(value: float) -> float:
    return round(float(value), 4)


def _assert_section3(payload: dict) -> None:
    missing = [key for key in _SECTION3_KEYS if key not in payload]
    if missing:
        raise RuntimeError(f"Output is missing keys: {', '.join(missing)}.")


def format_card(payload: dict) -> str:
    """Plain-text trade card for the CLI."""
    price = payload["price"]
    plan = payload["plan"]
    lines = [
        f"{payload['query']} → {payload['ticker']}",
        f"As of {payload['as_of']}",
        (
            f"Price ₹{price['current']:.2f}  "
            f"high ₹{price['day_high']:.2f}  "
            f"low ₹{price['day_low']:.2f}  "
            f"prev ₹{price['prev_close']:.2f}"
        ),
        f"Action: {plan['action']}"
        + (f" — {payload['skip_reason']}" if payload.get("skip_reason") else ""),
        (
            f"Buy ₹{plan['entry']:.2f}  "
            f"Stop ₹{plan['stop_loss']:.2f} ({plan['stop_pct']}%)  "
            f"Window {plan['window_days']} sessions"
        ),
    ]
    if plan.get("recommended_target") is not None:
        lines.append(
            f"Recommended ₹{plan['recommended_target']:.2f} "
            f"(+{plan['recommended_target_pct']}%)  "
            f"sell by day {plan['sell_by_day']} ({plan['sell_by_date']})"
        )
    else:
        lines.append("Recommended —")

    lines.append("Target  Gain  Prob  Days  Verdict")
    for row in payload["targets"]:
        days = "—" if row["days_median"] is None else str(row["days_median"])
        if row.get("days_p25_p75"):
            days = f"{row['days_p25_p75'][0]}–{row['days_p25_p75'][1]}"
        lines.append(
            f"₹{row['price']:.2f}  +{row['pct']}%  "
            f"{row['prob'] * 100:.1f}%  {days}  {row['verdict']}"
        )
    if payload.get("user_target"):
        row = payload["user_target"]
        days = "—" if row["days_median"] is None else str(row["days_median"])
        lines.append(
            f"Your target ₹{row['price']:.2f}  +{row['pct']}%  "
            f"{row['prob'] * 100:.1f}%  {days}  {row['verdict']}"
        )
    lines.append(gap_risk_note())
    lines.append(payload["disclaimer"])
    return "\n".join(lines)


def format_pick_list(query: str, matches: tuple[Candidate, ...]) -> str:
    lines = [f"Several matches for {query!r}. Pick one:", ""]
    for row in matches:
        lines.append(f"  {row.name}  {row.ticker}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NSE swing research card (numbers only).")
    parser.add_argument("query", help="Company name or NSE ticker")
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Wished target price in rupees",
    )
    args = parser.parse_args(argv)
    try:
        payload = analyze(args.query, user_target=args.target)
    except AmbiguousQuery as exc:
        print(format_pick_list(exc.query, exc.matches))
        return 2
    except (ResolveError, PriceError, ValueError) as exc:
        print(str(exc))
        return 1
    print(format_card(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
