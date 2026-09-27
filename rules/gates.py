"""BUY / WAIT / SKIP rules from project.md R2. One function per rule.

Each function returns a pass, or SKIP / WAIT with a plain reason.
The caller supplies numbers and event dates. This module does not fetch
prices and does not call a model. Any SKIP wins over WAIT; otherwise WAIT;
otherwise the caller may say BUY.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime

from config import (
    ATR_PCT_MAX,
    ATR_PCT_MIN,
    CHASE_PCT,
    MIN_AVG_TRADED_VALUE,
    MIN_SAMPLE_SIZE,
    MIN_TARGET_PROB,
    STOP_PCT,
)

GAP_RISK_NOTE = (
    "A 2% stop does not cap the loss at 2%. "
    "A gap-down can fill at −4% or worse."
)

_EVENT_LABELS = {
    "earnings": "Earnings",
    "results": "Results",
    "agm": "AGM",
    "dividend": "Dividend record date",
}


@dataclass(frozen=True)
class ScheduledEvent:
    """One company date supplied by the caller. Phase 2 fills this from the events agent."""

    kind: str
    on: date


@dataclass(frozen=True)
class GateResult:
    """``action`` is None when the rule does not fire."""

    rule: str
    action: str | None
    reason: str | None
    note: str | None = None


def gate_stop_width(stop_pct: float) -> GateResult:
    """R2.1. ``stop_pct`` is a fraction. Wider than 2% is a SKIP."""
    value = _require_fraction(stop_pct, "stop percent")
    if value > STOP_PCT:
        return _skip("R2.1", "Stop is wider than 2%.")
    return _ok("R2.1")


def gate_atr_too_high(atr_pct: float) -> GateResult:
    """R2.2. ``atr_pct`` is percentage points. Above 2.0 is a SKIP."""
    value = _require_percent_points(atr_pct, "ATR percent")
    if value > ATR_PCT_MAX:
        return _skip("R2.2", "ATR is above 2% of price, so a normal day can hit the stop.")
    return _ok("R2.2")


def gate_atr_too_low(atr_pct: float) -> GateResult:
    """R2.3. ``atr_pct`` is percentage points. Below 0.4 is a SKIP."""
    value = _require_percent_points(atr_pct, "ATR percent")
    if value < ATR_PCT_MIN:
        return _skip("R2.3", "ATR is below 0.4% of price, so a move to a real target is unlikely.")
    return _ok("R2.3")


def gate_events(
    events: list[ScheduledEvent] | None,
    *,
    window_start: date,
    window_end: date,
) -> GateResult:
    """R2.4. SKIP when earnings, results, AGM, or a dividend record date is inside the window."""
    start = _as_date(window_start, "window_start")
    end = _as_date(window_end, "window_end")
    if end < start:
        raise ValueError("window_end is before window_start.")
    if not events:
        return _ok("R2.4")

    inside: list[tuple[date, str]] = []
    for event in events:
        kind = _event_kind(event.kind)
        on = _as_date(event.on, "event date")
        if start <= on <= end:
            inside.append((on, kind))
    if not inside:
        return _ok("R2.4")
    on, kind = min(inside, key=lambda item: item[0])
    label = _EVENT_LABELS[kind]
    return _skip("R2.4", f"{label} on {on.isoformat()} is inside the window.")


def gate_liquidity(avg_traded_value: float) -> GateResult:
    """R2.5. Average daily traded value under ₹5 crore is a SKIP."""
    value = _require_rupees(avg_traded_value)
    if value < MIN_AVG_TRADED_VALUE:
        return _skip("R2.5", "Average daily traded value is under ₹5 crore.")
    return _ok("R2.5")


def gate_min_probability(
    probs: list[float],
    user_target_prob: float | None = None,
) -> GateResult:
    """R2.6. SKIP when no ladder rung has probability >= 0.35.

    A wished target under 0.35 does not SKIP the plan. The note says so.
    """
    if isinstance(probs, bool) or not isinstance(probs, (list, tuple)):
        raise ValueError("probs must be a list of probabilities.")
    cleaned = [_require_probability(prob) for prob in probs]
    note = None
    if user_target_prob is not None:
        wished = _require_probability(user_target_prob)
        if wished < MIN_TARGET_PROB:
            note = "The wished target is below a 35% probability."
    if any(prob >= MIN_TARGET_PROB for prob in cleaned):
        return GateResult("R2.6", None, None, note)
    return GateResult(
        "R2.6",
        "SKIP",
        "No target has a probability of at least 35%.",
        note,
    )


def gate_sample_size(sample_size: int) -> GateResult:
    """R2.7. Fewer than 50 windows is a SKIP."""
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise ValueError("sample_size must be a non-negative integer.")
    if sample_size < 0:
        raise ValueError("sample_size must be a non-negative integer.")
    if sample_size < MIN_SAMPLE_SIZE:
        return _skip("R2.7", "Fewer than 50 similar days.")
    return _ok("R2.7")


def gate_chase(price: float, prev_close: float) -> GateResult:
    """R2.8. Price more than 1.5% above the last close is WAIT, not SKIP."""
    live = _require_rupees(price)
    previous = _require_rupees(prev_close)
    if live / previous - 1.0 > CHASE_PCT:
        return GateResult("R2.8", "WAIT", "Price is more than 1.5% above the last close.")
    return _ok("R2.8")


def gap_risk_note() -> str:
    """R2.9. This sentence is part of every output."""
    return GAP_RISK_NOTE


def _ok(rule: str) -> GateResult:
    return GateResult(rule, None, None)


def _skip(rule: str, reason: str) -> GateResult:
    return GateResult(rule, "SKIP", reason)


def _event_kind(kind: str) -> str:
    if not isinstance(kind, str):
        raise ValueError("Event kind must be earnings, results, agm, or dividend.")
    name = kind.strip().lower().replace("-", "_").replace(" ", "_")
    if name in {"dividend_record", "dividend_record_date"}:
        name = "dividend"
    if name not in _EVENT_LABELS:
        raise ValueError("Event kind must be earnings, results, agm, or dividend.")
    return name


def _as_date(value: date | datetime, label: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise ValueError(f"{label} must be a date.")


def _require_fraction(value: float, label: str) -> float:
    number = _require_finite(value, label)
    if number < 0.0:
        raise ValueError(f"{label} must be zero or positive.")
    return number


def _require_percent_points(value: float, label: str) -> float:
    number = _require_finite(value, label)
    if number < 0.0:
        raise ValueError(f"{label} must be zero or positive.")
    return number


def _require_rupees(value: float) -> float:
    number = _require_finite(value, "amount")
    if number <= 0.0:
        raise ValueError("amount must be positive.")
    return number


def _require_probability(value: float) -> float:
    number = _require_finite(value, "probability")
    if number < 0.0 or number > 1.0:
        raise ValueError("probability must be from 0 to 1.")
    return number


def _require_finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a number.")
    return number
