"""Resolve logged calls after the 15-session window. Append-only outcomes.

Levels come from the logged card. Stop is checked before target on the same
bar. Gap fills use the open when the open is already through the level.
Historical call JSON is never rewritten.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from config import WINDOW_DAYS
from store.calls import DEFAULT_PATH, list_calls

IST = ZoneInfo("Asia/Kolkata")

_OUTCOMES_SCHEMA = """
CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id INTEGER NOT NULL UNIQUE,
    resolved_at TEXT NOT NULL,
    outcome TEXT NOT NULL,
    exit_day INTEGER NOT NULL,
    fill_price REAL,
    realised_pct REAL NOT NULL,
    worst_adverse_pct REAL NOT NULL
);
"""


def resolve_path(payload: dict, frame: pd.DataFrame) -> dict | None:
    """Walk the forward window for one call. ``None`` when not yet due.

    Uses ``plan.entry``, ``plan.stop_loss``, and ``plan.recommended_target``
    from the logged card. Does not recompute the plan.
    """
    plan = payload.get("plan") or {}
    entry = _require_positive(plan.get("entry"), "entry")
    stop = _require_positive(plan.get("stop_loss"), "stop_loss")
    target = plan.get("recommended_target")
    if target is not None:
        target = _require_positive(target, "recommended_target")
    window = int(plan.get("window_days") or WINDOW_DAYS)
    if window < 1:
        raise ValueError("window_days must be a positive integer.")

    entry_day = _entry_date(payload)
    prices = _require_ohlc(frame)
    entry_idx = _entry_index(prices.index, entry_day)
    if entry_idx is None:
        return None

    forward = prices.iloc[entry_idx + 1 : entry_idx + 1 + window]
    if len(forward) < window:
        return None

    min_low = entry
    for day_num, (_, bar) in enumerate(forward.iterrows(), start=1):
        open_px = float(bar["Open"])
        high = float(bar["High"])
        low = float(bar["Low"])
        if low < min_low:
            min_low = low

        stop_hit = low <= stop
        if stop_hit:
            fill = open_px if open_px <= stop else stop
            return _outcome(
                "stop",
                exit_day=day_num,
                fill_price=fill,
                entry=entry,
                min_low=min_low,
            )

        if target is not None and high >= target:
            fill = open_px if open_px >= target else target
            return _outcome(
                "target",
                exit_day=day_num,
                fill_price=fill,
                entry=entry,
                min_low=min_low,
            )

    last_close = float(forward["Close"].iloc[-1])
    return _outcome(
        "neither",
        exit_day=window,
        fill_price=last_close,
        entry=entry,
        min_low=min_low,
    )


def resolve_due(
    path: Path | str | None = None,
    *,
    as_of: date | None = None,
    price_loader: Callable[[str], pd.DataFrame] | None = None,
) -> list[dict]:
    """Resolve every call that has a complete forward window and no outcome yet.

    Returns the newly inserted outcome rows. A second pass is a no-op for
    calls already in the outcomes table. Does not rewrite call payloads.
    """
    day = _as_of_date(as_of)
    loader = price_loader if price_loader is not None else _default_price_loader
    db_path = Path(path) if path is not None else DEFAULT_PATH
    calls = list_calls(db_path)
    already = _resolved_call_ids(db_path)
    inserted: list[dict] = []

    for row in calls:
        call_id = int(row["id"])
        if call_id in already:
            continue
        payload = row["payload"]
        entry_day = _entry_date(payload)
        if entry_day > day:
            continue
        try:
            frame = loader(row["ticker"])
        except Exception:
            continue
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        result = resolve_path(payload, frame)
        if result is None:
            continue
        stored = _append_outcome(call_id, result, db_path)
        inserted.append(stored)
    return inserted


def list_outcomes(path: Path | str | None = None) -> list[dict]:
    """Return resolved outcomes oldest first."""
    db = _connect(path)
    try:
        rows = db.execute(
            "SELECT id, call_id, resolved_at, outcome, exit_day, "
            "fill_price, realised_pct, worst_adverse_pct "
            "FROM outcomes ORDER BY id ASC"
        ).fetchall()
    finally:
        db.close()
    return [
        {
            "id": row[0],
            "call_id": row[1],
            "resolved_at": row[2],
            "outcome": row[3],
            "exit_day": row[4],
            "fill_price": row[5],
            "realised_pct": row[6],
            "worst_adverse_pct": row[7],
        }
        for row in rows
    ]


def _outcome(
    label: str,
    *,
    exit_day: int,
    fill_price: float,
    entry: float,
    min_low: float,
) -> dict:
    realised = (fill_price - entry) / entry
    worst = (entry - min_low) / entry
    return {
        "outcome": label,
        "exit_day": exit_day,
        "fill_price": round(fill_price, 2),
        "realised_pct": realised,
        "worst_adverse_pct": worst,
    }


def _entry_date(payload: dict) -> date:
    raw = payload.get("as_of")
    if raw is None:
        raise ValueError("payload must include as_of.")
    stamp = pd.Timestamp(raw)
    if stamp.tzinfo is None:
        return stamp.date()
    return stamp.tz_convert(IST).date()


def _entry_index(index: pd.DatetimeIndex, entry_day: date) -> int | None:
    """Index of the last session on or before ``entry_day``."""
    if len(index) == 0:
        return None
    dates = [_session_date(stamp) for stamp in index]
    chosen = None
    for i, day in enumerate(dates):
        if day <= entry_day:
            chosen = i
        else:
            break
    return chosen


def _session_date(value) -> date:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC")
    return stamp.date()


def _require_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Price data must be a table.")
    needed = ("Open", "High", "Low", "Close")
    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise ValueError(f"Price table is missing {', '.join(missing)}.")
    out = frame.loc[:, list(needed)].copy()
    for column in needed:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=list(needed), how="any")
    if out.empty:
        raise ValueError("Price table has no usable bars.")
    return out


def _require_positive(value: object, label: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive number.") from exc
    if not (number > 0.0):
        raise ValueError(f"{label} must be a positive number.")
    return number


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


def _default_price_loader(ticker: str) -> pd.DataFrame:
    from data.prices import fetch_daily

    return fetch_daily(ticker, period="max")


def _resolved_call_ids(path: Path | str | None) -> set[int]:
    db = _connect(path)
    try:
        rows = db.execute("SELECT call_id FROM outcomes").fetchall()
    finally:
        db.close()
    return {int(row[0]) for row in rows}


def _append_outcome(call_id: int, result: dict, path: Path | str | None) -> dict:
    db = _connect(path)
    try:
        resolved_at = datetime.now(timezone.utc).isoformat()
        cursor = db.execute(
            "INSERT OR IGNORE INTO outcomes "
            "(call_id, resolved_at, outcome, exit_day, fill_price, "
            "realised_pct, worst_adverse_pct) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                call_id,
                resolved_at,
                result["outcome"],
                int(result["exit_day"]),
                result["fill_price"],
                float(result["realised_pct"]),
                float(result["worst_adverse_pct"]),
            ),
        )
        db.commit()
        if cursor.rowcount == 0:
            existing = db.execute(
                "SELECT id, call_id, resolved_at, outcome, exit_day, "
                "fill_price, realised_pct, worst_adverse_pct "
                "FROM outcomes WHERE call_id = ?",
                (call_id,),
            ).fetchone()
            return {
                "id": existing[0],
                "call_id": existing[1],
                "resolved_at": existing[2],
                "outcome": existing[3],
                "exit_day": existing[4],
                "fill_price": existing[5],
                "realised_pct": existing[6],
                "worst_adverse_pct": existing[7],
            }
        return {
            "id": int(cursor.lastrowid),
            "call_id": call_id,
            "resolved_at": resolved_at,
            "outcome": result["outcome"],
            "exit_day": int(result["exit_day"]),
            "fill_price": result["fill_price"],
            "realised_pct": float(result["realised_pct"]),
            "worst_adverse_pct": float(result["worst_adverse_pct"]),
        }
    finally:
        db.close()


def _connect(path: Path | str | None) -> sqlite3.Connection:
    target = Path(path) if path is not None else DEFAULT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(target))
    connection.execute(
        "CREATE TABLE IF NOT EXISTS calls ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "created_at TEXT NOT NULL, "
        "ticker TEXT NOT NULL, "
        "action TEXT NOT NULL, "
        "payload TEXT NOT NULL)"
    )
    connection.execute(_OUTCOMES_SCHEMA)
    return connection


def main() -> None:
    inserted = resolve_due()
    print(json.dumps({"resolved": len(inserted)}, indent=2))


if __name__ == "__main__":
    main()
