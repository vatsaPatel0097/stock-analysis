"""Watchlist marks: latest price and one-shot 15-day outcome.

The frozen snapshot row is never rewritten. Marks live in a side table.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from eval.resolve import resolve_path
from store.calls import DEFAULT_PATH
from store.watchlist import list_snapshots

IST = ZoneInfo("Asia/Kolkata")

_MARKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist_marks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL UNIQUE,
    checked_at TEXT NOT NULL,
    latest_price REAL NOT NULL,
    change_pct REAL NOT NULL,
    outcome TEXT,
    exit_day INTEGER,
    fill_price REAL,
    realised_pct REAL
);
"""

_WATCHLIST_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    added_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    price_at_add REAL NOT NULL,
    action TEXT NOT NULL,
    entry REAL,
    stop_loss REAL,
    recommended_target REAL,
    sell_by_day INTEGER,
    sell_by_date TEXT,
    payload TEXT NOT NULL
);
"""


def refresh_marks(
    path: Path | str | None = None,
    *,
    as_of: date | None = None,
    price_loader: Callable[[str], pd.DataFrame] | None = None,
) -> list[dict]:
    """Refresh latest price / change for every snapshot; store outcome once when due.

    Does not rewrite ``watchlist.payload``. A later pass may replace latest price
    and change_pct but must not replace an outcome already stored.
    """
    day = _as_of_date(as_of)
    loader = price_loader if price_loader is not None else _default_price_loader
    db_path = Path(path) if path is not None else DEFAULT_PATH
    snapshots = list_snapshots(db_path)
    updated: list[dict] = []

    for snap in snapshots:
        snapshot_id = int(snap["id"])
        try:
            frame = loader(snap["ticker"])
        except Exception:
            continue
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        latest = _latest_close(frame, day)
        if latest is None:
            continue
        price_at_add = float(snap["price_at_add"])
        change_pct = round((latest - price_at_add) / price_at_add * 100.0, 1)

        existing = _get_mark(db_path, snapshot_id)
        outcome_fields: dict | None = None
        if existing is None or existing.get("outcome") is None:
            result = resolve_path(snap["payload"], frame)
            if result is not None:
                outcome_fields = {
                    "outcome": result["outcome"],
                    "exit_day": int(result["exit_day"]),
                    "fill_price": result["fill_price"],
                    "realised_pct": float(result["realised_pct"]),
                }

        mark = _upsert_mark(
            db_path,
            snapshot_id=snapshot_id,
            latest_price=latest,
            change_pct=change_pct,
            outcome_fields=outcome_fields,
            keep_outcome=existing is not None and existing.get("outcome") is not None,
            existing=existing,
        )
        updated.append(mark)
    return updated


def list_memory(path: Path | str | None = None) -> list[dict]:
    """Return each snapshot plus its mark. Mark fields are null until refreshed."""
    db_path = Path(path) if path is not None else DEFAULT_PATH
    snapshots = list_snapshots(db_path)
    out: list[dict] = []
    for snap in snapshots:
        mark = _get_mark(db_path, int(snap["id"]))
        row = dict(snap)
        if mark is None:
            row.update(
                {
                    "checked_at": None,
                    "latest_price": None,
                    "change_pct": None,
                    "outcome": None,
                    "exit_day": None,
                    "fill_price": None,
                    "realised_pct": None,
                }
            )
        else:
            row.update(
                {
                    "checked_at": mark["checked_at"],
                    "latest_price": mark["latest_price"],
                    "change_pct": mark["change_pct"],
                    "outcome": mark["outcome"],
                    "exit_day": mark["exit_day"],
                    "fill_price": mark["fill_price"],
                    "realised_pct": mark["realised_pct"],
                }
            )
        out.append(row)
    return out


def _upsert_mark(
    path: Path,
    *,
    snapshot_id: int,
    latest_price: float,
    change_pct: float,
    outcome_fields: dict | None,
    keep_outcome: bool,
    existing: dict | None,
) -> dict:
    checked_at = datetime.now(timezone.utc).isoformat()
    if keep_outcome and existing is not None:
        outcome = existing["outcome"]
        exit_day = existing["exit_day"]
        fill_price = existing["fill_price"]
        realised_pct = existing["realised_pct"]
    elif outcome_fields is not None:
        outcome = outcome_fields["outcome"]
        exit_day = outcome_fields["exit_day"]
        fill_price = outcome_fields["fill_price"]
        realised_pct = outcome_fields["realised_pct"]
    else:
        outcome = None
        exit_day = None
        fill_price = None
        realised_pct = None

    db = _connect(path)
    try:
        if existing is None:
            cursor = db.execute(
                "INSERT INTO watchlist_marks ("
                "snapshot_id, checked_at, latest_price, change_pct, "
                "outcome, exit_day, fill_price, realised_pct"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot_id,
                    checked_at,
                    latest_price,
                    change_pct,
                    outcome,
                    exit_day,
                    fill_price,
                    realised_pct,
                ),
            )
            mark_id = int(cursor.lastrowid)
        else:
            db.execute(
                "UPDATE watchlist_marks SET checked_at = ?, latest_price = ?, "
                "change_pct = ?, outcome = ?, exit_day = ?, fill_price = ?, "
                "realised_pct = ? WHERE snapshot_id = ?",
                (
                    checked_at,
                    latest_price,
                    change_pct,
                    outcome,
                    exit_day,
                    fill_price,
                    realised_pct,
                    snapshot_id,
                ),
            )
            mark_id = int(existing["id"])
        db.commit()
    finally:
        db.close()

    return {
        "id": mark_id,
        "snapshot_id": snapshot_id,
        "checked_at": checked_at,
        "latest_price": latest_price,
        "change_pct": change_pct,
        "outcome": outcome,
        "exit_day": exit_day,
        "fill_price": fill_price,
        "realised_pct": realised_pct,
    }


def _get_mark(path: Path | str, snapshot_id: int) -> dict | None:
    db = _connect(path)
    try:
        row = db.execute(
            "SELECT id, snapshot_id, checked_at, latest_price, change_pct, "
            "outcome, exit_day, fill_price, realised_pct "
            "FROM watchlist_marks WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
    finally:
        db.close()
    if row is None:
        return None
    return {
        "id": row[0],
        "snapshot_id": row[1],
        "checked_at": row[2],
        "latest_price": row[3],
        "change_pct": row[4],
        "outcome": row[5],
        "exit_day": row[6],
        "fill_price": row[7],
        "realised_pct": row[8],
    }


def _latest_close(frame: pd.DataFrame, as_of: date) -> float | None:
    prices = _require_ohlc(frame)
    chosen = None
    for stamp, bar in prices.iterrows():
        if _session_date(stamp) <= as_of:
            chosen = float(bar["Close"])
        else:
            break
    return chosen


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


def _default_price_loader(ticker: str) -> pd.DataFrame:
    from data.prices import fetch_daily

    return fetch_daily(ticker, period="max")


def _connect(path: Path | str | None) -> sqlite3.Connection:
    target = Path(path) if path is not None else DEFAULT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(target))
    connection.execute(_WATCHLIST_SCHEMA)
    connection.execute(_MARKS_SCHEMA)
    return connection
