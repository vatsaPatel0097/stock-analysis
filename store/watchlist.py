"""Append-only watchlist snapshots. Freeze the card; never edit that JSON."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from store.calls import DEFAULT_PATH

_SCHEMA = """
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


def add_snapshot(payload: dict, path: Path | str | None = None) -> int:
    """Insert one frozen card. Returns the new row id. Same ticker next week is a new row."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict.")
    ticker = payload.get("ticker")
    if not isinstance(ticker, str) or not ticker:
        raise ValueError("payload must include a ticker.")
    price = payload.get("price") or {}
    current = price.get("current")
    if current is None:
        raise ValueError("payload must include price.current.")
    try:
        price_at_add = float(current)
    except (TypeError, ValueError) as exc:
        raise ValueError("payload.price.current must be a number.") from exc
    if not (price_at_add > 0.0):
        raise ValueError("payload.price.current must be a positive number.")

    plan = payload.get("plan") or {}
    action = plan.get("action")
    if action not in ("BUY", "WAIT", "SKIP"):
        raise ValueError("payload.plan.action must be BUY, WAIT, or SKIP.")

    entry = _optional_money(plan.get("entry"), "entry")
    stop_loss = _optional_money(plan.get("stop_loss"), "stop_loss")
    recommended_target = _optional_money(plan.get("recommended_target"), "recommended_target")
    sell_by_day = _optional_int(plan.get("sell_by_day"), "sell_by_day")
    sell_by_date = plan.get("sell_by_date")
    if sell_by_date is not None and not isinstance(sell_by_date, str):
        raise ValueError("payload.plan.sell_by_date must be a string or null.")

    db = _connect(path)
    try:
        added_at = datetime.now(timezone.utc).isoformat()
        cursor = db.execute(
            "INSERT INTO watchlist ("
            "added_at, ticker, price_at_add, action, entry, stop_loss, "
            "recommended_target, sell_by_day, sell_by_date, payload"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                added_at,
                ticker,
                price_at_add,
                action,
                entry,
                stop_loss,
                recommended_target,
                sell_by_day,
                sell_by_date,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        db.commit()
        return int(cursor.lastrowid)
    finally:
        db.close()


def list_snapshots(path: Path | str | None = None) -> list[dict]:
    """Return frozen snapshots oldest first."""
    db = _connect(path)
    try:
        rows = db.execute(
            "SELECT id, added_at, ticker, price_at_add, action, entry, stop_loss, "
            "recommended_target, sell_by_day, sell_by_date, payload "
            "FROM watchlist ORDER BY id ASC"
        ).fetchall()
    finally:
        db.close()

    out: list[dict] = []
    for row in rows:
        out.append(
            {
                "id": row[0],
                "added_at": row[1],
                "ticker": row[2],
                "price_at_add": row[3],
                "action": row[4],
                "entry": row[5],
                "stop_loss": row[6],
                "recommended_target": row[7],
                "sell_by_day": row[8],
                "sell_by_date": row[9],
                "payload": json.loads(row[10]),
            }
        )
    return out


def _optional_money(value: object, label: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload.plan.{label} must be a number or null.") from exc
    return number


def _optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload.plan.{label} must be an integer or null.") from exc


def _connect(path: Path | str | None) -> sqlite3.Connection:
    target = Path(path) if path is not None else DEFAULT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(target))
    connection.execute(_SCHEMA)
    return connection
