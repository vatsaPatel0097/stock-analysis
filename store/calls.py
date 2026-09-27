"""Append-only SQLite log of every BUY / WAIT / SKIP. No updates, no deletes."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path("calls.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""


def append_call(payload: dict, path: Path | str | None = None) -> int:
    """Insert one analysis JSON. Returns the new row id."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict.")
    ticker = payload.get("ticker")
    action = (payload.get("plan") or {}).get("action")
    if not isinstance(ticker, str) or not ticker:
        raise ValueError("payload must include a ticker.")
    if action not in ("BUY", "WAIT", "SKIP"):
        raise ValueError("payload.plan.action must be BUY, WAIT, or SKIP.")

    db = _connect(path)
    try:
        created = datetime.now(timezone.utc).isoformat()
        cursor = db.execute(
            "INSERT INTO calls (created_at, ticker, action, payload) VALUES (?, ?, ?, ?)",
            (created, ticker, action, json.dumps(payload, ensure_ascii=False)),
        )
        db.commit()
        return int(cursor.lastrowid)
    finally:
        db.close()


def list_calls(path: Path | str | None = None) -> list[dict]:
    """Return logged calls oldest first. Each item has id, created_at, ticker, action, payload."""
    db = _connect(path)
    try:
        rows = db.execute(
            "SELECT id, created_at, ticker, action, payload FROM calls ORDER BY id ASC"
        ).fetchall()
    finally:
        db.close()

    out: list[dict] = []
    for row in rows:
        out.append(
            {
                "id": row[0],
                "created_at": row[1],
                "ticker": row[2],
                "action": row[3],
                "payload": json.loads(row[4]),
            }
        )
    return out


def _connect(path: Path | str | None) -> sqlite3.Connection:
    target = Path(path) if path is not None else DEFAULT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(target))
    connection.execute(_SCHEMA)
    return connection
