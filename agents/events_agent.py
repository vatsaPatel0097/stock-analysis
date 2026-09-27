"""Events agent: filings / headlines → earnings, results, AGM, dividend dates."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Callable

from agents.client import LlmCapError, complete
from agents.schemas import EVENTS_JSON_SCHEMA, EventsModel
from rules.gates import ScheduledEvent

Completer = Callable[..., dict[str, Any] | None]


def run_events(
    ticker: str,
    headlines: list[dict[str, str]],
    *,
    as_of: date,
    completer: Completer | None = None,
    cache_dir: Path | str | None = None,
) -> list[ScheduledEvent]:
    """Return scheduled events. Parse failure or empty input → []."""
    if not headlines:
        return []

    messages = [
        {
            "role": "system",
            "content": (
                "Extract upcoming corporate event dates for an NSE stock. "
                "Return JSON with an events array. Each event has kind "
                "(earnings|results|agm|dividend) and date as YYYY-MM-DD. "
                "Use only the text provided. Omit uncertain items. "
                "Do not invent prices or probabilities."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "ticker": ticker,
                    "headlines": [
                        {
                            "title": row.get("title", ""),
                            "published": row.get("published", ""),
                        }
                        for row in headlines
                    ],
                },
                ensure_ascii=True,
            ),
        },
    ]

    call = completer if completer is not None else complete
    try:
        raw = call(
            ticker,
            as_of,
            "events",
            messages,
            EVENTS_JSON_SCHEMA,
            cache_dir=cache_dir,
        )
    except LlmCapError:
        return []
    except Exception:
        return []

    if raw is None:
        return []
    try:
        model = EventsModel.model_validate(raw)
    except Exception:
        return []

    rows: list[ScheduledEvent] = []
    for item in model.events:
        rows.append(ScheduledEvent(kind=item.kind, on=item.date))
    return rows
