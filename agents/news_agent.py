"""News agent: headlines → sentiment + short summary. Numbers stay in code."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Callable

from agents.client import LlmCapError, complete
from agents.schemas import NEWS_JSON_SCHEMA, NewsModel

Completer = Callable[..., dict[str, Any] | None]


def run_news(
    ticker: str,
    headlines: list[dict[str, str]],
    *,
    as_of: date,
    completer: Completer | None = None,
    cache_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    """Return §3 ``news`` object, or None when the model fails.

    Sources always come from ``headlines``. Invented model URLs are ignored.
    """
    if not headlines:
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "You summarise Indian NSE stock news for a research tool. "
                "Return JSON with sentiment (positive|neutral|negative) and "
                "a summary of at most three short sentences. "
                "Use only the headlines provided. Do not invent prices, "
                "percentages, probabilities, or event dates."
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
            "news",
            messages,
            NEWS_JSON_SCHEMA,
            cache_dir=cache_dir,
        )
    except LlmCapError:
        return None
    except Exception:
        return None

    if raw is None:
        return None
    try:
        model = NewsModel.model_validate(raw)
    except Exception:
        return None

    sources = [
        {
            "title": str(row.get("title") or ""),
            "url": str(row.get("url") or ""),
            "published": str(row.get("published") or ""),
        }
        for row in headlines
        if row.get("title")
    ]
    return {
        "sentiment": model.sentiment,
        "summary": model.summary.strip(),
        "sources": sources,
    }
