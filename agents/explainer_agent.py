"""Explainer agent: plain words only. Digit check discards invented numbers."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Callable

from agents.client import LlmCapError, complete
from agents.schemas import EXPLAINER_JSON_SCHEMA, ExplainerModel

Completer = Callable[..., dict[str, Any] | None]

# Tokens like 15, 2.0, 0.58, 2026-10-08, 2950.10
_DIGIT_TOKEN = re.compile(
    r"(?<![A-Za-z])(?:\d{4}-\d{2}-\d{2}|\d+(?:\.\d+)?)(?![A-Za-z])"
)


def run_explainer(
    payload: dict[str, Any],
    *,
    as_of: date,
    completer: Completer | None = None,
    cache_dir: Path | str | None = None,
) -> str | None:
    """Return explanation prose, or None when the model invents a number."""
    ticker = str(payload.get("ticker") or "")
    if not ticker:
        return None

    news = payload.get("news")
    news_line = "news unavailable"
    if isinstance(news, dict) and news.get("summary"):
        news_line = str(news["summary"])

    # Strip LLM-owned fields so the model cannot echo its own prior words as "facts".
    for_model = {
        key: value
        for key, value in payload.items()
        if key not in {"explanation", "disclaimer"}
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Explain this NSE swing research card in plain words. "
                "Return JSON with a single field text. "
                "Cite only numbers that already appear in the JSON. "
                "Do not invent any price, percent, probability, day count, or date. "
                "Do not give investment advice."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"card": for_model, "news_note": news_line},
                ensure_ascii=True,
                default=str,
            ),
        },
    ]

    call = completer if completer is not None else complete
    try:
        raw = call(
            ticker,
            as_of,
            "explain",
            messages,
            EXPLAINER_JSON_SCHEMA,
            cache_dir=cache_dir,
        )
    except LlmCapError:
        return None
    except Exception:
        return None

    if raw is None:
        return None
    try:
        model = ExplainerModel.model_validate(raw)
    except Exception:
        return None

    text = model.text.strip()
    if not text:
        return None
    if not digits_already_in_json(text, payload):
        return None
    return text


def digits_already_in_json(text: str, payload: dict[str, Any]) -> bool:
    """True when every digit token in ``text`` already appears in the card JSON."""
    allowed = _allowed_tokens(payload)
    for match in _DIGIT_TOKEN.finditer(text):
        token = match.group(0)
        if token not in allowed:
            return False
    return True


def _allowed_tokens(payload: dict[str, Any]) -> set[str]:
    """Serialise numbers from the card (without explanation) into digit token forms."""
    clone = {key: value for key, value in payload.items() if key != "explanation"}
    blob = json.dumps(clone, ensure_ascii=True, default=str)
    tokens = set(_DIGIT_TOKEN.findall(blob))
    # Also allow bare integers that appear inside decimals already in the JSON
    # only when the exact token is present — no freestyle rounding.
    return tokens
