"""OpenRouter LLM client: cache, daily cap, retry once. No other module talks to OpenRouter."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import httpx

from config import (
    LLM_DAILY_CAP,
    OPENROUTER_FALLBACK_MODEL,
    OPENROUTER_MODEL,
    OPENROUTER_REASONING_EFFORT,
)

IST = ZoneInfo("Asia/Kolkata")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_CACHE = Path("cache") / "llm"
_ENV_LOADED = False

Transport = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any] | None]


class LlmCapError(Exception):
    """Daily request budget would be exceeded."""


def complete(
    ticker: str,
    day: date,
    step: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any] | None = None,
    *,
    cache_dir: Path | str | None = None,
    transport: Transport | None = None,
    api_key: str | None = None,
    model: str | None = None,
    fallback_model: str | None = None,
) -> dict[str, Any] | None:
    """Return parsed JSON for ``(ticker, day, step)``, or None on failure.

    Cache hits do not count against the daily cap. A missing API key returns
    None without counting. The call that would be number 951 is refused.
    """
    symbol = _require_ticker(ticker)
    session = _as_of_date(day)
    name = _require_step(step)
    root = _cache_root(cache_dir)

    cached = _read_cache(root, symbol, session, name)
    if cached is not None:
        return cached

    _load_dotenv()
    key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
    if not key or not str(key).strip():
        return None

    primary = model if model is not None else OPENROUTER_MODEL
    backup = fallback_model if fallback_model is not None else OPENROUTER_FALLBACK_MODEL
    send = transport if transport is not None else _http_transport

    parsed, served = _try_model(
        send,
        key=str(key).strip(),
        model=primary,
        messages=messages,
        schema=schema,
        root=root,
        session=session,
    )
    if parsed is not None:
        _write_cache(root, symbol, session, name, parsed, served or primary)
        return parsed

    if backup and backup != primary:
        parsed, served = _try_model(
            send,
            key=str(key).strip(),
            model=backup,
            messages=messages,
            schema=schema,
            root=root,
            session=session,
        )
        if parsed is not None:
            _write_cache(root, symbol, session, name, parsed, served or backup)
            return parsed
    return None


def _try_model(
    send: Transport,
    *,
    key: str,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any] | None,
    root: Path,
    session: date,
) -> tuple[dict[str, Any] | None, str | None]:
    """One model: up to two HTTP attempts. Count every attempt. Cap refuses before send."""
    for _ in range(2):
        _reserve_slot(root, session)
        body = _request_body(model, messages, schema)
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        try:
            raw = send(OPENROUTER_URL, body, headers)
        except LlmCapError:
            raise
        except Exception:
            return None, None
        if raw is None:
            return None, None
        last = _extract_json(raw)
        if last is not None:
            return last, _served_model(raw, model)
    return None, None


def _request_body(
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "reasoning": {"effort": OPENROUTER_REASONING_EFFORT},
    }
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "ssa_response",
                "strict": True,
                "schema": schema,
            },
        }
    return body


def _http_transport(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any] | None:
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=body, headers=headers)
            response.raise_for_status()
            return response.json()
    except Exception:
        return None


def _extract_json(raw: dict[str, Any]) -> dict[str, Any] | None:
    try:
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if isinstance(content, dict):
            return content
        if not isinstance(content, str) or not content.strip():
            return None
        text = content.strip()
        # Some models wrap JSON in fences.
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, ValueError, json.JSONDecodeError, AttributeError, IndexError, KeyError):
        return None
    return None


def _reserve_slot(root: Path, session: date) -> None:
    """Increment the daily counter. Refuse when the next call would be over the cap."""
    path = root / "counts" / f"{session.isoformat()}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = 0
    if path.is_file():
        try:
            current = int(path.read_text(encoding="utf-8").strip() or "0")
        except ValueError:
            current = 0
    if current >= LLM_DAILY_CAP:
        raise LlmCapError(f"Daily LLM cap of {LLM_DAILY_CAP} reached.")
    path.write_text(str(current + 1), encoding="utf-8")


def daily_count(day: date, *, cache_dir: Path | str | None = None) -> int:
    """How many counted LLM HTTP attempts happened on ``day`` (IST)."""
    root = _cache_root(cache_dir)
    path = root / "counts" / f"{_as_of_date(day).isoformat()}.txt"
    if not path.is_file():
        return 0
    try:
        return int(path.read_text(encoding="utf-8").strip() or "0")
    except ValueError:
        return 0


def _read_cache(root: Path, ticker: str, day: date, step: str) -> dict[str, Any] | None:
    path = _cache_path(root, ticker, day, step)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    body = data.get("body")
    if "model" in data and isinstance(body, dict):
        return body
    return data


def _write_cache(
    root: Path,
    ticker: str,
    day: date,
    step: str,
    payload: dict[str, Any],
    model: str,
) -> None:
    path = _cache_path(root, ticker, day, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"model": model, "body": payload}
    path.write_text(json.dumps(envelope, ensure_ascii=True, indent=2), encoding="utf-8")


def model_used(
    ticker: str,
    day: date,
    step: str,
    *,
    cache_dir: Path | str | None = None,
) -> str | None:
    """Model id that produced the cached step, or None when that step did not run."""
    root = _cache_root(cache_dir)
    path = _cache_path(root, _require_ticker(ticker), _as_of_date(day), _require_step(step))
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("model")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _served_model(raw: dict[str, Any], requested: str) -> str:
    served = raw.get("model") if isinstance(raw, dict) else None
    if isinstance(served, str) and served.strip():
        return served.strip()
    return requested


def _load_dotenv() -> None:
    """Load ``.env`` once. An existing environment variable wins."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _cache_path(root: Path, ticker: str, day: date, step: str) -> Path:
    safe = ticker.replace("/", "_")
    return root / safe / day.isoformat() / f"{step}.json"


def _cache_root(cache_dir: Path | str | None) -> Path:
    if cache_dir is None:
        return _DEFAULT_CACHE
    return Path(cache_dir)


def _require_ticker(ticker: str) -> str:
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker is required.")
    return ticker.strip().upper()


def _require_step(step: str) -> str:
    if not isinstance(step, str) or not step.strip():
        raise ValueError("step is required.")
    name = step.strip().lower()
    if not name.replace("_", "").isalnum():
        raise ValueError("step must be a simple name.")
    return name


def _as_of_date(value: date) -> date:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.date()
        return value.astimezone(IST).date()
    if isinstance(value, date):
        return value
    raise ValueError("day must be a date.")
