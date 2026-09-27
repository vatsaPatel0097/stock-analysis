"""OpenRouter client: cache, cap, retry. Fake transport only — no network."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from agents.client import LlmCapError, complete, daily_count
from config import LLM_DAILY_CAP

_DAY = date(2026, 9, 27)
_TICKER = "RELIANCE.NS"


def _ok_response(payload: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(payload),
                }
            }
        ]
    }


class ClientCacheTests(unittest.TestCase):
    def test_second_call_same_day_is_cache_hit(self):
        calls: list[str] = []

        def transport(url, body, headers):
            calls.append(body["model"])
            return _ok_response({"sentiment": "neutral", "summary": "ok"})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = complete(
                _TICKER,
                _DAY,
                "news",
                [{"role": "user", "content": "hi"}],
                cache_dir=root,
                transport=transport,
                api_key="test-key",
                model="model-a",
                fallback_model="model-b",
            )
            second = complete(
                _TICKER,
                _DAY,
                "news",
                [{"role": "user", "content": "hi again"}],
                cache_dir=root,
                transport=transport,
                api_key="test-key",
                model="model-a",
                fallback_model="model-b",
            )
            counted = daily_count(_DAY, cache_dir=root)

        self.assertEqual(first, {"sentiment": "neutral", "summary": "ok"})
        self.assertEqual(second, first)
        self.assertEqual(len(calls), 1)
        self.assertEqual(counted, 1)


class ClientCapTests(unittest.TestCase):
    def test_951st_call_is_refused(self):
        def transport(url, body, headers):
            return _ok_response({"ok": True})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counts = root / "counts"
            counts.mkdir(parents=True)
            (counts / f"{_DAY.isoformat()}.txt").write_text(
                str(LLM_DAILY_CAP), encoding="utf-8"
            )
            with self.assertRaises(LlmCapError):
                complete(
                    _TICKER,
                    _DAY,
                    "news",
                    [{"role": "user", "content": "hi"}],
                    cache_dir=root,
                    transport=transport,
                    api_key="test-key",
                    model="model-a",
                    fallback_model="model-b",
                )
            self.assertEqual(daily_count(_DAY, cache_dir=root), LLM_DAILY_CAP)


class ClientRetryTests(unittest.TestCase):
    def test_parse_failure_then_valid_json_succeeds(self):
        payloads = [
            {"choices": [{"message": {"content": "not-json"}}]},
            _ok_response({"ok": True}),
        ]
        calls = {"n": 0}

        def transport(url, body, headers):
            i = calls["n"]
            calls["n"] += 1
            return payloads[i]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = complete(
                _TICKER,
                _DAY,
                "news",
                [{"role": "user", "content": "hi"}],
                cache_dir=root,
                transport=transport,
                api_key="test-key",
                model="model-a",
                fallback_model="model-b",
            )
            counted = daily_count(_DAY, cache_dir=root)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(calls["n"], 2)
        self.assertEqual(counted, 2)

    def test_two_bad_payloads_return_none(self):
        def transport(url, body, headers):
            return {"choices": [{"message": {"content": "still-bad"}}]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = complete(
                _TICKER,
                _DAY,
                "news",
                [{"role": "user", "content": "hi"}],
                cache_dir=root,
                transport=transport,
                api_key="test-key",
                model="model-a",
                fallback_model="model-a",
            )
            counted = daily_count(_DAY, cache_dir=root)

        self.assertIsNone(result)
        self.assertEqual(counted, 2)

    def test_primary_failure_uses_fallback_model(self):
        models: list[str] = []

        def transport(url, body, headers):
            models.append(body["model"])
            if body["model"] == "model-a":
                raise RuntimeError("primary down")
            return _ok_response({"via": "fallback"})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = complete(
                _TICKER,
                _DAY,
                "events",
                [{"role": "user", "content": "hi"}],
                cache_dir=root,
                transport=transport,
                api_key="test-key",
                model="model-a",
                fallback_model="model-b",
            )

        self.assertEqual(result, {"via": "fallback"})
        self.assertEqual(models, ["model-a", "model-b"])

    def test_missing_api_key_returns_none_without_counting(self):
        calls = {"n": 0}

        def transport(url, body, headers):
            calls["n"] += 1
            return _ok_response({"ok": True})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = complete(
                _TICKER,
                _DAY,
                "news",
                [{"role": "user", "content": "hi"}],
                cache_dir=root,
                transport=transport,
                api_key="",
                model="model-a",
                fallback_model="model-b",
            )
            counted = daily_count(_DAY, cache_dir=root)

        self.assertIsNone(result)
        self.assertEqual(calls["n"], 0)
        self.assertEqual(counted, 0)


if __name__ == "__main__":
    unittest.main()
