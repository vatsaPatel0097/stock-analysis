"""Append-only call log. No network."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import store.calls as calls_mod
from store.calls import append_call, list_calls


def _payload(ticker: str, action: str) -> dict:
    return {
        "query": ticker.replace(".NS", ""),
        "ticker": ticker,
        "plan": {"action": action, "entry": 100.0},
        "disclaimer": "Educational research only. Not investment advice.",
    }


class CallsTests(unittest.TestCase):
    def test_two_appends_are_two_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.db"
            first = _payload("RELIANCE.NS", "BUY")
            second = _payload("INFY.NS", "SKIP")
            append_call(first, path)
            append_call(second, path)
            rows = list_calls(path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["ticker"], "RELIANCE.NS")
        self.assertEqual(rows[0]["action"], "BUY")
        self.assertEqual(rows[0]["payload"]["plan"]["entry"], 100.0)
        self.assertEqual(rows[1]["ticker"], "INFY.NS")
        self.assertEqual(rows[1]["action"], "SKIP")

    def test_module_has_no_update_or_delete_api(self):
        names = [name for name in dir(calls_mod) if not name.startswith("_")]
        forbidden = [name for name in names if "update" in name.lower() or "delete" in name.lower()]
        self.assertEqual(forbidden, [])
        self.assertIn("append_call", names)
        self.assertIn("list_calls", names)


if __name__ == "__main__":
    unittest.main()
