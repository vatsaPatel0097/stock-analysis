"""Pydantic schemas for LLM agents. No network."""

from __future__ import annotations

import unittest
from datetime import date

from pydantic import ValidationError

from agents.schemas import EventItem, EventsModel, ExplainerModel, NewsModel


class NewsSchemaTests(unittest.TestCase):
    def test_valid_news(self):
        row = NewsModel.model_validate(
            {"sentiment": "neutral", "summary": "Quiet session for the stock."}
        )
        self.assertEqual(row.sentiment, "neutral")

    def test_extra_field_rejected(self):
        with self.assertRaises(ValidationError):
            NewsModel.model_validate(
                {
                    "sentiment": "positive",
                    "summary": "Up on news.",
                    "price": 100.0,
                }
            )


class EventsSchemaTests(unittest.TestCase):
    def test_valid_events(self):
        row = EventsModel.model_validate(
            {"events": [{"kind": "earnings", "date": "2026-10-03"}]}
        )
        self.assertEqual(row.events[0].date, date(2026, 10, 3))
        self.assertEqual(row.events[0].kind, "earnings")

    def test_extra_field_on_item_rejected(self):
        with self.assertRaises(ValidationError):
            EventItem.model_validate(
                {
                    "kind": "earnings",
                    "date": "2026-10-03",
                    "inside_window": True,
                }
            )

    def test_bad_date_rejected(self):
        with self.assertRaises(ValidationError):
            EventsModel.model_validate(
                {"events": [{"kind": "agm", "date": "not-a-date"}]}
            )


class ExplainerSchemaTests(unittest.TestCase):
    def test_text_only(self):
        row = ExplainerModel.model_validate({"text": "Buy setup looks ordinary."})
        self.assertIn("ordinary", row.text)

    def test_numeric_field_rejected(self):
        with self.assertRaises(ValidationError):
            ExplainerModel.model_validate({"text": "ok", "prob": 0.5})


if __name__ == "__main__":
    unittest.main()
