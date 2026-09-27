"""RSS news fetch and news/events agents. Fixture XML and fake completer only."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from agents.events_agent import run_events
from agents.news_agent import run_news
from data.news import fetch_headlines

_AS_OF = date(2026, 9, 27)

_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Markets</title>
    <item>
      <title>Reliance Industries gains on refining outlook</title>
      <link>https://example.com/ril-1</link>
      <pubDate>Fri, 26 Sep 2026 10:00:00 +0530</pubDate>
      <description>RIL saw buying interest.</description>
    </item>
    <item>
      <title>Infosys wins a deal</title>
      <link>https://example.com/infy-1</link>
      <pubDate>Fri, 26 Sep 2026 11:00:00 +0530</pubDate>
    </item>
    <item>
      <title>RELIANCE board to consider results on 3 Oct</title>
      <link>https://example.com/ril-results</link>
      <pubDate>Thu, 25 Sep 2026 09:00:00 +0530</pubDate>
    </item>
  </channel>
</rss>
"""

_BAD_RSS = "<not-xml"


class FetchHeadlinesTests(unittest.TestCase):
    def test_filters_by_company_and_symbol(self):
        def fetcher(url: str) -> str:
            return _RSS

        rows = fetch_headlines(
            "RELIANCE.NS",
            "Reliance Industries",
            as_of=_AS_OF,
            feeds=("https://example.com/feed.xml",),
            fetcher=fetcher,
        )
        titles = [row["title"] for row in rows]
        self.assertEqual(len(rows), 2)
        self.assertTrue(any("Reliance" in t for t in titles))
        self.assertTrue(any("RELIANCE" in t for t in titles))
        self.assertFalse(any("Infosys" in t for t in titles))
        self.assertEqual(rows[0]["url"], "https://example.com/ril-1")

    def test_broken_feed_returns_empty(self):
        def fetcher(url: str) -> str:
            return _BAD_RSS

        rows = fetch_headlines(
            "RELIANCE.NS",
            "Reliance",
            feeds=("https://example.com/bad.xml",),
            fetcher=fetcher,
        )
        self.assertEqual(rows, [])

    def test_fetcher_error_returns_empty(self):
        def fetcher(url: str) -> str:
            raise RuntimeError("network down")

        rows = fetch_headlines(
            "INFY.NS",
            "Infosys",
            feeds=("https://example.com/down.xml",),
            fetcher=fetcher,
        )
        self.assertEqual(rows, [])


class NewsAgentTests(unittest.TestCase):
    def test_valid_completer_builds_news_with_rss_sources(self):
        headlines = [
            {
                "title": "Reliance Industries gains",
                "url": "https://example.com/a",
                "published": "2026-09-26T10:00:00+05:30",
            }
        ]

        def completer(ticker, day, step, messages, schema, **kwargs):
            self.assertEqual(step, "news")
            return {"sentiment": "positive", "summary": "Refining outlook lifted shares."}

        with tempfile.TemporaryDirectory() as tmp:
            news = run_news(
                "RELIANCE.NS",
                headlines,
                as_of=_AS_OF,
                completer=completer,
                cache_dir=Path(tmp),
            )

        self.assertIsNotNone(news)
        assert news is not None
        self.assertEqual(news["sentiment"], "positive")
        self.assertEqual(news["sources"][0]["url"], "https://example.com/a")

    def test_broken_model_response_returns_none(self):
        headlines = [{"title": "Reliance", "url": "u", "published": ""}]

        def completer(ticker, day, step, messages, schema, **kwargs):
            return {"sentiment": "happy", "summary": "bad enum", "price": 1}

        news = run_news(
            "RELIANCE.NS",
            headlines,
            as_of=_AS_OF,
            completer=completer,
        )
        self.assertIsNone(news)


class EventsAgentTests(unittest.TestCase):
    def test_valid_events(self):
        headlines = [{"title": "Results on 3 Oct", "url": "u", "published": ""}]

        def completer(ticker, day, step, messages, schema, **kwargs):
            self.assertEqual(step, "events")
            return {"events": [{"kind": "results", "date": "2026-10-03"}]}

        rows = run_events(
            "RELIANCE.NS",
            headlines,
            as_of=_AS_OF,
            completer=completer,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].kind, "results")
        self.assertEqual(rows[0].on, date(2026, 10, 3))

    def test_broken_model_response_returns_empty(self):
        headlines = [{"title": "AGM", "url": "u", "published": ""}]

        def completer(ticker, day, step, messages, schema, **kwargs):
            return {"events": [{"kind": "agm", "date": "soon", "inside_window": True}]}

        rows = run_events(
            "RELIANCE.NS",
            headlines,
            as_of=_AS_OF,
            completer=completer,
        )
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
