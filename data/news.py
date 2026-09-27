"""RSS headline fetch for NSE names. No LLM and no analytics."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from typing import Callable
from zoneinfo import ZoneInfo

import httpx

IST = ZoneInfo("Asia/Kolkata")

# Public market RSS. Overridable in tests via ``feeds`` / ``fetcher``.
DEFAULT_FEEDS: tuple[str, ...] = (
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml",
)

Fetcher = Callable[[str], str]
_WORD = re.compile(r"[A-Za-z0-9.&-]+")


def fetch_headlines(
    ticker: str,
    company_name: str | None = None,
    *,
    as_of: date | None = None,
    feeds: tuple[str, ...] | list[str] | None = None,
    fetcher: Fetcher | None = None,
    limit: int = 12,
) -> list[dict[str, str]]:
    """Return recent headlines that mention the ticker or company name.

    A bad feed becomes an empty contribution, not an exception. ``as_of`` is
    reserved for callers that cache by day; filtering is by text match.
    """
    del as_of  # reserved for future per-day cache; matching is text-only for now
    symbol = _bare_symbol(ticker)
    names = _match_tokens(symbol, company_name)
    urls = tuple(feeds) if feeds is not None else DEFAULT_FEEDS
    load = fetcher if fetcher is not None else _http_get
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for url in urls:
        try:
            xml_text = load(url)
        except Exception:
            continue
        if not xml_text:
            continue
        for item in _parse_rss(xml_text):
            blob = f"{item['title']} {item.get('summary', '')}"
            if not _mentions(blob, names):
                continue
            key = item["url"] or item["title"]
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "title": item["title"],
                    "url": item["url"],
                    "published": item["published"],
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def _http_get(url: str) -> str:
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        response = client.get(
            url,
            headers={"User-Agent": "StockSwingAdvisor/0.1 (research; RSS)"},
        )
        response.raise_for_status()
        return response.text


def _parse_rss(xml_text: str) -> list[dict[str, str]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    items: list[dict[str, str]] = []
    for node in root.iter():
        tag = _local(node.tag).lower()
        if tag not in {"item", "entry"}:
            continue
        title = _child_text(node, "title")
        link = _child_text(node, "link")
        if not link:
            link = _attr_link(node)
        summary = _child_text(node, "description") or _child_text(node, "summary")
        published = (
            _child_text(node, "pubDate")
            or _child_text(node, "published")
            or _child_text(node, "updated")
        )
        if not title:
            continue
        items.append(
            {
                "title": title.strip(),
                "url": (link or "").strip(),
                "summary": (summary or "").strip(),
                "published": _normalise_published(published),
            }
        )
    return items


def _child_text(node: ET.Element, name: str) -> str:
    for child in list(node):
        if _local(child.tag).lower() == name.lower():
            if child.text and child.text.strip():
                return child.text.strip()
            # Atom <link href="..."/> often has empty text.
            href = child.attrib.get("href")
            if href:
                return href.strip()
    return ""


def _attr_link(node: ET.Element) -> str:
    for child in list(node):
        if _local(child.tag).lower() == "link":
            href = child.attrib.get("href")
            if href:
                return href.strip()
    return ""


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _normalise_published(value: str | None) -> str:
    if not value or not value.strip():
        return ""
    text = value.strip()
    try:
        stamp = parsedate_to_datetime(text)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=IST)
        return stamp.astimezone(IST).isoformat()
    except (TypeError, ValueError, IndexError, OverflowError):
        pass
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=IST)
        return stamp.astimezone(IST).isoformat()
    except ValueError:
        return text


def _bare_symbol(ticker: str) -> str:
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker is required.")
    symbol = ticker.strip().upper()
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]
    return symbol


def _match_tokens(symbol: str, company_name: str | None) -> list[str]:
    tokens = [symbol.lower()]
    if company_name and company_name.strip():
        tokens.append(company_name.strip().lower())
        # Also match significant words from the company name.
        for part in _WORD.findall(company_name):
            if len(part) >= 4:
                tokens.append(part.lower())
    # Unique, longest first so "reliance industries" beats "ance".
    uniq: list[str] = []
    seen: set[str] = set()
    for token in sorted(tokens, key=len, reverse=True):
        if token not in seen:
            seen.add(token)
            uniq.append(token)
    return uniq


def _mentions(text: str, tokens: list[str]) -> bool:
    blob = text.lower()
    for token in tokens:
        if len(token) <= 3:
            # Short tickers: whole-word match.
            if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", blob):
                return True
        elif token in blob:
            return True
    return False
