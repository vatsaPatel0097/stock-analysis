"""NSE name and ticker resolver. No prices and no analytics."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.&-]{0,30}$")
_SEARCH_LIMIT = 8
_NSE_EXCHANGES = frozenset({"NSI", "NSE", "XNSE"})


class ResolveError(Exception):
    """The name or ticker could not be resolved to an NSE listing."""


@dataclass(frozen=True)
class Candidate:
    """One NSE listing the user can be shown."""

    name: str
    ticker: str


@dataclass(frozen=True)
class Listing:
    """A row in the local NSE symbol map."""

    name: str
    ticker: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolveResult:
    """A unique ticker, or a pick list when more than one listing matches.

    ``ticker`` is set only for a single match. Several matches leave it
    ``None`` so a caller cannot treat the first row as the answer.
    """

    query: str
    ticker: str | None
    matches: tuple[Candidate, ...]


def _listed(name: str, ticker: str, *aliases: str) -> Listing:
    return Listing(name=name, ticker=ticker, aliases=aliases)


# Small offline map. Search covers everything else. TATAMOTORS was renamed
# to TMPV in October 2025; the commercial-vehicle company is TMCV.
NSE_LISTINGS: tuple[Listing, ...] = (
    _listed("Reliance Industries", "RELIANCE.NS", "Reliance", "RIL"),
    _listed("Tata Consultancy Services", "TCS.NS", "TCS"),
    _listed("Tata Steel", "TATASTEEL.NS"),
    _listed("Tata Power", "TATAPOWER.NS"),
    _listed("Tata Consumer Products", "TATACONSUM.NS"),
    _listed("Tata Chemicals", "TATACHEM.NS"),
    _listed("Tata Motors Limited", "TMCV.NS"),
    _listed("Tata Motors Passenger Vehicles", "TMPV.NS"),
    _listed("Infosys", "INFY.NS", "Infy"),
    _listed("HDFC Bank", "HDFCBANK.NS"),
    _listed("HDFC Life Insurance", "HDFCLIFE.NS"),
    _listed("ICICI Bank", "ICICIBANK.NS", "ICICI"),
    _listed("State Bank of India", "SBIN.NS", "SBI", "State Bank"),
    _listed("SBI Life Insurance", "SBILIFE.NS"),
    _listed("Bharti Airtel", "BHARTIARTL.NS", "Airtel"),
    _listed("ITC", "ITC.NS"),
    _listed("Larsen & Toubro", "LT.NS", "L&T", "Larsen"),
    _listed("Hindustan Unilever", "HINDUNILVR.NS", "HUL"),
    _listed("Bajaj Finance", "BAJFINANCE.NS"),
    _listed("Bajaj Finserv", "BAJAJFINSV.NS"),
    _listed("Bajaj Auto", "BAJAJ-AUTO.NS"),
    _listed("Asian Paints", "ASIANPAINT.NS"),
    _listed("Maruti Suzuki", "MARUTI.NS", "Maruti"),
    _listed("Mahindra & Mahindra", "M&M.NS", "M&M"),
    _listed("Axis Bank", "AXISBANK.NS"),
    _listed("Kotak Mahindra Bank", "KOTAKBANK.NS", "Kotak", "Kotak Bank"),
    _listed("HCL Technologies", "HCLTECH.NS", "HCL"),
    _listed("Wipro", "WIPRO.NS"),
    _listed("Adani Enterprises", "ADANIENT.NS"),
    _listed("Adani Ports", "ADANIPORTS.NS"),
    _listed("Sun Pharmaceutical", "SUNPHARMA.NS", "Sun Pharma"),
)


def resolve(
    query: str,
    *,
    searcher=None,
    symbols: Sequence[Listing] | None = None,
    cache_dir: Path | str | None = None,
    as_of: date | None = None,
) -> ResolveResult:
    """Resolve a company name or NSE ticker to ``SYMBOL.NS``.

    ``Reliance``, ``RELIANCE``, and ``RELIANCE.NS`` are the same listing.
    Two or more matches return a pick list and leave ``ticker`` unset.
    An unknown name raises ``ResolveError`` and does not invent a ticker.

    Listings missing from ``symbols`` (the local map) are looked up with
    ``searcher``, which defaults to Yahoo search. Hits are cached for the
    IST ``as_of`` date. Tests pass a fake ``searcher`` and never call Yahoo.
    """
    shown = _require_query(query)
    listings = _listings(symbols)
    local = _match_local(_norm(shown), listings)
    if local:
        return _pack(shown, [_from_listing(row) for row in local])

    found = _search_cached(shown, searcher, cache_dir, as_of)
    if not found:
        raise ResolveError(f"No NSE stock matches {shown!r}.")
    return _pack(shown, found)


def search_nse(query: str) -> list[Candidate]:
    """Search Yahoo for NSE equities. Not used by tests."""
    import yfinance as yf

    try:
        found = yf.Search(
            query,
            max_results=_SEARCH_LIMIT,
            news_count=0,
            lists_count=0,
            include_cb=False,
            include_nav_links=False,
            include_research=False,
            include_cultural_assets=False,
            enable_fuzzy_query=False,
            recommended=_SEARCH_LIMIT,
        )
    except ResolveError:
        raise
    except Exception as exc:
        raise ResolveError(f"Could not look up {query!r}.") from exc
    return candidates_from_quotes(getattr(found, "quotes", None) or [])


def candidates_from_quotes(quotes) -> list[Candidate]:
    """Keep NSE equities from a Yahoo quote list. Drop ``.BO`` and other types."""
    if not quotes:
        return []
    if isinstance(quotes, (str, bytes)):
        raise ResolveError("Search returned an unexpected result.")
    found: list[Candidate] = []
    seen: set[str] = set()
    for quote in quotes:
        ticker = _ticker_from_quote(quote)
        if ticker is None or ticker in seen:
            continue
        seen.add(ticker)
        found.append(Candidate(name=_quote_name(quote, ticker), ticker=ticker))
    return found


def _pack(query: str, matches: list[Candidate]) -> ResolveResult:
    ordered = tuple(sorted(matches, key=lambda item: (item.ticker, item.name)))
    ticker = ordered[0].ticker if len(ordered) == 1 else None
    return ResolveResult(query=query, ticker=ticker, matches=ordered)


def _from_listing(row: Listing) -> Candidate:
    return Candidate(name=row.name, ticker=row.ticker)


def _match_local(norm: str, listings: Sequence[Listing]) -> list[Listing]:
    """Exact ticker, then exact name or alias, then a shared prefix.

    An exact alias wins over a longer name that merely starts with the same
    letters: ``SBI`` is State Bank of India, not SBI Life. A prefix that hits
    two listings stays a pick list: ``Tata`` and ``Tata Motors``.
    """
    typed = _symbol_ticker(norm)
    if typed is not None:
        by_ticker = [row for row in listings if row.ticker == typed]
        if by_ticker:
            return _dedupe(by_ticker)

    exact = [
        row
        for row in listings
        if _norm(row.name) == norm or any(_norm(alias) == norm for alias in row.aliases)
    ]
    if exact:
        return _dedupe(exact)

    if len(norm) < 2:
        return []
    return _dedupe(row for row in listings if _prefix_hit(norm, row))


def _prefix_hit(norm: str, row: Listing) -> bool:
    for label in (row.name, *row.aliases):
        text = _norm(label)
        if text.startswith(norm):
            return True
        if " " in norm:
            continue
        words = text.replace("&", " ").replace("-", " ").split()
        if any(word.startswith(norm) for word in words):
            return True
    return False


def _search_cached(
    shown: str,
    searcher,
    cache_dir: Path | str | None,
    as_of: date | None,
) -> list[Candidate]:
    day = _as_of_date(as_of)
    path = _cache_path(cache_dir, _norm(shown), day)
    if path.is_file():
        return _read_cache(path)

    source = searcher if searcher is not None else search_nse
    try:
        raw = source(shown)
    except ResolveError:
        raise
    except AssertionError:
        raise
    except Exception as exc:
        raise ResolveError(f"Could not look up {shown!r}.") from exc

    found = _filter_typed_ticker(_norm(shown), _clean_candidates(raw))
    _write_cache(path, found)
    return found


def _filter_typed_ticker(norm: str, found: list[Candidate]) -> list[Candidate]:
    """Do not swap in a different company when the user typed ``SYMBOL.NS``."""
    if not norm.endswith(".NS"):
        return found
    typed = _symbol_ticker(norm)
    if typed is None:
        return []
    same = [item for item in found if item.ticker == typed]
    if same:
        return same
    if len(found) >= 2:
        return found
    return []


def _clean_candidates(items) -> list[Candidate]:
    if items is None or isinstance(items, (str, bytes)):
        raise ResolveError("Search returned an unexpected result.")
    try:
        iterator = iter(items)
    except TypeError as exc:
        raise ResolveError("Search returned an unexpected result.") from exc

    found: list[Candidate] = []
    seen: set[str] = set()
    for item in iterator:
        if not isinstance(item, Candidate):
            raise ResolveError("Search returned an unexpected result.")
        ticker = item.ticker.strip().upper()
        if _symbol_ticker(ticker) != ticker or ticker in seen:
            continue
        name = " ".join(item.name.split()) or ticker
        seen.add(ticker)
        found.append(Candidate(name=name, ticker=ticker))
    return found


def _ticker_from_quote(quote: object) -> str | None:
    if not isinstance(quote, dict) or not _is_equity(quote):
        return None
    raw = quote.get("symbol")
    if not isinstance(raw, str):
        return None
    symbol = raw.strip().upper()
    if symbol.endswith(".NS"):
        return _symbol_ticker(symbol)
    if _nse_exchange(quote):
        return _symbol_ticker(symbol)
    return None


def _is_equity(quote: dict) -> bool:
    kind = quote.get("quoteType")
    if kind is None:
        return True
    return isinstance(kind, str) and kind.strip().upper() == "EQUITY"


def _nse_exchange(quote: dict) -> bool:
    for key in ("exchange", "exchDisp"):
        value = quote.get(key)
        if isinstance(value, str) and value.strip().upper() in _NSE_EXCHANGES:
            return True
    return False


def _quote_name(quote: dict, ticker: str) -> str:
    for key in ("longname", "shortname"):
        value = quote.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ticker


def _listings(symbols: Sequence[Listing] | None) -> tuple[Listing, ...]:
    if symbols is None:
        return NSE_LISTINGS
    rows = tuple(symbols)
    for row in rows:
        if not isinstance(row, Listing) or not row.name.strip():
            raise ResolveError("Symbol map is invalid.")
        if _symbol_ticker(row.ticker) != row.ticker:
            raise ResolveError("Symbol map is invalid.")
    return rows


def _dedupe(rows: Iterable[Listing]) -> list[Listing]:
    seen: set[str] = set()
    out: list[Listing] = []
    for row in rows:
        if row.ticker in seen:
            continue
        seen.add(row.ticker)
        out.append(row)
    return out


def _require_query(query: object) -> str:
    if not isinstance(query, str):
        raise ResolveError("Type a stock name or ticker.")
    shown = query.strip()
    if not shown:
        raise ResolveError("Type a stock name or ticker.")
    return shown


def _norm(text: str) -> str:
    return " ".join(text.strip().upper().split())


def _symbol_ticker(norm: str) -> str | None:
    base = norm[:-3] if norm.endswith(".NS") else norm
    if not base or _SYMBOL_RE.fullmatch(base) is None:
        return None
    return f"{base}.NS"


def _as_of_date(value: date | None) -> date:
    if value is None:
        return datetime.now(IST).date()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.date()
        return value.astimezone(IST).date()
    if isinstance(value, date):
        return value
    raise ResolveError("as_of must be a date.")


def _cache_path(cache_dir: Path | str | None, norm: str, day: date) -> Path:
    root = Path(cache_dir) if cache_dir is not None else Path("cache") / "resolve"
    return root / _safe_key(norm) / f"{day.isoformat()}.json"


def _safe_key(norm: str) -> str:
    cleaned = []
    for char in norm:
        if char.isalnum() or char in {".", "&", "-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    key = "".join(cleaned).strip("._")
    return (key or "query")[:80]


def _read_cache(path: Path) -> list[Candidate]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResolveError("Cached lookup is unreadable.") from exc
    if not isinstance(payload, list):
        raise ResolveError("Cached lookup is unreadable.")
    rows: list[Candidate] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ResolveError("Cached lookup is unreadable.")
        name = item.get("name")
        ticker = item.get("ticker")
        if not isinstance(name, str) or not isinstance(ticker, str):
            raise ResolveError("Cached lookup is unreadable.")
        rows.append(Candidate(name=name, ticker=ticker))
    return _clean_candidates(rows)


def _write_cache(path: Path, matches: list[Candidate]) -> None:
    payload = [{"name": item.name, "ticker": item.ticker} for item in matches]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
