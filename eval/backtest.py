"""Pure-code backtest over 2025→. No LLM. Does not tune thresholds.

Calls ``analyze()`` with news/events/explainer stubs into a scratch database,
resolves outcomes, and writes a report file. Never writes the live ``calls.db``.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from analyze import analyze
from config import WINDOW_DAYS
from data.resolve import Candidate
from eval.report import DISCLAIMER, build_report
from eval.resolve import resolve_due
from eval.universe import FROZEN_ON, SURVIVORSHIP_NOTE, UNIVERSE

IST = ZoneInfo("Asia/Kolkata")

DEFAULT_START = date(2025, 1, 1)
DEFAULT_REPORT_PATH = Path("cache") / "backtest" / "report.json"
# Need enough history for SMA200 and sample_size before the first entry.
_MIN_HISTORY = 220


def run_backtest(
    universe: Sequence[str],
    price_loader: Callable[[str], pd.DataFrame],
    *,
    start: date = DEFAULT_START,
    report_path: Path | str | None = None,
    calls_path: Path | str,
    entry_dates: Sequence[date] | None = None,
    stride: int = 1,
) -> dict:
    """Run the numbers-only path on ``universe`` and write a report file.

    ``price_loader(ticker)`` returns the full OHLCV frame. Each entry session
    gets a slice through that day so later bars cannot change the card.
    ``stride`` keeps every Nth eligible session (1 = every session).
    """
    if stride < 1:
        raise ValueError("stride must be a positive integer.")
    out_path = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    scratch = Path(calls_path)
    if scratch.resolve() == Path("calls.db").resolve():
        raise ValueError("Backtest must use a scratch database, not the live calls.db.")

    if scratch.exists():
        scratch.unlink()

    last_entry: date | None = None
    analyzed = 0
    errors = 0

    for ticker in universe:
        try:
            full = price_loader(ticker)
        except Exception:
            errors += 1
            print(f"skip {ticker}: load failed", flush=True)
            continue
        if not isinstance(full, pd.DataFrame) or full.empty:
            errors += 1
            print(f"skip {ticker}: empty frame", flush=True)
            continue

        sessions = _entry_sessions(full, start=start, forced=entry_dates, stride=stride)
        print(f"{ticker}: {len(sessions)} entry sessions", flush=True)
        for entry_day in sessions:
            sliced = _slice_through(full, entry_day)
            if len(sliced) < _MIN_HISTORY:
                continue
            try:
                analyze(
                    ticker,
                    as_of=entry_day,
                    price_loader=_fixed_loader(sliced),
                    calls_path=scratch,
                    events=[],
                    headline_fetcher=_empty_headlines,
                    completer=_null_completer,
                    searcher=_identity_search,
                )
                analyzed += 1
                last_entry = entry_day if last_entry is None else max(last_entry, entry_day)
            except Exception as exc:
                errors += 1
                print(f"skip {ticker} {entry_day}: {exc}", flush=True)
                continue
        print(f"{ticker}: analyzed so far {analyzed}", flush=True)

    resolve_as_of = _as_of_date(None)
    resolve_due(
        scratch,
        as_of=resolve_as_of,
        price_loader=price_loader,
    )
    base = build_report(scratch)
    report = {
        "period": {
            "start": start.isoformat(),
            "end": None if last_entry is None else last_entry.isoformat(),
            "as_of": resolve_as_of.isoformat(),
        },
        "universe_size": len(universe),
        "universe_frozen_on": FROZEN_ON,
        "stride": stride,
        "analyzed": analyzed,
        "load_errors": errors,
        "resolved": base["resolved"],
        "counts": base["counts"],
        "buy": base["buy"],
        "calibration": base["calibration"],
        "loss_distribution": base["loss_distribution"],
        "no_rules_changed_on_this_period": True,
        "survivorship_note": SURVIVORSHIP_NOTE,
        "disclaimer": DISCLAIMER,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def _entry_sessions(
    frame: pd.DataFrame,
    *,
    start: date,
    forced: Sequence[date] | None,
    stride: int = 1,
) -> list[date]:
    dates = [_session_date(stamp) for stamp in frame.index]
    if forced is not None:
        available = set(dates)
        return [day for day in forced if day in available]

    out: list[date] = []
    for i, day in enumerate(dates):
        if day < start:
            continue
        if i + 1 < _MIN_HISTORY:
            continue
        forward = len(dates) - (i + 1)
        if forward < WINDOW_DAYS:
            continue
        out.append(day)
    if stride > 1:
        out = out[::stride]
    return out


def _slice_through(frame: pd.DataFrame, entry_day: date) -> pd.DataFrame:
    mask = [_session_date(stamp) <= entry_day for stamp in frame.index]
    return frame.loc[mask].copy()


def _fixed_loader(frame: pd.DataFrame):
    def load(ticker: str, *, as_of=None):
        return frame.copy()

    return load


def _session_date(value) -> date:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC")
    return stamp.date()


def _as_of_date(value: date | None) -> date:
    if value is None:
        return datetime.now(IST).date()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.date()
        return value.astimezone(IST).date()
    if isinstance(value, date):
        return value
    raise ValueError("as_of must be a date.")


def _empty_headlines(*args, **kwargs) -> list:
    return []


def _null_completer(*args, **kwargs):
    return None


def _identity_search(query: str) -> list[Candidate]:
    """Accept a typed ``SYMBOL.NS`` without calling Yahoo."""
    text = query.strip().upper()
    if text.endswith(".NS") and len(text) > 3:
        return [Candidate(name=text[:-3], ticker=text)]
    return []


def main() -> None:
    from data.prices import fetch_daily

    cache: dict[str, pd.DataFrame] = {}

    def loader(ticker: str) -> pd.DataFrame:
        if ticker not in cache:
            cache[ticker] = fetch_daily(ticker, period="max")
        return cache[ticker]

    scratch = Path("cache") / "backtest" / "calls.db"
    report = run_backtest(
        UNIVERSE,
        loader,
        start=DEFAULT_START,
        report_path=DEFAULT_REPORT_PATH,
        calls_path=scratch,
        # Every 5th session (~weekly). Overlapping daily windows are not
        # independent; stride=1 remains available for a full daily pass.
        stride=5,
    )
    print(json.dumps(
        {
            "report_path": str(DEFAULT_REPORT_PATH),
            "analyzed": report["analyzed"],
            "resolved": report["resolved"],
            "buy_win_rate": report["buy"]["win_rate"],
            "mean_abs_error": report["calibration"]["mean_abs_error"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
