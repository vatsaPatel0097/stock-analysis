"""NSE trading sessions from the XNSE exchange calendar. No network."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pandas_market_calendars as mcal

EXCHANGE = "XNSE"


class CalendarError(Exception):
    """The exchange calendar could not answer the requested range."""


def trading_days(start: date | datetime | str, end: date | datetime | str) -> pd.DatetimeIndex:
    """Inclusive XNSE session dates, stored as UTC midnight.

    Weekends and NSE holidays come from the exchange calendar.
    """
    start_day = _coerce_bound(start)
    end_day = _coerce_bound(end)
    if end_day < start_day:
        raise CalendarError("end is before start.")
    calendar = mcal.get_calendar(EXCHANGE)
    sessions = calendar.valid_days(start_date=start_day, end_date=end_day)
    index = pd.DatetimeIndex(pd.to_datetime(sessions, utc=True)).tz_convert("UTC")
    return index.normalize()


def is_trading_day(day: date | datetime | str) -> bool:
    """True when ``day`` is an XNSE session."""
    return len(trading_days(day, day)) == 1


def _coerce_bound(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise CalendarError("start and end must be dates.")
