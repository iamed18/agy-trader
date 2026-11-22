from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Any, ClassVar

from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)


class _NYSEHolidayCalendar(AbstractHolidayCalendar):
    rules: ClassVar[list[Any]] = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_NYSE_CAL = _NYSEHolidayCalendar()


@lru_cache(maxsize=64)
def nyse_holidays_for_year(year: int) -> frozenset[date]:
    """Return cached NYSE market holiday dates for the given calendar year."""
    hols = _NYSE_CAL.holidays(start=f"{year}-01-01", end=f"{year}-12-31")
    return frozenset(ts.date() for ts in hols)


def is_nyse_trading_day(d: date) -> bool:
    """Return True if d is a weekday and not an NYSE holiday."""
    return d.weekday() < 5 and d not in nyse_holidays_for_year(d.year)


def nyse_trading_days_apart(d1: date, d2: date) -> int:
    """Count NYSE trading sessions strictly separating two dates (symmetric, >= 0).

    For two trading days, consecutive sessions (e.g., Thu & Fri, or Fri & Tue after a
    Monday NYSE holiday) return 1. If an event date falls on a weekend/holiday,
    the immediately preceding and following trading sessions both have distance 0
    (treated conservatively so blackout_days=0 still blocks adjacent sessions).
    """
    if d1 == d2:
        return 0
    lo, hi = (d1, d2) if d1 < d2 else (d2, d1)
    between = 0
    cur = lo + timedelta(days=1)
    while cur < hi:
        if is_nyse_trading_day(cur):
            between += 1
        cur += timedelta(days=1)
    if is_nyse_trading_day(lo) and is_nyse_trading_day(hi):
        return between + 1
    return between
