from __future__ import annotations

import logging
import math
from datetime import date

import httpx
import pandas as pd
import yfinance as yf

from agy_trader.core.models import EarningsEvent, MacroPoint
from agy_trader.data.providers import EarningsProvider, MacroProvider

logger = logging.getLogger(__name__)

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES_IDS: tuple[str, ...] = ("T10Y2Y", "BAMLH0A0HYM2", "VIXCLS", "DGS3MO")


class FREDMacroProvider(MacroProvider):
    """FRED macroeconomic series provider."""

    def __init__(
        self,
        api_key: str,
        client: httpx.Client | None = None,
        base_url: str = FRED_OBSERVATIONS_URL,
        series_ids: tuple[str, ...] = FRED_SERIES_IDS,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key.strip()
        self._external_client = client
        self.base_url = base_url
        self.series_ids = series_ids
        self._timeout = timeout

    def fetch_macro(
        self,
        start_date: date,
        end_date: date,
    ) -> list[MacroPoint]:
        if end_date < start_date:
            raise ValueError("end_date must be >= start_date")
        if not self.api_key:
            return []

        points: list[MacroPoint] = []
        own_client = self._external_client is None
        client = self._external_client or httpx.Client(timeout=self._timeout)

        try:
            for series_id in self.series_ids:
                params = {
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "observation_start": start_date.isoformat(),
                    "observation_end": end_date.isoformat(),
                }
                response = client.get(self.base_url, params=params)
                response.raise_for_status()
                payload = response.json()
                observations = (
                    payload.get("observations", []) if isinstance(payload, dict) else []
                )
                for obs in observations:
                    if not isinstance(obs, dict):
                        continue
                    raw_val = str(obs.get("value", "")).strip()
                    raw_date = str(obs.get("date", "")).strip()
                    if not raw_date or not raw_val or raw_val == ".":
                        continue
                    try:
                        val = float(raw_val)
                    except ValueError:
                        continue
                    if not math.isfinite(val):
                        continue
                    obs_date = date.fromisoformat(raw_date)
                    if obs_date < start_date or obs_date > end_date:
                        continue
                    points.append(
                        MacroPoint(
                            as_of_date=obs_date,
                            series_id=series_id,
                            value=val,
                        )
                    )
        finally:
            if own_client:
                client.close()

        points.sort(key=lambda p: (p.series_id, p.as_of_date))
        return points


class YFinanceEarningsProvider(EarningsProvider):
    """Best-effort earnings calendar provider backed by yfinance."""

    def __init__(self, limit: int = 40) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self.limit = limit

    def fetch_earnings(self, symbols: list[str]) -> list[EarningsEvent]:
        clean_symbols = list(
            dict.fromkeys(sym.strip().upper() for sym in symbols if sym.strip())
        )
        events_by_key: dict[tuple[str, date], EarningsEvent] = {}

        for sym in clean_symbols:
            try:
                ticker = yf.Ticker(sym)
                df = ticker.get_earnings_dates(limit=self.limit)
                if df is None or df.empty:
                    continue
                for idx_val in df.index:
                    ts = pd.Timestamp(idx_val)
                    if pd.isna(ts):
                        continue
                    edate = ts.date()
                    events_by_key[(sym, edate)] = EarningsEvent(
                        symbol=sym,
                        earnings_date=edate,
                    )
            except Exception as exc:  # noqa: BLE001 - best-effort per symbol
                logger.warning("Failed to fetch earnings dates for %s: %s", sym, exc)
                continue

        return sorted(
            events_by_key.values(),
            key=lambda ev: (ev.symbol, ev.earnings_date),
        )
