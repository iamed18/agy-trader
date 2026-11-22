from __future__ import annotations

import logging
import math
import re
from datetime import date

import httpx
import pandas as pd
import yfinance as yf

from agy_trader.core.models import EarningsEvent, MacroPoint
from agy_trader.data.providers import EarningsProvider, EtfHoldingsProvider, MacroProvider

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


_HOLDING_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


class YFinanceEtfHoldingsProvider(EtfHoldingsProvider):
    """Best-effort ETF top-holdings provider backed by yfinance funds_data."""

    @staticmethod
    def _resolve_weight_column(df: pd.DataFrame) -> str | None:
        lowered = {str(col).strip().lower(): str(col) for col in df.columns}
        for candidate in ("holding percent", "holdingpercent", "weight", "pct"):
            if candidate in lowered:
                return lowered[candidate]
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                return str(col)
        return None

    @staticmethod
    def _resolve_symbol_column(df: pd.DataFrame) -> str | None:
        if not isinstance(df.index, pd.RangeIndex) and not pd.api.types.is_integer_dtype(df.index):
            return None
        lowered = {str(col).strip().lower(): str(col) for col in df.columns}
        for candidate in ("symbol", "holding", "ticker", "holding symbol"):
            if candidate in lowered:
                return lowered[candidate]
        return None

    def fetch_etf_holdings(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        clean_symbols = list(
            dict.fromkeys(sym.strip().upper() for sym in symbols if sym.strip())
        )
        holdings_by_etf: dict[str, dict[str, float]] = {}

        for sym in clean_symbols:
            try:
                ticker = yf.Ticker(sym)
                funds_data = getattr(ticker, "funds_data", None)
                df = getattr(funds_data, "top_holdings", None) if funds_data is not None else None
                if df is None or df.empty:
                    holdings_by_etf[sym] = {}
                    continue

                weight_col = self._resolve_weight_column(df)
                if weight_col is None:
                    logger.warning("Missing weight column in ETF holdings for %s", sym)
                    continue

                symbol_col = self._resolve_symbol_column(df)
                if (
                    symbol_col is None
                    and (isinstance(df.index, pd.RangeIndex) or pd.api.types.is_integer_dtype(df.index))
                ):
                    logger.warning("Integer/RangeIndex without symbol column for %s", sym)
                    continue

                raw_weights: dict[str, float] = {}
                invalid_row = False
                for idx_val, row in df.iterrows():
                    raw_sym = row[symbol_col] if symbol_col is not None else idx_val
                    if raw_sym is None or pd.isna(raw_sym):
                        continue
                    h_sym = str(raw_sym).strip().upper()
                    if not h_sym or h_sym == "NAN" or _HOLDING_TICKER_RE.fullmatch(h_sym) is None:
                        logger.warning("Invalid holding ticker %r for %s; skipping ETF", raw_sym, sym)
                        invalid_row = True
                        break
                    try:
                        w = float(row[weight_col])
                    except (TypeError, ValueError):
                        invalid_row = True
                        break
                    if not math.isfinite(w) or w <= 0.0:
                        invalid_row = True
                        break
                    raw_weights[h_sym] = raw_weights.get(h_sym, 0.0) + w

                if invalid_row or not raw_weights:
                    continue

                max_w = max(raw_weights.values())
                total_w = sum(raw_weights.values())
                if (max_w > 1.05 and total_w > 1.05) or total_w > 1.5:
                    norm_weights = {k: v / 100.0 for k, v in raw_weights.items()}
                else:
                    norm_weights = dict(raw_weights)

                if (
                    any(not math.isfinite(w) or w <= 0.0 or w > 1.0 for w in norm_weights.values())
                    or sum(norm_weights.values()) > 1.05
                ):
                    logger.warning(
                        "Corrupt normalized ETF holdings weights for %s (max=%.4f, sum=%.4f); skipping",
                        sym,
                        max(norm_weights.values()),
                        sum(norm_weights.values()),
                    )
                    continue

                holdings_by_etf[sym] = {k: float(v) for k, v in norm_weights.items()}
            except Exception as exc:  # noqa: BLE001 - best-effort per ETF symbol
                logger.warning("Failed to fetch ETF holdings for %s: %s", sym, exc)
                continue

        return holdings_by_etf
