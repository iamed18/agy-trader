from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from agy_trader.core.models import ensure_utc
from agy_trader.data.providers import MarketDataProvider
from agy_trader.storage.parquet_store import CANONICAL_BAR_COLUMNS

NY_TZ = ZoneInfo("America/New_York")

_TIMEFRAME_DELTAS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
}

_TIMEFRAME_ALIASES: dict[str, str] = {
    "1d": "1d",
    "1day": "1d",
    "day": "1d",
    "1h": "1h",
    "1hour": "1h",
    "hour": "1h",
    "1m": "1m",
    "1min": "1m",
    "minute": "1m",
    "5m": "5m",
    "5min": "5m",
    "15m": "15m",
    "15min": "15m",
    "30m": "30m",
    "30min": "30m",
    "1w": "1w",
    "1wk": "1w",
    "1month": "1month",
}


def _empty_canonical_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "vwap": pd.Series(dtype="float64"),
        }
    )


def _canonical_timeframe(timeframe: str) -> str:
    tf = timeframe.strip().lower()
    if tf in _TIMEFRAME_ALIASES:
        return _TIMEFRAME_ALIASES[tf]
    raise ValueError(f"Unsupported timeframe: {timeframe!r}")


def _parse_timeframe(timeframe: str) -> TimeFrame:
    canonical = _canonical_timeframe(timeframe)
    mapping: dict[str, TimeFrame] = {
        "1d": TimeFrame.Day,
        "1h": TimeFrame.Hour,
        "1m": TimeFrame.Minute,
        "5m": TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "1w": TimeFrame.Week,
        "1month": TimeFrame.Month,
    }
    return mapping[canonical]


def _extract_bar_field(bar: Any, name: str, default: Any = None) -> Any:
    if isinstance(bar, dict):
        return bar.get(name, default)
    return getattr(bar, name, default)


class AlpacaMarketDataProvider(MarketDataProvider):
    """Alpaca historical stock bars provider using alpaca-py."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        feed: Literal["iex", "sip"] = "iex",
        batch_size: int = 25,
        client: StockHistoricalDataClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.feed: Literal["iex", "sip"] = feed
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = batch_size
        self._client = client or StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )

    def _data_feed(self) -> DataFeed:
        return DataFeed.SIP if self.feed == "sip" else DataFeed.IEX

    def fetch_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        start_utc = ensure_utc(start)
        end_utc = ensure_utc(end)
        if end_utc < start_utc:
            raise ValueError("end must be >= start")

        clean_symbols = list(
            dict.fromkeys(sym.strip().upper() for sym in symbols if sym.strip())
        )
        if not clean_symbols:
            return {}

        canonical_tf = _canonical_timeframe(timeframe)
        alpaca_tf = _parse_timeframe(canonical_tf)
        feed_enum = self._data_feed()
        results: dict[str, pd.DataFrame] = {
            sym: _empty_canonical_frame() for sym in clean_symbols
        }

        if canonical_tf == "1d":
            ny_start_date = start_utc.astimezone(NY_TZ).date()
            request_start = datetime.combine(
                ny_start_date, time(0, 0), tzinfo=NY_TZ
            ).astimezone(UTC)
        elif canonical_tf in _TIMEFRAME_DELTAS:
            request_start = start_utc - _TIMEFRAME_DELTAS[canonical_tf]
        else:
            request_start = start_utc

        start_ts = pd.Timestamp(start_utc)
        end_ts = pd.Timestamp(end_utc)

        for idx in range(0, len(clean_symbols), self.batch_size):
            batch = clean_symbols[idx : idx + self.batch_size]
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=alpaca_tf,
                start=request_start,
                end=end_utc,
                adjustment=Adjustment.ALL,
                feed=feed_enum,
            )
            response = self._client.get_stock_bars(request)
            raw_data = getattr(response, "data", response)
            if not isinstance(raw_data, dict):
                continue

            for raw_sym, bars in raw_data.items():
                sym = str(raw_sym).strip().upper()
                if not bars:
                    continue
                rows = [
                    {
                        "timestamp": _extract_bar_field(b, "timestamp"),
                        "open": _extract_bar_field(b, "open"),
                        "high": _extract_bar_field(b, "high"),
                        "low": _extract_bar_field(b, "low"),
                        "close": _extract_bar_field(b, "close"),
                        "volume": _extract_bar_field(b, "volume"),
                        "vwap": _extract_bar_field(b, "vwap"),
                    }
                    for b in bars
                ]
                df = pd.DataFrame(rows)
                if df.empty:
                    continue
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                if df["timestamp"].isna().any():
                    raise ValueError(f"Null timestamp received for {sym}")

                if canonical_tf == "1d":
                    ny_dates = df["timestamp"].dt.tz_convert(NY_TZ).dt.date
                    close_timestamps = [
                        datetime.combine(d, time(16, 0), tzinfo=NY_TZ).astimezone(UTC)
                        for d in ny_dates
                    ]
                    df["timestamp"] = pd.DatetimeIndex(close_timestamps, tz="UTC")
                elif canonical_tf in _TIMEFRAME_DELTAS:
                    df["timestamp"] = df["timestamp"] + _TIMEFRAME_DELTAS[canonical_tf]

                numeric_cols = ["open", "high", "low", "close", "volume"]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")
                df["vwap"] = pd.to_numeric(df["vwap"], errors="coerce").astype("float64")

                if not np.isfinite(df[numeric_cols].to_numpy()).all():
                    raise ValueError(f"Non-finite OHLCV values received for {sym}")

                price_cols = ["open", "high", "low", "close"]
                if (df[price_cols] <= 0.0).any().any():
                    raise ValueError(f"OHLC prices must be strictly positive for {sym}")
                if (df["volume"] < 0.0).any():
                    raise ValueError(f"Volume must be non-negative for {sym}")
                if (df["high"] < df["low"]).any():
                    raise ValueError(f"Bar high must be >= low for {sym}")
                if ((df["high"] < df["open"]) | (df["high"] < df["close"])).any():
                    raise ValueError(f"Bar high must be >= open and close for {sym}")
                if ((df["low"] > df["open"]) | (df["low"] > df["close"])).any():
                    raise ValueError(f"Bar low must be <= open and close for {sym}")

                invalid_vwap = (~np.isfinite(df["vwap"])) | (df["vwap"] <= 0.0)
                if invalid_vwap.any():
                    df.loc[invalid_vwap, "vwap"] = float("nan")

                df = df[CANONICAL_BAR_COLUMNS]
                df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
                df = (
                    df.drop_duplicates(subset=["timestamp"], keep="last")
                    .sort_values("timestamp", ascending=True)
                    .reset_index(drop=True)
                )
                results[sym] = df

        return results
