from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime

import pandas as pd

from agy_trader.core.models import EarningsEvent, MacroPoint, NewsArticle, SecFilingEvent


class MarketDataProvider(ABC):
    """Abstract interface for historical and recent OHLCV bar providers."""

    @abstractmethod
    def fetch_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch canonical OHLCV bars keyed by uppercase symbol."""


class NewsProvider(ABC):
    """Abstract interface for financial news providers."""

    @abstractmethod
    def fetch_news(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> list[NewsArticle]:
        """Fetch news articles for the requested symbols and time window."""


class MacroProvider(ABC):
    """Abstract interface for macroeconomic time-series providers."""

    @abstractmethod
    def fetch_macro(
        self,
        start_date: date,
        end_date: date,
    ) -> list[MacroPoint]:
        """Fetch macro indicator observations over [start_date, end_date]."""


class EarningsProvider(ABC):
    """Abstract interface for corporate earnings calendar providers."""

    @abstractmethod
    def fetch_earnings(self, symbols: list[str]) -> list[EarningsEvent]:
        """Fetch earnings calendar events for the requested symbols."""


class SecFilingsProvider(ABC):
    """Abstract interface for point-in-time SEC EDGAR filings providers."""

    @abstractmethod
    def fetch_filings(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> list[SecFilingEvent]:
        """Fetch SEC filings for the requested symbols within [start, end]."""


class EtfHoldingsProvider(ABC):
    """Abstract interface for ETF top-holdings providers."""

    @abstractmethod
    def fetch_etf_holdings(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        """Fetch top-holding weights in [0.0, 1.0] keyed by uppercase ETF symbol."""
