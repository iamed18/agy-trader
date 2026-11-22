from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from agy_trader.core.models import NewsArticle, ensure_utc
from agy_trader.data.news_filters import (
    canonicalize_article_symbols,
    expand_query_symbols,
    is_institutional_noise_headline,
)
from agy_trader.data.providers import NewsProvider

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _parse_float_header(response: httpx.Response, name: str) -> float | None:
    raw = response.headers.get(name)
    if raw is None:
        return None
    try:
        return float(raw.strip())
    except ValueError:
        return None


def _extract_related_symbols(raw_related: Any, query_symbol: str) -> list[str]:
    syms: list[str] = [query_symbol]
    if isinstance(raw_related, str):
        for part in raw_related.split(","):
            cleaned = part.strip()
            if cleaned:
                syms.append(cleaned)
    elif isinstance(raw_related, list):
        for part in raw_related:
            if part is not None and str(part).strip():
                syms.append(str(part).strip())
    return syms


class FinnhubNewsProvider(NewsProvider):
    """Finnhub `/company-news` historical provider with 30-day window slicing and rate pacing."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://finnhub.io/api/v1/company-news",
        client: httpx.Client | None = None,
        chunk_days: int = 30,
        min_interval_seconds: float = 1.25,
        timeout: float = 30.0,
        max_retries: int = 6,
        backoff_base_seconds: float = 1.0,
        max_backoff_seconds: float = 65.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if chunk_days < 1:
            raise ValueError("chunk_days must be >= 1")
        if min_interval_seconds < 0.0:
            raise ValueError("min_interval_seconds must be >= 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.chunk_days = chunk_days
        self.min_interval_seconds = min_interval_seconds
        self._timeout = timeout
        self._external_client = client
        self._max_retries = max_retries
        self._backoff_base_seconds = backoff_base_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._sleep = sleep_fn
        self._now = now_fn
        self._last_request_ts: float | None = None

    def _pace(self) -> None:
        if self.min_interval_seconds <= 0.0 or self._last_request_ts is None:
            return
        elapsed = self._now() - self._last_request_ts
        if elapsed < self.min_interval_seconds:
            self._sleep(self.min_interval_seconds - elapsed)

    def _compute_retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = _parse_float_header(response, "Retry-After")
        if retry_after is not None and retry_after > 0:
            return min(self._max_backoff_seconds, retry_after)
        return min(self._max_backoff_seconds, self._backoff_base_seconds * (2**attempt))

    def _get_window(self, client: httpx.Client, params: dict[str, Any]) -> httpx.Response:
        headers = {"X-Finnhub-Token": self.api_key}
        for attempt in range(self._max_retries + 1):
            self._pace()
            self._last_request_ts = self._now()
            response = client.get(self.base_url, params=params, headers=headers)
            if response.status_code not in _RETRYABLE_STATUS_CODES:
                response.raise_for_status()
                return response
            if attempt >= self._max_retries:
                response.raise_for_status()
            delay = self._compute_retry_delay(response, attempt)
            logger.warning(
                "Finnhub news HTTP %s (attempt %d/%d); retrying in %.2fs",
                response.status_code,
                attempt + 1,
                self._max_retries,
                delay,
            )
            self._sleep(delay)
        raise AssertionError("Unreachable")

    def _date_windows(self, start_date: date, end_date: date) -> list[tuple[date, date]]:
        windows: list[tuple[date, date]] = []
        cur = start_date
        step = timedelta(days=self.chunk_days - 1)
        while cur <= end_date:
            win_end = min(end_date, cur + step)
            windows.append((cur, win_end))
            cur = win_end + timedelta(days=1)
        return windows

    def fetch_news(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> list[NewsArticle]:
        start_utc = ensure_utc(start)
        end_utc = ensure_utc(end)
        if end_utc < start_utc:
            raise ValueError("end must be >= start")
        if not self.api_key:
            return []

        requested_symbols = list(
            dict.fromkeys(sym.strip().upper() for sym in symbols if sym.strip())
        )
        requested_set = set(requested_symbols)
        query_symbols = expand_query_symbols(requested_symbols)
        if not query_symbols:
            return []

        windows = self._date_windows(start_utc.date(), end_utc.date())
        articles_by_id: dict[str, NewsArticle] = {}
        own_client = self._external_client is None
        client = self._external_client or httpx.Client(timeout=self._timeout)

        try:
            for q_sym in query_symbols:
                for win_start, win_end in windows:
                    params = {
                        "symbol": q_sym,
                        "from": win_start.isoformat(),
                        "to": win_end.isoformat(),
                    }
                    response = self._get_window(client, params)
                    payload = response.json()
                    if not isinstance(payload, list):
                        continue

                    for item in payload:
                        if not isinstance(item, dict):
                            continue
                        raw_id = item.get("id")
                        item_id = str(raw_id).strip() if raw_id is not None else ""
                        headline = str(item.get("headline") or "").strip()
                        raw_ts = item.get("datetime")
                        if not item_id or not headline or raw_ts is None or isinstance(raw_ts, bool):
                            continue
                        if is_institutional_noise_headline(headline):
                            continue

                        try:
                            epoch_sec = float(raw_ts)
                            if not math.isfinite(epoch_sec):
                                continue
                            published_at = datetime.fromtimestamp(epoch_sec, tz=UTC)
                        except (TypeError, ValueError, OverflowError, OSError):
                            continue
                        if published_at < start_utc or published_at > end_utc:
                            continue

                        raw_syms = _extract_related_symbols(item.get("related"), q_sym)
                        filtered_syms = canonicalize_article_symbols(raw_syms, requested_set)
                        article_id = f"finnhub:{item_id}"

                        existing = articles_by_id.get(article_id)
                        if existing is not None:
                            merged_syms = list(dict.fromkeys([*existing.symbols, *filtered_syms]))
                            articles_by_id[article_id] = existing.model_copy(
                                update={"symbols": merged_syms}
                            )
                        else:
                            articles_by_id[article_id] = NewsArticle(
                                article_id=article_id,
                                published_at=published_at,
                                headline=headline,
                                summary=str(item.get("summary") or "").strip(),
                                source="finnhub",
                                symbols=filtered_syms,
                            )
        finally:
            if own_client:
                client.close()

        return sorted(articles_by_id.values(), key=lambda a: (a.published_at, a.article_id))
