from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import httpx

from agy_trader.core.models import NewsArticle, ensure_utc
from agy_trader.data.providers import NewsProvider

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _format_utc_iso(dt: datetime) -> str:
    return ensure_utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc_timestamp(raw: str) -> datetime:
    cleaned = raw.strip()
    if cleaned.endswith("Z"):
        cleaned = f"{cleaned[:-1]}+00:00"
    return ensure_utc(datetime.fromisoformat(cleaned))


def _parse_float_header(response: httpx.Response, name: str) -> float | None:
    raw = response.headers.get(name)
    if raw is None:
        return None
    try:
        return float(raw.strip())
    except ValueError:
        return None


class AlpacaNewsProvider(NewsProvider):
    """Alpaca v1beta1 News API provider with pagination, rate-limit backoff, and respx-friendly client."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "https://data.alpaca.markets/v1beta1/news",
        client: httpx.Client | None = None,
        batch_size: int = 50,
        timeout: float = 30.0,
        max_retries: int = 6,
        backoff_base_seconds: float = 1.0,
        max_backoff_seconds: float = 65.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.batch_size = batch_size
        self._timeout = timeout
        self._external_client = client
        self._max_retries = max_retries
        self._backoff_base_seconds = backoff_base_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._sleep = sleep_fn
        self._now = now_fn

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

    def _compute_retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = _parse_float_header(response, "Retry-After")
        if retry_after is not None and retry_after > 0:
            return min(self._max_backoff_seconds, retry_after)

        reset_epoch = _parse_float_header(response, "X-RateLimit-Reset")
        if reset_epoch is not None:
            wait = reset_epoch - self._now() + 0.5
            if wait > 0:
                return min(self._max_backoff_seconds, wait)

        return min(self._max_backoff_seconds, self._backoff_base_seconds * (2**attempt))

    def _maybe_pace_rate_limit(self, response: httpx.Response) -> None:
        remaining = _parse_float_header(response, "X-RateLimit-Remaining")
        if remaining is None or remaining > 1:
            return
        reset_epoch = _parse_float_header(response, "X-RateLimit-Reset")
        if reset_epoch is None:
            return
        wait = reset_epoch - self._now() + 0.5
        if wait > 0:
            delay = min(self._max_backoff_seconds, wait)
            logger.info("Alpaca news rate limit nearly exhausted; sleeping %.2fs until reset", delay)
            self._sleep(delay)

    def _get_page(self, client: httpx.Client, params: dict[str, Any]) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            response = client.get(
                self.base_url,
                params=params,
                headers=self._headers(),
            )
            if response.status_code not in _RETRYABLE_STATUS_CODES:
                response.raise_for_status()
                return response
            if attempt >= self._max_retries:
                response.raise_for_status()
            delay = self._compute_retry_delay(response, attempt)
            logger.warning(
                "Alpaca news HTTP %s (attempt %d/%d); retrying in %.2fs",
                response.status_code,
                attempt + 1,
                self._max_retries,
                delay,
            )
            self._sleep(delay)
        raise AssertionError("Unreachable")

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

        requested_symbols = list(
            dict.fromkeys(sym.strip().upper() for sym in symbols if sym.strip())
        )
        requested_set = set(requested_symbols)
        if not requested_symbols:
            return []

        articles_by_id: dict[str, NewsArticle] = {}
        own_client = self._external_client is None
        client = self._external_client or httpx.Client(timeout=self._timeout)

        try:
            for idx in range(0, len(requested_symbols), self.batch_size):
                batch = requested_symbols[idx : idx + self.batch_size]
                page_token: str | None = None
                while True:
                    params: dict[str, Any] = {
                        "symbols": ",".join(batch),
                        "start": _format_utc_iso(start_utc),
                        "end": _format_utc_iso(end_utc),
                        "limit": 50,
                        "include_content": "false",
                        "sort": "asc",
                    }
                    if page_token:
                        params["page_token"] = page_token

                    response = self._get_page(client, params)
                    payload = response.json()
                    raw_news = (
                        (payload.get("news") or []) if isinstance(payload, dict) else []
                    )

                    for item in raw_news:
                        if not isinstance(item, dict):
                            continue
                        raw_id = str(item.get("id") or "").strip()
                        headline = str(item.get("headline") or "").strip()
                        created_raw = item.get("created_at") or item.get("updated_at")
                        if not raw_id or not headline or not created_raw:
                            continue

                        published_at = _parse_utc_timestamp(str(created_raw))
                        if published_at < start_utc or published_at > end_utc:
                            continue

                        raw_syms = [
                            str(s).strip().upper()
                            for s in (item.get("symbols") or [])
                            if s is not None and str(s).strip()
                        ]
                        filtered_syms = (
                            [s for s in dict.fromkeys(raw_syms) if s in requested_set]
                            if requested_set
                            else list(dict.fromkeys(raw_syms))
                        )

                        article = NewsArticle(
                            article_id=raw_id,
                            published_at=published_at,
                            headline=headline,
                            summary=str(item.get("summary") or "").strip(),
                            source=str(item.get("source") or "benzinga").strip() or "benzinga",
                            symbols=filtered_syms,
                        )
                        articles_by_id[article.article_id] = article

                    next_token = (
                        payload.get("next_page_token") if isinstance(payload, dict) else None
                    )
                    if not next_token:
                        break
                    self._maybe_pace_rate_limit(response)
                    page_token = str(next_token)
        finally:
            if own_client:
                client.close()

        return sorted(articles_by_id.values(), key=lambda a: (a.published_at, a.article_id))
