from __future__ import annotations

import inspect
import socket
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import structlog

from agy_trader.core.config import EnvSettings, load_universe
from agy_trader.core.models import AssetType, NewsArticle, ensure_utc
from agy_trader.data.alpaca_news import AlpacaNewsProvider
from agy_trader.data.feeds import collect_official_rss_feeds, collect_yfinance_news
from agy_trader.data.finnhub_news import FinnhubNewsProvider
from agy_trader.data.providers import NewsProvider
from agy_trader.nlp.sentiment import SentimentScorer, create_sentiment_scorer
from agy_trader.storage.db import StateDB

from .common import DaemonRunner, release_memory, wait_for_network

logger = structlog.get_logger(__name__)


class NewsNlpDaemon(DaemonRunner):
    """Continuous news/RSS ingestion and bounded-memory FinBERT/lexicon scoring daemon."""

    def __init__(
        self,
        settings: EnvSettings,
        data_dir: Path,
        config_dir: Path,
        *,
        db: StateDB | None = None,
        alpaca_news_provider: NewsProvider | None = None,
        finnhub_news_provider: NewsProvider | None = None,
        rss_collector: Callable[..., list[NewsArticle]] | None = None,
        yfinance_collector: Callable[..., list[NewsArticle]] | None = None,
        scorer: SentimentScorer | None = None,
        network_check_fn: Callable[[], bool] | None = None,
        poll_interval_seconds: float = 900.0,
        score_batch_size: int = 64,
        max_network_wait_seconds: float = 120.0,
        stop_event: threading.Event | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)
        resolved_db = db or StateDB(self.data_dir / "state.db")
        super().__init__(
            daemon_name="news-nlp",
            db=resolved_db,
            poll_interval_seconds=poll_interval_seconds,
            stop_event=stop_event,
            sleep_fn=sleep_fn,
        )
        self.score_batch_size = max(1, int(score_batch_size))
        self.max_network_wait_seconds = float(max_network_wait_seconds)

        self._alpaca_news_provider = alpaca_news_provider
        self._finnhub_news_provider = finnhub_news_provider
        self._rss_collector = rss_collector
        self._yfinance_collector = yfinance_collector
        self._scorer = scorer
        self._network_check_fn = network_check_fn
        self._last_poll_utc: str | None = None

    def _get_scorer(self) -> SentimentScorer:
        if self._scorer is None:
            self._scorer = create_sentiment_scorer(self.settings.nlp_backend)
        return self._scorer

    def _default_network_check(self) -> bool:
        if self._alpaca_news_provider is not None or self._rss_collector is not None:
            return True
        if not self.settings.alpaca_api_key.strip():
            return True
        try:
            socket.getaddrinfo("data.alpaca.markets", 443, type=socket.SOCK_STREAM)
            return True
        except OSError:
            return False

    def _check_network(self) -> bool:
        fn = self._network_check_fn or self._default_network_check
        return bool(fn())

    def startup(self, now_utc: datetime | None = None) -> None:
        now = ensure_utc(now_utc) if now_utc is not None else datetime.now(UTC)
        self.verify_wal_integrity(now_utc=now)
        net_ok = wait_for_network(
            self._check_network,
            max_wait_seconds=self.max_network_wait_seconds,
            stop_event=self.stop_event,
            sleep_fn=self._sleep_fn,
        )
        scorer = self._get_scorer()
        unscored = self.db.count_unscored_news(scorer_model=scorer.model_name)
        boot_details = {
            "network_ok": net_ok,
            "scorer_model": scorer.model_name,
            "unscored_remaining": unscored,
        }
        self.db.record_daemon_event(
            self.daemon_name,
            "boot",
            boot_details,
            timestamp=now,
        )
        self.write_heartbeat("healthy" if net_ok else "degraded", boot_details, now_utc=now)

    def _score_pending_batches(self, scorer: SentimentScorer) -> int:
        total_scored = 0
        while not self.stop_event.is_set():
            batch = self.db.get_unscored_news(
                scorer_model=scorer.model_name,
                limit=self.score_batch_size,
            )
            if not batch:
                break
            preds = scorer.score_articles(batch)
            updates = [
                (art.article_id, pred.score, pred.label, pred.confidence, pred.scorer_model)
                for art, pred in zip(batch, preds, strict=True)
            ]
            updated = self.db.update_news_scores(updates)
            total_scored += updated
            del batch, preds, updates
            release_memory()
            if updated <= 0:
                break
        return total_scored

    def run_once(self, now_utc: datetime | None = None) -> dict[str, Any]:
        now = ensure_utc(now_utc) if now_utc is not None else datetime.now(UTC)
        universe = load_universe(self.config_dir)
        all_symbols = [u.symbol for u in universe]
        etf_symbols = [u.symbol for u in universe if u.asset_type == AssetType.etf]

        def _source_start(sources: list[str], *, max_days: int, default_days: int) -> datetime:
            last_ts = self.db.get_latest_news_timestamp(sources=sources)
            if last_ts is None:
                return now - timedelta(days=default_days)
            candidate = max(now - timedelta(days=max_days), last_ts - timedelta(hours=1))
            return min(candidate, now - timedelta(hours=1))

        alpaca_start_utc = _source_start(["benzinga", "fiscal.ai"], max_days=7, default_days=3)
        finnhub_start_utc = _source_start(["finnhub"], max_days=2, default_days=2)

        alpaca_saved = 0
        finnhub_saved = 0
        rss_saved = 0
        yf_saved = 0
        poll_errors: list[str] = []

        http_client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "agy-trader/0.1 (news-nlp-daemon)"},
        )
        try:
            # 1. Incremental Alpaca news (1h overlap)
            try:
                alpaca_prov = self._alpaca_news_provider
                if (
                    alpaca_prov is None
                    and self.settings.alpaca_api_key.strip()
                    and self.settings.alpaca_secret_key.strip()
                ):
                    alpaca_prov = AlpacaNewsProvider(
                        api_key=self.settings.alpaca_api_key,
                        secret_key=self.settings.alpaca_secret_key,
                        client=http_client,
                    )
                if alpaca_prov is not None:
                    arts = alpaca_prov.fetch_news(
                        symbols=all_symbols,
                        start=alpaca_start_utc,
                        end=now,
                    )
                    alpaca_saved = self.db.upsert_news_articles(arts)
            except Exception as exc:  # noqa: BLE001
                logger.warning("alpaca_news_poll_failed", error=str(exc))
                poll_errors.append(f"alpaca:{exc}")

            # 2. Finnhub company news (if key set or injected)
            try:
                fh_prov = self._finnhub_news_provider
                if fh_prov is None and self.settings.finnhub_api_key.strip():
                    fh_prov = FinnhubNewsProvider(
                        api_key=self.settings.finnhub_api_key,
                        client=http_client,
                    )
                if fh_prov is not None:
                    fh_arts = fh_prov.fetch_news(
                        symbols=all_symbols,
                        start=finnhub_start_utc,
                        end=now,
                    )
                    finnhub_saved = self.db.upsert_news_articles(fh_arts)
            except Exception as exc:  # noqa: BLE001
                logger.warning("finnhub_news_poll_failed", error=str(exc))
                poll_errors.append(f"finnhub:{exc}")

            # 3. Forward-only official RSS feeds (Fed/EIA/FDA)
            try:
                if self._rss_collector is not None:
                    params = inspect.signature(self._rss_collector).parameters
                    if "client" in params:
                        rss_arts = self._rss_collector(client=http_client)
                    else:
                        rss_arts = self._rss_collector()
                else:
                    rss_arts = collect_official_rss_feeds(client=http_client)
                rss_saved = self.db.upsert_news_articles(rss_arts)
            except Exception as exc:  # noqa: BLE001
                logger.warning("rss_poll_failed", error=str(exc))
                poll_errors.append(f"rss:{exc}")

            # 4. Forward-only yfinance ETF news
            try:
                yf_fn: Callable[[Sequence[str]], list[NewsArticle]] = (
                    self._yfinance_collector
                    if self._yfinance_collector is not None
                    else collect_yfinance_news
                )
                yf_arts = yf_fn(etf_symbols)
                yf_saved = self.db.upsert_news_articles(yf_arts)
            except Exception as exc:  # noqa: BLE001
                logger.warning("yfinance_news_poll_failed", error=str(exc))
                poll_errors.append(f"yfinance:{exc}")
        finally:
            http_client.close()

        scorer = self._get_scorer()
        scored_count = self._score_pending_batches(scorer)
        unscored_remaining = self.db.count_unscored_news(scorer_model=scorer.model_name)
        self._last_poll_utc = now.isoformat()

        total_saved = alpaca_saved + finnhub_saved + rss_saved + yf_saved
        hb_status = "healthy" if len(poll_errors) < 4 else "degraded"
        details: dict[str, Any] = {
            "last_poll_utc": self._last_poll_utc,
            "scorer_model": scorer.model_name,
            "articles_saved": total_saved,
            "alpaca_saved": alpaca_saved,
            "finnhub_saved": finnhub_saved,
            "rss_saved": rss_saved,
            "yfinance_saved": yf_saved,
            "articles_scored": scored_count,
            "unscored_remaining": unscored_remaining,
            "poll_errors": poll_errors,
        }
        self.write_heartbeat(hb_status, details, now_utc=now)
        release_memory()
        return {"status": hb_status, **details}
