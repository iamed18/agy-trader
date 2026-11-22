from __future__ import annotations

import hashlib
import html
import logging
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import yfinance as yf

from agy_trader.core.models import NewsArticle, ensure_utc
from agy_trader.data.news_filters import (
    canonicalize_article_symbols,
    is_institutional_noise_headline,
)

logger = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class RssFeedSpec:
    name: str
    url: str
    source: str
    symbols: tuple[str, ...]


OFFICIAL_RSS_FEEDS: tuple[RssFeedSpec, ...] = (
    RssFeedSpec(
        name="Fed FOMC",
        url="https://www.federalreserve.gov/feeds/press_monetary.xml",
        source="rss:fed_fomc",
        symbols=("SPY", "TLT", "GLD", "XLF"),
    ),
    RssFeedSpec(
        name="EIA Today in Energy",
        url="https://www.eia.gov/rss/todayinenergy.xml",
        source="rss:eia",
        symbols=("XLE", "XOM", "CVX", "COP"),
    ),
    RssFeedSpec(
        name="FDA Press Releases",
        url="https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml",
        source="rss:fda",
        symbols=("XLV", "LLY", "JNJ", "ABBV", "MRK"),
    ),
)


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _clean_text(raw: str | None) -> str:
    if not raw:
        return ""
    unescaped = html.unescape(raw)
    stripped = _HTML_TAG_RE.sub(" ", unescaped)
    return _WS_RE.sub(" ", stripped).strip()


def _parse_feed_datetime(raw: str | None) -> datetime | None:
    if not raw or not raw.strip():
        return None
    cleaned = raw.strip()
    try:
        dt = parsedate_to_datetime(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return ensure_utc(dt)
    except (TypeError, ValueError):
        pass

    iso_candidate = f"{cleaned[:-1]}+00:00" if cleaned.endswith("Z") else cleaned
    try:
        dt = datetime.fromisoformat(iso_candidate)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return ensure_utc(dt)
    except ValueError:
        return None


def _child_text(node: ET.Element, *names: str) -> str:
    for child in node:
        if _local_name(child.tag) in names:
            txt = "".join(child.itertext())
            if txt and txt.strip():
                return txt.strip()
    return ""


def _atom_link(node: ET.Element) -> str:
    for child in node:
        if _local_name(child.tag) == "link":
            href = (child.attrib.get("href") or "").strip()
            if href:
                return href
            if child.text and child.text.strip():
                return child.text.strip()
    return ""


def _make_rss_article_id(source: str, uid_raw: str, headline: str, pub_at: datetime) -> str:
    seed = uid_raw.strip() or f"{headline}|{pub_at.isoformat()}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"{source}:{digest}"


def _parse_rss_or_atom_xml(
    xml_text: str,
    spec: RssFeedSpec,
    fallback_now: datetime,
) -> list[NewsArticle]:
    root = ET.fromstring(xml_text)
    articles: list[NewsArticle] = []
    root_name = _local_name(root.tag).lower()

    if root_name == "feed":
        entries = [el for el in root if _local_name(el.tag) == "entry"]
        for entry in entries:
            headline = _clean_text(_child_text(entry, "title"))
            if not headline:
                continue
            summary = _clean_text(_child_text(entry, "summary", "content"))
            raw_dt = _child_text(entry, "published", "updated")
            pub_at = _parse_feed_datetime(raw_dt) or fallback_now
            uid_raw = _child_text(entry, "id") or _atom_link(entry)
            article_id = _make_rss_article_id(spec.source, uid_raw, headline, pub_at)
            articles.append(
                NewsArticle(
                    article_id=article_id,
                    published_at=pub_at,
                    headline=headline,
                    summary=summary,
                    source=spec.source,
                    symbols=list(spec.symbols),
                )
            )
        return articles

    items = [el for el in root.iter() if _local_name(el.tag) == "item"]
    for item in items:
        headline = _clean_text(_child_text(item, "title"))
        if not headline:
            continue
        summary = _clean_text(_child_text(item, "description", "summary", "encoded"))
        raw_dt = _child_text(item, "pubDate", "date", "published", "updated")
        pub_at = _parse_feed_datetime(raw_dt) or fallback_now
        uid_raw = _child_text(item, "guid", "id", "link")
        article_id = _make_rss_article_id(spec.source, uid_raw, headline, pub_at)
        articles.append(
            NewsArticle(
                article_id=article_id,
                published_at=pub_at,
                headline=headline,
                summary=summary,
                source=spec.source,
                symbols=list(spec.symbols),
            )
        )
    return articles


def collect_official_rss_feeds(
    feeds: Sequence[RssFeedSpec] = OFFICIAL_RSS_FEEDS,
    client: httpx.Client | None = None,
    timeout: float = 20.0,
    now_fn: Callable[[], datetime] | None = None,
) -> list[NewsArticle]:
    """Fetch and parse official Fed/EIA/FDA RSS/Atom feeds (forward-only ops intelligence)."""
    fallback_now = ensure_utc(now_fn() if now_fn is not None else datetime.now(UTC))
    articles_by_id: dict[str, NewsArticle] = {}
    own_client = client is None
    http_client = client or httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "agy-trader/0.1 (rss-collector)"},
    )

    try:
        for spec in feeds:
            try:
                response = http_client.get(spec.url)
                response.raise_for_status()
                parsed = _parse_rss_or_atom_xml(response.text, spec, fallback_now)
                for art in parsed:
                    articles_by_id[art.article_id] = art
            except (httpx.HTTPError, ET.ParseError, ValueError, OSError) as exc:
                logger.warning("Failed to collect RSS feed %s (%s): %s", spec.name, spec.url, exc)
                continue
    finally:
        if own_client:
            http_client.close()

    return sorted(articles_by_id.values(), key=lambda a: (a.published_at, a.article_id))


def _parse_yfinance_news_item(
    item: dict[str, Any],
    symbol: str,
) -> NewsArticle | None:
    content = item.get("content")
    nested = content if isinstance(content, dict) else {}

    headline = _clean_text(str(nested.get("title") or item.get("title") or ""))
    if not headline or is_institutional_noise_headline(headline):
        return None

    summary = _clean_text(
        str(
            nested.get("summary")
            or nested.get("description")
            or item.get("summary")
            or ""
        )
    )

    pub_at: datetime | None = None
    raw_epoch = item.get("providerPublishTime")
    if raw_epoch is not None and not isinstance(raw_epoch, bool):
        try:
            sec = float(raw_epoch)
            if math.isfinite(sec):
                pub_at = datetime.fromtimestamp(sec, tz=UTC)
        except (TypeError, ValueError, OverflowError, OSError):
            pub_at = None

    if pub_at is None:
        raw_iso = nested.get("pubDate") or nested.get("displayTime") or item.get("pubDate")
        if raw_iso is not None:
            pub_at = _parse_feed_datetime(str(raw_iso))

    if pub_at is None:
        return None

    raw_uid = str(
        item.get("id")
        or nested.get("id")
        or item.get("uuid")
        or ""
    ).strip()
    if not raw_uid:
        raw_uid = hashlib.sha256(f"{headline}|{pub_at.isoformat()}".encode()).hexdigest()[:16]

    canonical_syms = canonicalize_article_symbols([symbol], set())
    return NewsArticle(
        article_id=f"yf:{raw_uid}",
        published_at=pub_at,
        headline=headline,
        summary=summary,
        source="yfinance_news",
        symbols=canonical_syms,
    )


def collect_yfinance_news(
    symbols: Sequence[str],
    max_per_symbol: int = 10,
) -> list[NewsArticle]:
    """Fetch recent Yahoo Finance headlines per symbol (forward-only ops intelligence)."""
    if max_per_symbol < 1:
        raise ValueError("max_per_symbol must be >= 1")

    clean_symbols = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    articles_by_id: dict[str, NewsArticle] = {}

    for sym in clean_symbols:
        try:
            ticker = yf.Ticker(sym)
            raw_items = ticker.news
            if not isinstance(raw_items, list):
                continue
            for item in raw_items[:max_per_symbol]:
                if not isinstance(item, dict):
                    continue
                article = _parse_yfinance_news_item(item, sym)
                if article is None:
                    continue
                existing = articles_by_id.get(article.article_id)
                if existing is not None:
                    merged_syms = list(dict.fromkeys([*existing.symbols, *article.symbols]))
                    articles_by_id[article.article_id] = existing.model_copy(
                        update={"symbols": merged_syms}
                    )
                else:
                    articles_by_id[article.article_id] = article
        except Exception as exc:  # noqa: BLE001 - best-effort per symbol
            logger.warning("Failed to collect yfinance news for %s: %s", sym, exc)
            continue

    return sorted(articles_by_id.values(), key=lambda a: (a.published_at, a.article_id))
