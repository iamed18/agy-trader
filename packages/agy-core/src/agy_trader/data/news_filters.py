from __future__ import annotations

import re
from collections.abc import Sequence

SYMBOL_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "GOOGL": ("GOOGL", "GOOG"),
}

SYMBOL_CANONICAL_MAP: dict[str, str] = {
    "GOOG": "GOOGL",
}

_INSTITUTIONAL_NOISE_RE = re.compile(
    r"(?:"
    r"\bvoting rights\b"
    r"|\bform 13[fgd]\b"
    r"|\bschedule 13\b"
    r"|\bschedule 13[dg]\b"
    r"|\bdiscloses stake\b"
    r"|\bdiscloses increased\b"
    r"|\bdiscloses decreased\b"
    r"|\bmajor shareholding\b"
    r"|\btr-1 notification\b"
    r"|\bnotification of major holdings\b"
    r"|\bsec filing:"
    r")",
    re.IGNORECASE,
)


def expand_query_symbols(symbols: Sequence[str]) -> list[str]:
    """Expand requested tickers with query aliases while preserving order and deduplicating."""
    expanded: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        sym = raw.strip().upper()
        if not sym:
            continue
        for candidate in SYMBOL_QUERY_ALIASES.get(sym, (sym,)):
            if candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


def canonicalize_article_symbols(
    raw_symbols: Sequence[str],
    requested_set: set[str],
) -> list[str]:
    """Map aliased tickers (e.g. GOOG -> GOOGL), deduplicate, and filter by requested_set."""
    canonical_requested = {
        SYMBOL_CANONICAL_MAP.get(s.strip().upper(), s.strip().upper())
        for s in requested_set
        if s.strip()
    }
    result: list[str] = []
    seen: set[str] = set()
    for raw in raw_symbols:
        sym = raw.strip().upper()
        if not sym:
            continue
        canonical = SYMBOL_CANONICAL_MAP.get(sym, sym)
        if canonical_requested and canonical not in canonical_requested:
            continue
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def is_institutional_noise_headline(headline: str) -> bool:
    """Return True if headline is routine third-party holdings/disclosure spam."""
    if not headline or not headline.strip():
        return False
    return _INSTITUTIONAL_NOISE_RE.search(headline) is not None


def is_forward_only_source(source: str) -> bool:
    """Return True if source is a forward-only ops intelligence feed excluded from modeling."""
    cleaned = source.strip().lower()
    return cleaned == "yfinance_news" or cleaned.startswith("rss:")
