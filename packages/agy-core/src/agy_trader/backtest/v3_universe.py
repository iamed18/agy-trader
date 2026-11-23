from __future__ import annotations

import csv
import io
import json
import logging
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, NamedTuple

import httpx
import numpy as np
import pandas as pd
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict

from agy_trader.backtest.sp500_cohorts import (
    atomic_write_text,
    map_gics_sector,
    parse_wikipedia_sp500_html,
    validate_research_dir,
)
from agy_trader.core.models import AssetType, MacroPoint, UniverseSymbol, ensure_utc
from agy_trader.data.alpaca_market import AlpacaMarketDataProvider
from agy_trader.data.macro_and_calendar import FREDMacroProvider
from agy_trader.storage.parquet_store import ParquetBarStore

logger = logging.getLogger(__name__)

V3_CORE_ETFS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "IWM",
    "GLD",
    "TLT",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "XLB",
    "XLC",
    "XLRE",
)

SECTOR_TO_ETF: dict[str, str] = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Communication Services": "XLC",
    "Real Estate": "XLRE",
}

HISTORICAL_DELISTED_SECTORS: dict[str, str] = {
    "SIVB": "Financials",
    "FRC": "Financials",
    "SBNY": "Financials",
    "DFS": "Financials",
    "PBCT": "Financials",
    "ATVI": "Communication Services",
    "TWTR": "Communication Services",
    "DISH": "Communication Services",
    "VMW": "Technology",
    "XLNX": "Technology",
    "RHT": "Technology",
    "CTXS": "Technology",
    "NLSN": "Technology",
    "PXD": "Energy",
    "MRO": "Energy",
    "HES": "Energy",
    "CXO": "Energy",
    "NBL": "Energy",
    "CELG": "Healthcare",
    "CERN": "Healthcare",
    "ALXN": "Healthcare",
    "VAR": "Healthcare",
    "PKI": "Healthcare",
    "KSU": "Industrials",
    "TIF": "Consumer Discretionary",
    "DRE": "Real Estate",
}

_DUAL_CLASS_DROP: dict[str, str] = {
    "GOOG": "GOOGL",
    "FOX": "FOXA",
    "NWS": "NWSA",
    "BRK.A": "BRK.B",
    "BF.A": "BF.B",
}

SP500_HISTORY_CSV_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
)
WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKIPEDIA_SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"

_FOOTNOTE_RE = re.compile(r"\[[^\]]*\]")



def normalize_equity_symbol(symbol: str) -> str:
    """Normalize equity ticker symbol by stripping whitespace, uppercase, and mapping hyphens to dots."""
    cleaned = _FOOTNOTE_RE.sub("", symbol or "").strip().upper()
    return cleaned.replace("-", ".")


class V3EquityCandidate(BaseModel):
    """Point-in-time equity candidate metadata for V3 universe construction."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    symbol: str
    security: str = ""
    sector: str = "Unknown"
    sources: tuple[str, ...] = ()
    sp500_intervals: tuple[tuple[str, str | None], ...] = ()


def is_in_sp500_on(candidate: V3EquityCandidate, as_of: date) -> bool:
    """Check whether candidate was in the S&P 500 on as_of date based on its intervals."""
    for start_iso, end_iso in candidate.sp500_intervals:
        try:
            start_date = date.fromisoformat(start_iso)
        except (ValueError, TypeError):
            continue
        if start_date <= as_of:
            if end_iso is None:
                return True
            try:
                end_date = date.fromisoformat(end_iso)
            except (ValueError, TypeError):
                continue
            if as_of <= end_date:
                return True
    return False


class Sp400Constituent(NamedTuple):
    """Parsed S&P 400 constituent record."""

    symbol: str
    security: str
    sector: str


def parse_sp500_start_end_csv(
    csv_text: str,
    *,
    min_overlap_date: date = date(2016, 1, 1),
    max_overlap_date: date = date(2026, 11, 20),
) -> dict[str, list[tuple[str, str | None]]]:
    """Parse historical S&P 500 ticker intervals overlapping [min_overlap_date, max_overlap_date]."""
    reader = csv.reader(io.StringIO(csv_text))
    rows = [r for r in reader if r]
    if not rows:
        return {}

    header = [c.strip().lower() for c in rows[0]]
    col_ticker = 0
    col_start = 1
    col_end = 2

    has_header = False
    for idx, col in enumerate(header):
        if col in ("ticker", "symbol"):
            col_ticker = idx
            has_header = True
        elif col in ("start_date", "start"):
            col_start = idx
            has_header = True
        elif col in ("end_date", "end"):
            col_end = idx
            has_header = True

    data_rows = rows[1:] if has_header else rows
    intervals_by_sym: dict[str, list[tuple[str, str | None]]] = {}

    for row in data_rows:
        if len(row) <= col_ticker:
            continue
        raw_ticker = row[col_ticker].strip()
        if not raw_ticker or raw_ticker.lower() in ("ticker", "symbol"):
            continue

        sym = normalize_equity_symbol(raw_ticker)
        raw_start = row[col_start].strip() if col_start < len(row) else ""
        raw_end = row[col_end].strip() if col_end < len(row) else ""

        start_d: date | None = None
        if raw_start:
            try:
                start_d = date.fromisoformat(raw_start)
            except ValueError:
                continue

        end_d: date | None = None
        if raw_end and raw_end.lower() not in ("none", "null", "nan", "active", "present", ""):
            try:
                end_d = date.fromisoformat(raw_end)
            except ValueError:
                continue

        if start_d is not None and end_d is not None and end_d < start_d:
            continue

        start_bound = start_d if start_d is not None else min_overlap_date
        overlaps = (start_bound <= max_overlap_date) and (end_d is None or end_d >= min_overlap_date)
        if not overlaps:
            continue

        start_iso = start_d.isoformat() if start_d is not None else min_overlap_date.isoformat()
        end_iso = end_d.isoformat() if end_d is not None else None
        intervals_by_sym.setdefault(sym, []).append((start_iso, end_iso))

    return intervals_by_sym


def parse_wikipedia_sp400_html(html: str) -> list[Sp400Constituent]:
    """Parse Wikipedia's S&P 400 constituents table (<table id='constituents'> or wikitable)."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="constituents")
    if table is None:
        for candidate_table in soup.find_all("table"):
            first_row = candidate_table.find("tr")
            if not first_row:
                continue
            headers = [
                _FOOTNOTE_RE.sub("", cell.get_text(" ", strip=True)).strip().lower()
                for cell in first_row.find_all(["th", "td"])
            ]
            has_sym = any("symbol" in h or "ticker" in h for h in headers)
            has_sec = any("security" in h or "company" in h for h in headers)
            has_sect = any("sector" in h for h in headers)
            if has_sym and has_sec and has_sect:
                table = candidate_table
                break

    if table is None:
        raise ValueError("Could not find S&P 400 constituents table in HTML")

    rows = table.find_all("tr")
    if not rows:
        raise ValueError("S&P 400 constituents table contains no rows")

    header_cells = [
        _FOOTNOTE_RE.sub("", th.get_text(" ", strip=True)).strip().lower()
        for th in rows[0].find_all(["th", "td"])
    ]

    def _find_col(candidates: tuple[str, ...], fallback: int) -> int:
        for idx, name in enumerate(header_cells):
            if any(cand in name for cand in candidates):
                return idx
        return fallback

    col_symbol = _find_col(("ticker symbol", "symbol", "ticker"), 0)
    col_security = _find_col(("security", "company"), 1)
    col_sector = _find_col(("gics sector", "gics_sector", "sector"), 2)

    results: list[Sp400Constituent] = []
    seen: set[str] = set()
    for tr in rows[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) <= max(col_symbol, col_security, col_sector):
            continue
        raw_sym = _FOOTNOTE_RE.sub("", cells[col_symbol].get_text(" ", strip=True))
        sym = normalize_equity_symbol(raw_sym)
        if not sym or sym in seen:
            continue
        sec = _FOOTNOTE_RE.sub("", cells[col_security].get_text(" ", strip=True)).strip()
        gics_sec = _FOOTNOTE_RE.sub("", cells[col_sector].get_text(" ", strip=True)).strip()
        sector = map_gics_sector(gics_sec)
        seen.add(sym)
        results.append(Sp400Constituent(symbol=sym, security=sec, sector=sector))

    if not results:
        raise ValueError("Parsed 0 constituents from S&P 400 table")
    return results


def load_or_fetch_v3_equity_pool(
    research_dir: Path,
    *,
    refresh: bool = False,
    min_overlap_date: date = date(2016, 1, 1),
    max_overlap_date: date = date(2026, 11, 20),
    client: httpx.Client | None = None,
    history_csv_url: str = SP500_HISTORY_CSV_URL,
    sp500_url: str = WIKIPEDIA_SP500_URL,
    sp400_url: str = WIKIPEDIA_SP400_URL,
) -> list[V3EquityCandidate]:
    """Load cached V3 equity pool or discover & merge S&P 500 history, S&P 500 current, and S&P 400."""
    validate_research_dir(research_dir)
    cache_path = research_dir / "v3_equity_pool.json"

    if not refresh and cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return [V3EquityCandidate.model_validate(item) for item in raw]

    own_client = client is None
    http_client = client or httpx.Client(
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": "agy-trader-v3/1.0"},
    )
    try:
        csv_resp = http_client.get(history_csv_url)
        csv_resp.raise_for_status()
        csv_text = csv_resp.text

        sp500_resp = http_client.get(sp500_url)
        sp500_resp.raise_for_status()
        sp500_html = sp500_resp.text

        sp400_resp = http_client.get(sp400_url)
        sp400_resp.raise_for_status()
        sp400_html = sp400_resp.text
    finally:
        if own_client:
            http_client.close()

    hist_intervals = parse_sp500_start_end_csv(
        csv_text,
        min_overlap_date=min_overlap_date,
        max_overlap_date=max_overlap_date,
    )
    sp500_constituents = parse_wikipedia_sp500_html(sp500_html)
    sp400_constituents = parse_wikipedia_sp400_html(sp400_html)

    symbol_data: dict[str, dict[str, Any]] = {}

    def _get(sym: str) -> dict[str, Any]:
        if sym not in symbol_data:
            symbol_data[sym] = {
                "symbol": sym,
                "security": "",
                "sector": "Unknown",
                "sources": set(),
                "sp500_intervals": [],
            }
        return symbol_data[sym]

    for ticker, ivs in hist_intervals.items():
        sym = normalize_equity_symbol(ticker)
        entry = _get(sym)
        entry["sources"].add("sp500_hist")
        entry["sp500_intervals"].extend(ivs)

    for c500 in sp500_constituents:
        sym = normalize_equity_symbol(c500.symbol)
        entry = _get(sym)
        entry["sources"].add("sp500_curr")
        if not entry["security"]:
            entry["security"] = c500.security
        entry["sector"] = c500.sector
        has_active = any(end is None for _, end in entry["sp500_intervals"])
        if not has_active:
            start_str = (
                c500.date_added.isoformat() if c500.date_added else min_overlap_date.isoformat()
            )
            entry["sp500_intervals"].append((start_str, None))

    for c400 in sp400_constituents:
        sym = normalize_equity_symbol(c400.symbol)
        entry = _get(sym)
        entry["sources"].add("sp400_curr")
        if not entry["security"]:
            entry["security"] = c400.security
        if entry["sector"] == "Unknown":
            entry["sector"] = c400.sector

    for sym, entry in symbol_data.items():
        if entry["sector"] == "Unknown" and sym in HISTORICAL_DELISTED_SECTORS:
            entry["sector"] = HISTORICAL_DELISTED_SECTORS[sym]

    for dropped, preferred in _DUAL_CLASS_DROP.items():
        if dropped in symbol_data:
            dropped_entry = symbol_data.pop(dropped)
            preferred_entry = _get(preferred)
            preferred_entry["sources"].update(dropped_entry["sources"])
            preferred_entry["sp500_intervals"].extend(dropped_entry["sp500_intervals"])
            if not preferred_entry["security"] and dropped_entry["security"]:
                preferred_entry["security"] = dropped_entry["security"]
            if preferred_entry["sector"] == "Unknown" and dropped_entry["sector"] != "Unknown":
                preferred_entry["sector"] = dropped_entry["sector"]

    candidates: list[V3EquityCandidate] = []
    known_sources = ("sp500_hist", "sp500_curr", "sp400_curr")

    for sym in sorted(symbol_data.keys()):
        entry = symbol_data[sym]
        raw_ivs = entry["sp500_intervals"]
        unique_ivs = sorted(
            set(raw_ivs),
            key=lambda iv: (iv[0], iv[1] or "9999-12-31"),
        )
        src_tuple = tuple(s for s in known_sources if s in entry["sources"])
        extra_src = tuple(sorted(s for s in entry["sources"] if s not in known_sources))
        cand = V3EquityCandidate(
            symbol=sym,
            security=entry["security"],
            sector=entry["sector"],
            sources=src_tuple + extra_src,
            sp500_intervals=tuple(unique_ivs),
        )
        candidates.append(cand)

    serialized = json.dumps([c.model_dump(mode="json") for c in candidates], indent=2)
    atomic_write_text(cache_path, serialized)
    return candidates


def _is_rate_limit(exc: Exception) -> bool:
    """Check whether an exception represents an HTTP 429 rate limit."""
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    resp = getattr(exc, "response", None)
    return bool(resp is not None and getattr(resp, "status_code", None) == 429)


def fetch_and_cache_v3_sip_bars(
    symbols: Sequence[str],
    research_dir: Path,
    *,
    api_key: str,
    secret_key: str,
    start_utc: datetime = datetime(2016, 1, 4, tzinfo=UTC),
    end_utc: datetime = datetime(2026, 11, 19, 23, 59, tzinfo=UTC),
    refresh: bool = False,
    batch_size: int = 25,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, pd.DataFrame]:
    """Fetch daily SIP bars from Alpaca into research_dir with persistent manifest tracking."""
    validate_research_dir(research_dir)
    store = ParquetBarStore(research_dir)
    manifest_path = research_dir / "sip_fetch_manifest.json"

    manifest: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load existing sip_fetch_manifest.json: %s", exc)

    manifest_dirty = False

    def _flush_manifest() -> None:
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2))

    clean_symbols = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    for sym in clean_symbols:
        if not refresh:
            entry = manifest.get(sym)
            if entry is not None:
                status = entry.get("status")
                if status == "error":
                    to_fetch.append(sym)
                    continue
                if status == "empty":
                    continue
                if status == "ok":
                    parquet_path = store.path_for(sym, timeframe="1d")
                    if parquet_path.exists():
                        cached_df = store.load_bars(sym, timeframe="1d", start=start_utc, end=end_utc)
                        if not cached_df.empty:
                            bars_by_symbol[sym] = cached_df
                            continue
            else:
                parquet_path = store.path_for(sym, timeframe="1d")
                if parquet_path.exists():
                    cached_df = store.load_bars(sym, timeframe="1d", start=start_utc, end=end_utc)
                    if not cached_df.empty:
                        bars_by_symbol[sym] = cached_df
                        manifest[sym] = {
                            "status": "ok",
                            "bars": len(cached_df),
                            "first": str(cached_df["timestamp"].min()),
                            "last": str(cached_df["timestamp"].max()),
                            "fetched_at": datetime.now(UTC).isoformat(),
                        }
                        manifest_dirty = True
                        continue
        to_fetch.append(sym)

    if not to_fetch:
        if manifest_dirty:
            _flush_manifest()
        return bars_by_symbol

    provider: AlpacaMarketDataProvider | None = None
    symbols_processed = 0

    try:
        provider = AlpacaMarketDataProvider(
            api_key=api_key,
            secret_key=secret_key,
            feed="sip",
            batch_size=batch_size,
        )

        for idx in range(0, len(to_fetch), batch_size):
            batch = to_fetch[idx : idx + batch_size]
            batch_bars: dict[str, pd.DataFrame] = {}
            batch_success = False

            for attempt in range(1, 4):
                try:
                    batch_bars = provider.fetch_bars(batch, start_utc, end_utc, timeframe="1d")
                    batch_success = True
                    break
                except Exception as exc:  # noqa: BLE001
                    if _is_rate_limit(exc) and attempt < 3:
                        sleep_fn(1.5 * attempt)
                        continue
                    logger.warning(
                        "Batch fetch failed (%s); falling back to single-symbol fetches",
                        exc,
                    )
                    break

            failed_syms: dict[str, str] = {}
            if not batch_success:
                batch_bars = {}
                for sym in batch:
                    sym_bars: dict[str, pd.DataFrame] = {}
                    sym_exc: Exception | None = None
                    for attempt in range(1, 4):
                        try:
                            sym_bars = provider.fetch_bars(
                                [sym], start_utc, end_utc, timeframe="1d"
                            )
                            sym_exc = None
                            break
                        except Exception as exc:  # noqa: BLE001
                            sym_exc = exc
                            if _is_rate_limit(exc) and attempt < 3:
                                sleep_fn(1.5 * attempt)
                                continue
                            logger.warning("Single-symbol fetch failed for %s: %s", sym, exc)
                            break
                    if sym_exc is not None:
                        failed_syms[sym] = str(sym_exc)
                    elif sym in sym_bars:
                        batch_bars[sym] = sym_bars[sym]

            for sym in batch:
                if sym in failed_syms:
                    manifest[sym] = {
                        "status": "error",
                        "error": failed_syms[sym],
                        "bars": 0,
                        "first": None,
                        "last": None,
                        "fetched_at": datetime.now(UTC).isoformat(),
                    }
                    manifest_dirty = True
                else:
                    df = batch_bars.get(sym)
                    if df is not None and not df.empty:
                        store.save_bars(sym, df, timeframe="1d", allow_overwrite=True)
                        bars_by_symbol[sym] = df
                        manifest[sym] = {
                            "status": "ok",
                            "bars": len(df),
                            "first": str(df["timestamp"].min()),
                            "last": str(df["timestamp"].max()),
                            "fetched_at": datetime.now(UTC).isoformat(),
                        }
                    else:
                        manifest[sym] = {
                            "status": "empty",
                            "bars": 0,
                            "first": None,
                            "last": None,
                            "fetched_at": datetime.now(UTC).isoformat(),
                        }
                    manifest_dirty = True
                symbols_processed += 1
                if symbols_processed >= 100:
                    _flush_manifest()
                    manifest_dirty = False
                    symbols_processed = 0
    finally:
        try:
            if manifest_dirty:
                _flush_manifest()
        finally:
            if provider is not None:
                client = getattr(provider, "_client", None)
                session = getattr(client, "_session", None)
                if session is not None and hasattr(session, "close"):
                    session.close()

    return bars_by_symbol


def fetch_and_cache_fred_macro(
    research_dir: Path,
    *,
    fred_api_key: str,
    start_date: date = date(2015, 1, 1),
    end_date: date = date(2026, 11, 20),
    refresh: bool = False,
    provider: FREDMacroProvider | None = None,
) -> list[MacroPoint]:
    """Fetch and cache FRED macroeconomic time series offline without touching state.db."""
    validate_research_dir(research_dir)
    cache_path = research_dir / "fred_macro_2015_2026.json"

    if not refresh and cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return [MacroPoint.model_validate(p) for p in raw]

    active_provider = provider or FREDMacroProvider(api_key=fred_api_key)
    points = active_provider.fetch_macro(start_date=start_date, end_date=end_date)
    serialized = json.dumps([p.model_dump(mode="json") for p in points], indent=2)
    atomic_write_text(cache_path, serialized)
    return points


@dataclass(slots=True)
class _CandidateArrayBundle:
    candidate: V3EquityCandidate
    timestamps_ns: np.ndarray
    timestamps_dt: list[datetime]
    closes: np.ndarray
    volumes: np.ndarray


def build_dynamic_pit_universe_schedule(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    pool: Sequence[V3EquityCandidate],
    *,
    benchmark_symbol: str = "SPY",
    top_n: int = 200,
    hysteresis_buffer: int = 50,
    min_price: float = 10.0,
    min_history_bars: int = 252,
    mddv_window: int = 60,
    max_stale_calendar_days: int = 5,
    mode: Literal["broad_mddv", "pit_sp500_mddv"] = "broad_mddv",
    start_date: date | None = None,
) -> dict[date, frozenset[str]]:
    """Build point-in-time monthly dynamic universe schedule with MDDV ranking and turnover hysteresis."""
    if mode not in ("broad_mddv", "pit_sp500_mddv"):
        raise ValueError(f"Unsupported universe mode: {mode!r}")

    bm_sym = benchmark_symbol.strip().upper()
    spy_df = bars_by_symbol.get(bm_sym)
    if spy_df is None or spy_df.empty:
        raise ValueError(f"Benchmark symbol {bm_sym!r} missing or empty in bars_by_symbol")

    spy_ts_series = pd.to_datetime(spy_df["timestamp"], utc=True, errors="coerce")
    valid_spy = spy_ts_series.notna()
    if not valid_spy.any():
        raise ValueError(f"Benchmark symbol {bm_sym!r} has no valid timestamps")
    spy_df_clean = spy_df.loc[valid_spy].copy()
    spy_df_clean["timestamp"] = spy_ts_series[valid_spy]
    spy_df_clean = spy_df_clean.sort_values("timestamp").reset_index(drop=True)
    spy_dates = [ts.date() for ts in spy_df_clean["timestamp"]]
    spy_timestamps = spy_df_clean["timestamp"].to_list()

    first_trading_days: list[tuple[int, date, datetime]] = []
    seen_months: set[tuple[int, int]] = set()

    for idx, (d, ts) in enumerate(zip(spy_dates, spy_timestamps)):
        month_key = (d.year, d.month)
        if month_key not in seen_months:
            seen_months.add(month_key)
            if idx + 1 >= min_history_bars:
                first_trading_days.append((idx, d, ts))

    deduped_pool: list[V3EquityCandidate] = []
    seen_pool_syms: set[str] = set()
    for c in pool:
        norm_sym = normalize_equity_symbol(c.symbol)
        if not norm_sym or norm_sym in seen_pool_syms:
            continue
        seen_pool_syms.add(norm_sym)
        if c.symbol != norm_sym:
            c = c.model_copy(update={"symbol": norm_sym})
        deduped_pool.append(c)

    candidate_bundles: list[_CandidateArrayBundle] = []

    for c in deduped_pool:
        sym = c.symbol
        if sym == bm_sym or sym in V3_CORE_ETFS:
            continue
        df = bars_by_symbol.get(sym)
        if df is None or df.empty:
            continue
        ts_series = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        valid_mask = ts_series.notna()
        if not valid_mask.any():
            continue
        df_clean = df.loc[valid_mask].copy()
        df_clean["timestamp"] = ts_series[valid_mask]
        df_sorted = df_clean.sort_values("timestamp").reset_index(drop=True)
        ts_ns = df_sorted["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        ts_dt = df_sorted["timestamp"].to_list()
        c_arr = df_sorted["close"].to_numpy(dtype=float)
        v_arr = df_sorted["volume"].to_numpy(dtype=float)
        candidate_bundles.append(
            _CandidateArrayBundle(
                candidate=c,
                timestamps_ns=ts_ns,
                timestamps_dt=ts_dt,
                closes=c_arr,
                volumes=v_arr,
            )
        )

    full_schedule: dict[date, frozenset[str]] = {}
    prev_members: frozenset[str] = frozenset()
    stale_delta = timedelta(days=max_stale_calendar_days)

    for _, r_date, r_ts in first_trading_days:
        r_ts_ns = int(pd.Timestamp(r_ts).value)
        eligible: list[tuple[str, float]] = []

        for bundle in candidate_bundles:
            cand = bundle.candidate
            sym = cand.symbol

            if mode == "pit_sp500_mddv" and not is_in_sp500_on(cand, r_date):
                continue

            idx = int(np.searchsorted(bundle.timestamps_ns, r_ts_ns, side="right"))
            if idx < min_history_bars or idx < mddv_window:
                continue

            last_bar_dt = bundle.timestamps_dt[idx - 1]
            if (r_ts - last_bar_dt) > stale_delta:
                continue

            last_close = bundle.closes[idx - 1]
            if last_close < min_price:
                continue

            closes_win = bundle.closes[idx - mddv_window : idx]
            volumes_win = bundle.volumes[idx - mddv_window : idx]

            if not np.isfinite(closes_win).all() or (closes_win <= 0.0).any():
                continue
            if not np.isfinite(volumes_win).all() or (volumes_win < 0.0).any():
                continue

            mddv = float(np.median(closes_win * volumes_win))
            if not math.isfinite(mddv) or mddv < 0.0:
                continue
            eligible.append((sym, mddv))

        eligible.sort(key=lambda item: (-item[1], item[0]))
        ranked_symbols = [sym for sym, _ in eligible]

        kept_prev: list[str] = []
        for rank, sym in enumerate(ranked_symbols):
            if sym in prev_members and rank < (top_n + hysteresis_buffer):
                kept_prev.append(sym)
                if len(kept_prev) == top_n:
                    break

        selected = set(kept_prev)
        for sym in ranked_symbols:
            if len(selected) >= top_n:
                break
            selected.add(sym)

        selected_frozen = frozenset(selected)
        full_schedule[r_date] = selected_frozen
        prev_members = selected_frozen

    if start_date is not None:
        anchor_date = max((d for d in full_schedule if d <= start_date), default=None)
        return {
            d: members
            for d, members in full_schedule.items()
            if (anchor_date is not None and d == anchor_date) or d > start_date
        }

    return full_schedule


def build_live_dynamic_universe(
    trading_client: Any,
    data_provider: AlpacaMarketDataProvider,
    *,
    top_n: int = 200,
    min_price: float = 10.0,
    as_of: datetime | None = None,
    sector_lookup: Mapping[str, str] | None = None,
) -> list[UniverseSymbol]:
    """Fetch active NYSE/NASDAQ US equities, filter non-common shares, rank by 60d MDDV, and return UniverseSymbol list."""
    try:
        req = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
        raw_assets = trading_client.get_all_assets(req)
    except Exception:  # noqa: BLE001
        raw_assets = trading_client.get_all_assets()

    valid_symbols: list[str] = []
    seen_symbols: set[str] = set()
    for asset in raw_assets:
        raw_sym = str(getattr(asset, "symbol", "")).strip()
        if not raw_sym:
            continue
        sym = normalize_equity_symbol(raw_sym)
        if not sym or sym in V3_CORE_ETFS or sym == "SPY":
            continue

        if sym in _DUAL_CLASS_DROP:
            continue

        status = getattr(asset, "status", None)
        if status is not None and str(status).lower() not in ("active", "assetstatus.active"):
            continue

        tradable = getattr(asset, "tradable", None)
        if tradable is False:
            continue

        ac = getattr(asset, "asset_class", None)
        if ac is not None and str(ac).lower() not in ("us_equity", "assetclass.us_equity"):
            continue

        exchange = getattr(asset, "exchange", None)
        ex_str = str(exchange).upper() if exchange is not None else ""
        if not any(ex in ex_str for ex in ("NYSE", "NASDAQ")):
            continue

        attrs = getattr(asset, "attributes", None) or []
        if any(str(a).lower() in ("etf", "ptp") for a in attrs):
            continue

        name = getattr(asset, "name", "") or ""
        if re.search(r"\b(?:warrants?|units?|rights?|preferred|etfs?)\b", name, re.IGNORECASE):
            continue

        is_valid_format = (sym in ("BRK.B", "BF.B")) or (
            len(sym) <= 5 and "." not in sym and "-" not in sym and sym.isalpha()
        )
        if is_valid_format and sym not in seen_symbols:
            seen_symbols.add(sym)
            valid_symbols.append(sym)

    if not valid_symbols:
        return []

    as_of_dt = ensure_utc(as_of) if as_of is not None else datetime.now(UTC)
    start_dt = as_of_dt - timedelta(days=120)
    bars_by_symbol = data_provider.fetch_bars(
        valid_symbols, start=start_dt, end=as_of_dt, timeframe="1d"
    )

    ranked: list[tuple[str, float]] = []
    stale_delta = timedelta(days=5)

    for sym in valid_symbols:
        df = bars_by_symbol.get(sym)
        if df is None or df.empty:
            continue
        ts_series = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        valid_mask = ts_series.notna() & (ts_series <= as_of_dt)
        if not valid_mask.any():
            continue
        df_clean = df.loc[valid_mask].copy()
        df_clean["timestamp"] = ts_series[valid_mask]
        df_sorted = df_clean.sort_values("timestamp").reset_index(drop=True)
        if len(df_sorted) < 60:
            continue

        latest_bar_dt = pd.to_datetime(df_sorted["timestamp"].iloc[-1], utc=True).to_pydatetime()
        if (as_of_dt - latest_bar_dt) > stale_delta:
            continue

        closes = df_sorted["close"].iloc[-60:].to_numpy(dtype=float)
        volumes = df_sorted["volume"].iloc[-60:].to_numpy(dtype=float)
        last_close = closes[-1]
        if last_close < min_price:
            continue
        if not np.isfinite(closes).all() or (closes <= 0.0).any():
            continue
        if not np.isfinite(volumes).all() or (volumes < 0.0).any():
            continue
        mddv = float(np.median(closes * volumes))
        if not math.isfinite(mddv) or mddv < 0.0:
            continue
        ranked.append((sym, mddv))

    ranked.sort(key=lambda item: (-item[1], item[0]))
    selected = ranked[:top_n]

    results: list[UniverseSymbol] = []
    for sym, _ in selected:
        sec = "Equity"
        if sector_lookup and sym in sector_lookup:
            sec = sector_lookup[sym]
        elif sym in HISTORICAL_DELISTED_SECTORS:
            sec = HISTORICAL_DELISTED_SECTORS[sym]
        results.append(
            UniverseSymbol(
                symbol=sym,
                asset_type=AssetType.equity,
                sector=sec,
                shortable_default=True,
            )
        )

    return results
