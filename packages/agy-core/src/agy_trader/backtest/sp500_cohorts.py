from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field

from agy_trader.core.blackout import issuer_share_class_group
from agy_trader.core.config import AppConfig
from agy_trader.core.models import AssetType, MacroPoint, UniverseSymbol, ensure_utc
from agy_trader.data.alpaca_market import AlpacaMarketDataProvider
from agy_trader.storage.parquet_store import ParquetBarStore

from .engine import BacktestEngine
from .metrics import (
    BacktestMetrics,
    _compute_return_stats,
    _extract_period_returns,
)

logger = logging.getLogger(__name__)

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_CUTOFF_DATE = date(2021, 11, 23)
DEFAULT_COHORT_SEED = 20261122

_SECTOR_MAP: dict[str, str] = {
    "Information Technology": "Technology",
    "Health Care": "Healthcare",
}
_PREFERRED_SHARE_CLASS_TICKERS: tuple[str, ...] = (
    "GOOGL",
    "FOXA",
    "NWSA",
    "BRK.B",
    "BRK-B",
    "BF.B",
    "BF-B",
)
_FOOTNOTE_RE = re.compile(r"\[[^\]]*\]")
_PAREN_RE = re.compile(r"\([^)]*\)")
_DATE_FULL_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DATE_YM_RE = re.compile(r"^(\d{4})-(\d{2})$")
_DATE_Y_RE = re.compile(r"^(\d{4})$")


def _normalize_cik(raw: str) -> str:
    """Normalize numeric SEC CIK; return '' for missing/placeholder values like '0' or 'N/A'."""
    cleaned = _FOOTNOTE_RE.sub("", raw or "").strip()
    if not cleaned or not (cleaned.isascii() and cleaned.isdigit()):
        return ""
    return cleaned.lstrip("0")


def validate_research_dir(research_dir: Path, data_dir: Path | None = None) -> Path:
    """Reject writes targeting live production data directories (e.g., data/parquet)."""
    resolved_res = research_dir.resolve()
    protected_parquet_dirs: list[Path] = [Path("data/parquet").resolve()]
    if data_dir is not None:
        resolved_data = data_dir.resolve()
        if resolved_res == resolved_data:
            raise ValueError(
                f"research_dir ({research_dir}) must not equal protected data_dir ({resolved_data})"
            )
        protected_parquet_dirs.append((resolved_data / "parquet").resolve())
    for parquet_dir in protected_parquet_dirs:
        if resolved_res == parquet_dir or parquet_dir in resolved_res.parents:
            raise ValueError(
                f"research_dir ({research_dir}) must not target or live inside {parquet_dir}"
            )
    return research_dir


def map_gics_sector(gics_sector: str) -> str:
    """Normalize Wikipedia GICS sector names to internal sector taxonomy."""
    cleaned = " ".join(gics_sector.split()).strip()
    return _SECTOR_MAP.get(cleaned, cleaned)


def parse_wikipedia_date_added(raw: str) -> date | None:
    """Parse Wikipedia 'Date added' strings flexibly (YYYY-MM-DD, YYYY-MM, YYYY)."""
    cleaned = _FOOTNOTE_RE.sub("", raw or "")
    cleaned = _PAREN_RE.sub("", cleaned)
    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned:
        return None
    if m := _DATE_FULL_RE.match(cleaned):
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    if m := _DATE_YM_RE.match(cleaned):
        try:
            return date(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            return None
    if m := _DATE_Y_RE.match(cleaned):
        try:
            return date(int(m.group(1)), 1, 1)
        except ValueError:
            return None
    return None


class Sp500Constituent(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    symbol: str
    security: str
    gics_sector: str
    sector: str
    date_added: date | None = None
    cik: str = ""


class SeriesRiskMetrics(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    total_return_pct: float
    cagr_pct: float
    ann_volatility_pct: float
    sharpe_ratio: float
    excess_sharpe_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float


class CohortRunSummary(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    cohort_id: int
    label: str
    symbols: list[str]
    sector_counts: dict[str, int]
    metrics: BacktestMetrics
    ew_buy_hold_return_pct: float
    ew_buy_hold_cagr_pct: float
    ew_buy_hold_sharpe: float
    ew_buy_hold_excess_sharpe: float = 0.0
    ew_buy_hold_ann_volatility_pct: float = 0.0
    ew_buy_hold_calmar_ratio: float = 0.0
    ew_buy_hold_max_drawdown_pct: float
    ew_cagr_pct: float
    ew_excess_sharpe: float = 0.0
    ew_ann_volatility_pct: float = 0.0
    ew_calmar_ratio: float = 0.0
    benchmark_cagr_pct: float
    benchmark_excess_sharpe: float = 0.0
    benchmark_ann_volatility_pct: float = 0.0
    benchmark_calmar_ratio: float = 0.0
    annual_returns: dict[str, dict[str, float]] = Field(default_factory=dict)


class DistributionStats(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    mean: float
    median: float
    std: float
    p10: float
    p25: float
    p75: float
    p90: float
    min: float
    max: float


class SeriesAggregateStats(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    cagr_pct: DistributionStats
    total_return_pct: DistributionStats
    sharpe_ratio: DistributionStats
    excess_sharpe_ratio: DistributionStats
    ann_volatility_pct: DistributionStats
    max_drawdown_pct: DistributionStats
    calmar_ratio: DistributionStats


class WinRateStats(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    total_cohorts: int
    beat_spy_return: int
    beat_spy_cagr: int
    beat_spy_sharpe: int
    beat_spy_excess_sharpe: int
    beat_spy_max_drawdown: int
    beat_ew_return: int
    beat_ew_cagr: int
    beat_ew_sharpe: int
    beat_ew_excess_sharpe: int
    beat_ew_max_drawdown: int
    beat_ew_bh_return: int
    beat_ew_bh_cagr: int
    beat_ew_bh_sharpe: int
    beat_ew_bh_excess_sharpe: int
    beat_ew_bh_max_drawdown: int


class Sp500CohortStudyReport(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    start_date: str
    end_date: str
    raw_bar_start: str
    raw_bar_end: str
    trading_days: int
    cutoff_date: str
    total_wikipedia_constituents: int
    pre_cutoff_constituents: int
    deduped_issuer_constituents: int
    active_eligible_pool_size: int
    unique_symbols_across_cohorts: int
    n_cohorts: int
    cohort_size: int
    seed: int
    execution_timing: str
    max_drawdown_halt_pct: float = 14.0
    top_k_short: int = 0
    cohorts: list[CohortRunSummary]
    strategy_stats: SeriesAggregateStats
    ew_cohort_stats: SeriesAggregateStats
    ew_buy_hold_stats: SeriesAggregateStats
    spy_stats: SeriesAggregateStats
    win_rates: WinRateStats
    plot_paths: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class CohortTimeSeriesBundle:
    timestamps: list[datetime]
    strategy_equities: np.ndarray
    ew_cohort_equities: np.ndarray
    ew_buy_hold_equities: np.ndarray
    spy_equities: np.ndarray
    cohort_labels: list[str]


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def parse_wikipedia_sp500_html(html: str) -> list[Sp500Constituent]:
    """Parse Wikipedia's S&P 500 constituents table (<table id='constituents'>)."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="constituents")
    if table is None:
        raise ValueError("Could not find <table id='constituents'> in S&P 500 HTML")

    rows = table.find_all("tr")
    if not rows:
        raise ValueError("S&P 500 constituents table contains no rows")

    header_cells = [
        _FOOTNOTE_RE.sub("", th.get_text(" ", strip=True)).strip().lower()
        for th in rows[0].find_all(["th", "td"])
    ]

    def _find_col(candidates: tuple[str, ...], fallback: int) -> int:
        for idx, name in enumerate(header_cells):
            if any(cand in name for cand in candidates):
                return idx
        return fallback

    col_symbol = _find_col(("symbol", "ticker"), 0)
    col_security = _find_col(("security", "company"), 1)
    col_sector = _find_col(("gics sector", "sector"), 2)
    col_added = _find_col(("date added", "date first added", "added"), 5)
    col_cik = _find_col(("cik",), 6)

    constituents: list[Sp500Constituent] = []
    seen_symbols: set[str] = set()
    for tr in rows[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) <= max(col_symbol, col_security, col_sector):
            continue

        raw_sym = _FOOTNOTE_RE.sub("", cells[col_symbol].get_text(" ", strip=True))
        symbol = raw_sym.strip().upper()
        if not symbol or symbol in seen_symbols:
            continue

        security = _FOOTNOTE_RE.sub("", cells[col_security].get_text(" ", strip=True)).strip()
        gics_sector = _FOOTNOTE_RE.sub("", cells[col_sector].get_text(" ", strip=True)).strip()
        sector = map_gics_sector(gics_sector)
        raw_added = cells[col_added].get_text(" ", strip=True) if col_added < len(cells) else ""
        date_added = parse_wikipedia_date_added(raw_added)
        raw_cik = cells[col_cik].get_text(" ", strip=True) if col_cik < len(cells) else ""
        cik = _normalize_cik(raw_cik)

        seen_symbols.add(symbol)
        constituents.append(
            Sp500Constituent(
                symbol=symbol,
                security=security,
                gics_sector=gics_sector,
                sector=sector,
                date_added=date_added,
                cik=cik,
            )
        )

    if not constituents:
        raise ValueError("Parsed 0 constituents from S&P 500 table")
    return constituents


def load_or_fetch_sp500_constituents(
    cache_path: Path,
    refresh: bool = False,
    *,
    url: str = WIKIPEDIA_SP500_URL,
) -> list[Sp500Constituent]:
    """Load cached S&P 500 constituents JSON or fetch & cache from Wikipedia."""
    if not refresh and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return [Sp500Constituent.model_validate(item) for item in payload]

    with httpx.Client(
        timeout=20.0,
        follow_redirects=True,
        headers={"User-Agent": "agy-trader-research/1.0"},
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        constituents = parse_wikipedia_sp500_html(response.text)

    serialized = json.dumps(
        [c.model_dump(mode="json") for c in constituents],
        indent=2,
    )
    atomic_write_text(cache_path, serialized)
    return constituents


def _pick_preferred_share_class(group: Sequence[Sp500Constituent]) -> Sp500Constituent:
    by_sym = {c.symbol: c for c in group}
    for preferred in _PREFERRED_SHARE_CLASS_TICKERS:
        if preferred in by_sym:
            return by_sym[preferred]
    return min(group, key=lambda c: c.symbol)


def select_point_in_time_candidates(
    constituents: Sequence[Sp500Constituent],
    *,
    cutoff_date: date = DEFAULT_CUTOFF_DATE,
) -> tuple[list[Sp500Constituent], int]:
    """Filter constituents added on/before cutoff_date and deduplicate multi-share-class issuers."""
    pre_cutoff = [
        c for c in constituents if c.date_added is not None and c.date_added <= cutoff_date
    ]
    pre_cutoff_count = len(pre_cutoff)

    # First deduplicate by share-class equivalence group, then by normalized non-empty CIK
    by_share_group: dict[frozenset[str], list[Sp500Constituent]] = {}
    for c in pre_cutoff:
        grp = issuer_share_class_group(c.symbol)
        by_share_group.setdefault(grp, []).append(c)

    stage_one = [_pick_preferred_share_class(grp_items) for grp_items in by_share_group.values()]

    by_cik: dict[str, list[Sp500Constituent]] = {}
    no_cik: list[Sp500Constituent] = []
    for c in stage_one:
        clean_cik = _normalize_cik(c.cik)
        if clean_cik:
            by_cik.setdefault(clean_cik, []).append(c)
        else:
            no_cik.append(c)

    deduped = [_pick_preferred_share_class(grp_items) for grp_items in by_cik.values()]
    deduped.extend(no_cik)
    deduped.sort(key=lambda c: c.symbol)
    return deduped, pre_cutoff_count


def _fetch_batches_with_fallback(
    provider: AlpacaMarketDataProvider,
    store: ParquetBarStore,
    symbols_to_fetch: Sequence[str],
    start_utc: datetime,
    end_utc: datetime,
    batch_size: int,
) -> dict[str, pd.DataFrame]:
    fetched: dict[str, pd.DataFrame] = {}
    for idx in range(0, len(symbols_to_fetch), batch_size):
        batch = list(symbols_to_fetch[idx : idx + batch_size])
        try:
            batch_bars = provider.fetch_bars(batch, start_utc, end_utc, timeframe="1d")
        except Exception as exc:  # noqa: BLE001 - best-effort batch fallback to single-symbol
            logger.warning(
                "Batch bar fetch failed (%s); falling back to single-symbol fetches",
                exc,
            )
            batch_bars = {}
            for sym in batch:
                try:
                    batch_bars.update(
                        provider.fetch_bars([sym], start_utc, end_utc, timeframe="1d")
                    )
                except Exception as sym_exc:  # noqa: BLE001 - best-effort single-symbol skip
                    logger.warning("Single-symbol bar fetch failed for %s: %s", sym, sym_exc)

        for sym, df in batch_bars.items():
            if df is not None and not df.empty:
                store.save_bars(sym, df, timeframe="1d", allow_overwrite=True)
                fetched[sym] = df
    return fetched


def fetch_and_cache_sp500_bars(
    symbols: Sequence[str],
    research_dir: Path,
    *,
    api_key: str,
    secret_key: str,
    feed: Literal["iex", "sip"] = "iex",
    start_utc: datetime,
    end_utc: datetime,
    benchmark_symbol: str = "SPY",
    refresh: bool = False,
    batch_size: int = 25,
) -> dict[str, pd.DataFrame]:
    """Fetch uncached S&P 500 daily bars from Alpaca into research_dir (never data/parquet/)."""
    validate_research_dir(research_dir)
    start_ts = pd.Timestamp(ensure_utc(start_utc))
    end_ts = pd.Timestamp(ensure_utc(end_utc))
    bm_sym = benchmark_symbol.strip().upper()
    constituent_symbols = list(
        dict.fromkeys(s.strip().upper() for s in symbols if s.strip() and s.strip().upper() != bm_sym)
    )

    store = ParquetBarStore(research_dir)
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    provider: AlpacaMarketDataProvider | None = None

    def _get_provider() -> AlpacaMarketDataProvider:
        nonlocal provider
        if provider is None:
            if not api_key.strip() or not secret_key.strip():
                raise ValueError("Alpaca credentials are required to fetch uncached market bars.")
            provider = AlpacaMarketDataProvider(
                api_key=api_key,
                secret_key=secret_key,
                feed=feed,
                batch_size=batch_size,
            )
        return provider

    try:
        # 1. Resolve or fetch benchmark first so all constituent caches align to its span
        if not refresh:
            bm_cached = store.load_bars(bm_sym, timeframe="1d", start=start_utc, end=end_utc)
            if (
                not bm_cached.empty
                and bm_cached["timestamp"].min() <= start_ts + pd.Timedelta(days=14)
                and bm_cached["timestamp"].max() >= end_ts - pd.Timedelta(days=7)
            ):
                bars_by_symbol[bm_sym] = bm_cached

        if bm_sym not in bars_by_symbol:
            fetched_bm = _fetch_batches_with_fallback(
                _get_provider(), store, [bm_sym], start_utc, end_utc, batch_size
            )
            bars_by_symbol.update(fetched_bm)

        spy_df = bars_by_symbol.get(bm_sym)
        if spy_df is None or spy_df.empty:
            raise ValueError(f"Benchmark bars missing or empty for {bm_sym!r}")

        spy_ts = pd.to_datetime(spy_df["timestamp"], utc=True)
        spy_min_ts = spy_ts.min()
        spy_max_ts = spy_ts.max()

        # 2. Check constituent caches against benchmark min/max timestamps (4d tolerance)
        to_fetch: list[str] = []
        for sym in constituent_symbols:
            if not refresh:
                cached = store.load_bars(sym, timeframe="1d", start=start_utc, end=end_utc)
                if (
                    not cached.empty
                    and cached["timestamp"].min() <= spy_min_ts + pd.Timedelta(days=4)
                    and cached["timestamp"].max() >= spy_max_ts - pd.Timedelta(days=4)
                ):
                    bars_by_symbol[sym] = cached
                    continue
            to_fetch.append(sym)

        if to_fetch:
            fetched_syms = _fetch_batches_with_fallback(
                _get_provider(), store, to_fetch, start_utc, end_utc, batch_size
            )
            bars_by_symbol.update(fetched_syms)
    finally:
        if provider is not None:
            client = getattr(provider, "_client", None)
            session = getattr(client, "_session", None)
            if session is not None and hasattr(session, "close"):
                session.close()

    return bars_by_symbol


def filter_active_for_duration(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    candidates: Sequence[Sp500Constituent],
    *,
    benchmark_symbol: str = "SPY",
    min_coverage_ratio: float = 0.995,
    max_start_lag_days: int = 4,
    max_end_lag_days: int = 4,
    max_internal_gap_days: int = 10,
) -> list[Sp500Constituent]:
    """Keep candidates continuously active with clean OHLC prices over the benchmark span."""
    spy_df = bars_by_symbol.get(benchmark_symbol)
    if spy_df is None or spy_df.empty:
        raise ValueError(f"Benchmark bars missing or empty for {benchmark_symbol!r}")

    spy_ts = (
        pd.to_datetime(spy_df["timestamp"], utc=True)
        .drop_duplicates(keep="last")
        .sort_values()
        .reset_index(drop=True)
    )
    spy_ts_set = set(spy_ts.tolist())
    spy_first = spy_ts.iloc[0]
    spy_last = spy_ts.iloc[-1]
    n_spy = len(spy_ts)
    min_required = math.ceil(min_coverage_ratio * n_spy)

    eligible: list[Sp500Constituent] = []
    for cand in candidates:
        df = bars_by_symbol.get(cand.symbol)
        if df is None or df.empty:
            continue
        if any(col not in df.columns for col in ("timestamp", "open", "high", "low", "close")):
            continue

        work = df.copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
        if work["timestamp"].isna().any():
            continue
        work = (
            work.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        work = work.loc[work["timestamp"].isin(spy_ts_set)].reset_index(drop=True)
        if len(work) < min_required:
            continue

        ts = work["timestamp"]
        start_lag = (ts.iloc[0] - spy_first).total_seconds() / 86400.0
        end_lag = (spy_last - ts.iloc[-1]).total_seconds() / 86400.0
        if start_lag > max_start_lag_days or end_lag > max_end_lag_days:
            continue

        ohlc = work[["open", "high", "low", "close"]].to_numpy(dtype=float)
        if not np.isfinite(ohlc).all() or not (ohlc > 0.0).all():
            continue

        if len(ts) >= 2:
            max_gap_days = float(ts.diff().dropna().dt.total_seconds().max() / 86400.0)
            if max_gap_days > max_internal_gap_days:
                continue

        eligible.append(cand)

    return eligible


def sample_sp500_cohorts(
    eligible: Sequence[Sp500Constituent],
    *,
    n_cohorts: int = 20,
    cohort_size: int = 50,
    seed: int = DEFAULT_COHORT_SEED,
) -> list[list[Sp500Constituent]]:
    """Deterministically sample n_cohorts distinct subsets of size cohort_size without replacement."""
    if n_cohorts < 1:
        raise ValueError("n_cohorts must be >= 1")
    if cohort_size < 1:
        raise ValueError("cohort_size must be >= 1")
    if len(eligible) < cohort_size:
        raise ValueError(
            f"Eligible symbol pool ({len(eligible)}) is smaller than cohort_size ({cohort_size})"
        )

    ordered = sorted(eligible, key=lambda c: c.symbol)
    rng = np.random.default_rng(seed)

    cohorts: list[list[Sp500Constituent]] = []
    seen: set[frozenset[str]] = set()
    max_attempts = max(n_cohorts * 50, 100)
    attempts = 0
    while len(cohorts) < n_cohorts:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                f"Unable to sample {n_cohorts} distinct cohorts of size {cohort_size} "
                f"from {len(ordered)} eligible symbols"
            )
        chosen_idx = rng.choice(len(ordered), size=cohort_size, replace=False)
        picked = sorted((ordered[int(i)] for i in chosen_idx), key=lambda c: c.symbol)
        key = frozenset(c.symbol for c in picked)
        if key in seen:
            continue
        seen.add(key)
        cohorts.append(picked)

    return cohorts


def _compute_series_risk_metrics(
    equities: Sequence[float] | np.ndarray,
    daily_rf_rates: Sequence[float] | None = None,
) -> SeriesRiskMetrics:
    arr = np.asarray(equities, dtype=float)
    if arr.size < 2 or arr[0] <= 0.0:
        return SeriesRiskMetrics(
            total_return_pct=0.0,
            cagr_pct=0.0,
            ann_volatility_pct=0.0,
            sharpe_ratio=0.0,
            excess_sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            calmar_ratio=0.0,
        )

    eq_list: list[float] = arr.tolist()
    rets, _, _ = _extract_period_returns(eq_list)
    cagr, ann_vol, sharpe, _, max_dd = _compute_return_stats(eq_list)
    total_ret = (float(arr[-1]) / float(arr[0])) - 1.0
    calmar = (cagr / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0

    if daily_rf_rates is not None and rets.size > 0:
        rf_arr = np.asarray(daily_rf_rates, dtype=float)
        if rf_arr.size == len(arr):
            rf_aligned = rf_arr[1 : 1 + rets.size]
        elif rf_arr.size >= rets.size:
            rf_aligned = rf_arr[: rets.size]
        else:
            rf_aligned = np.zeros_like(rets)
        rf_aligned = np.where(np.isfinite(rf_aligned), rf_aligned, 0.0)
        excess_rets = rets - rf_aligned
    else:
        excess_rets = rets

    ex_std = float(np.std(excess_rets, ddof=1)) if excess_rets.size > 1 else 0.0
    if excess_rets.size > 1 and ex_std > 1e-7:
        excess_sharpe = float(np.mean(excess_rets)) / ex_std * math.sqrt(252.0)
    else:
        excess_sharpe = 0.0

    return SeriesRiskMetrics(
        total_return_pct=round(total_ret * 100.0, 2),
        cagr_pct=round(cagr * 100.0, 2),
        ann_volatility_pct=round(ann_vol * 100.0, 2),
        sharpe_ratio=round(sharpe, 3),
        excess_sharpe_ratio=round(excess_sharpe, 3),
        max_drawdown_pct=round(max_dd * 100.0, 2),
        calmar_ratio=round(calmar, 3),
    )


def _compute_distribution_stats(values: Sequence[float]) -> DistributionStats:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return DistributionStats(
            mean=0.0,
            median=0.0,
            std=0.0,
            p10=0.0,
            p25=0.0,
            p75=0.0,
            p90=0.0,
            min=0.0,
            max=0.0,
        )
    std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return DistributionStats(
        mean=round(float(np.mean(arr)), 4),
        median=round(float(np.median(arr)), 4),
        std=round(std_val, 4),
        p10=round(float(np.percentile(arr, 10)), 4),
        p25=round(float(np.percentile(arr, 25)), 4),
        p75=round(float(np.percentile(arr, 75)), 4),
        p90=round(float(np.percentile(arr, 90)), 4),
        min=round(float(np.min(arr)), 4),
        max=round(float(np.max(arr)), 4),
    )


def _aggregate_series_metrics(items: Sequence[SeriesRiskMetrics]) -> SeriesAggregateStats:
    return SeriesAggregateStats(
        cagr_pct=_compute_distribution_stats([x.cagr_pct for x in items]),
        total_return_pct=_compute_distribution_stats([x.total_return_pct for x in items]),
        sharpe_ratio=_compute_distribution_stats([x.sharpe_ratio for x in items]),
        excess_sharpe_ratio=_compute_distribution_stats([x.excess_sharpe_ratio for x in items]),
        ann_volatility_pct=_compute_distribution_stats([x.ann_volatility_pct for x in items]),
        max_drawdown_pct=_compute_distribution_stats([x.max_drawdown_pct for x in items]),
        calmar_ratio=_compute_distribution_stats([x.calmar_ratio for x in items]),
    )


def _aligned_close_series(
    df: pd.DataFrame,
    timestamps: Sequence[datetime],
) -> np.ndarray:
    ts_col = pd.to_datetime(df["timestamp"], utc=True)
    closes = df["close"].astype(float).tolist()
    lookup: dict[datetime, float] = {
        ensure_utc(ts.to_pydatetime()): float(px)
        for ts, px in zip(ts_col, closes, strict=True)
        if math.isfinite(px) and px > 0.0
    }
    if not lookup:
        raise ValueError("Cannot align empty price series")

    sorted_keys = sorted(lookup.keys())
    out = np.empty(len(timestamps), dtype=float)
    last_px: float | None = None
    for idx, t in enumerate(timestamps):
        px = lookup.get(t)
        if px is None:
            if last_px is not None:
                px = last_px
            else:
                # Carry forward latest close <= t
                prior = [k for k in sorted_keys if k <= t]
                if not prior:
                    raise ValueError(f"No price available on or before {t.isoformat()}")
                px = lookup[prior[-1]]
        last_px = px
        out[idx] = px
    return out


def _compute_annual_returns(
    timestamps: Sequence[datetime],
    series_map: Mapping[str, np.ndarray],
    initial_capital: float,
) -> dict[str, dict[str, float]]:
    if not timestamps:
        return {}

    year_last_idx: dict[int, int] = {}
    for idx, ts in enumerate(timestamps):
        year_last_idx[ts.year] = idx

    years = sorted(year_last_idx.keys())
    annual: dict[str, dict[str, float]] = {}
    prev_end_idx: int | None = None
    for yr in years:
        end_idx = year_last_idx[yr]
        yr_dict: dict[str, float] = {}
        for name, arr in series_map.items():
            anchor = initial_capital if prev_end_idx is None else float(arr[prev_end_idx])
            end_val = float(arr[end_idx])
            ret_pct = ((end_val / anchor) - 1.0) * 100.0 if anchor > 0.0 else 0.0
            yr_dict[name] = round(ret_pct, 2)
        annual[str(yr)] = yr_dict
        prev_end_idx = end_idx
    return annual


def _clip_df_to_timestamps(df: pd.DataFrame, valid_ts_set: set[Any]) -> pd.DataFrame:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work = (
        work.drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return work.loc[work["timestamp"].isin(valid_ts_set)].reset_index(drop=True)


def run_sp500_cohort_study(
    *,
    app_config: AppConfig,
    constituents_total_count: int,
    pre_cutoff_count: int,
    deduped_count: int,
    eligible: Sequence[Sp500Constituent],
    bars_by_symbol: Mapping[str, pd.DataFrame],
    macro_points: Sequence[MacroPoint] | None = None,
    n_cohorts: int = 20,
    cohort_size: int = 50,
    seed: int = DEFAULT_COHORT_SEED,
    cutoff_date: date = DEFAULT_CUTOFF_DATE,
    execution_timing: Literal["same_close", "next_open"] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[Sp500CohortStudyReport, CohortTimeSeriesBundle]:
    """Run n_cohorts S&P 500 backtests and aggregate cohort vs benchmark statistics."""
    benchmark_symbol = app_config.backtest.benchmark_symbol
    raw_spy_df = bars_by_symbol.get(benchmark_symbol)
    if raw_spy_df is None or raw_spy_df.empty:
        raise ValueError(f"Benchmark bars missing for {benchmark_symbol!r}")

    spy_df = raw_spy_df.copy()
    spy_df["timestamp"] = pd.to_datetime(spy_df["timestamp"], utc=True)
    spy_df = (
        spy_df.drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    spy_ts_set = set(spy_df["timestamp"].tolist())

    cohorts = sample_sp500_cohorts(
        eligible,
        n_cohorts=n_cohorts,
        cohort_size=cohort_size,
        seed=seed,
    )

    initial_capital = float(app_config.backtest.initial_capital)
    eff_timing = execution_timing or app_config.backtest.execution_timing

    ref_timestamps: list[datetime] | None = None
    strat_matrix: list[np.ndarray] = []
    ew_matrix: list[np.ndarray] = []
    ew_bh_matrix: list[np.ndarray] = []
    cohort_summaries: list[CohortRunSummary] = []

    strat_series_metrics: list[SeriesRiskMetrics] = []
    ew_series_metrics: list[SeriesRiskMetrics] = []
    ew_bh_series_metrics: list[SeriesRiskMetrics] = []
    spy_series_metrics: list[SeriesRiskMetrics] = []

    for cohort_idx, cohort_members in enumerate(cohorts, start=1):
        label = f"Cohort {cohort_idx:02d}"
        cohort_universe = [
            UniverseSymbol(
                symbol=c.symbol,
                asset_type=AssetType.equity,
                sector=c.sector,
                shortable_default=True,
            )
            for c in cohort_members
        ]
        cohort_bars: dict[str, pd.DataFrame] = {
            c.symbol: _clip_df_to_timestamps(bars_by_symbol[c.symbol], spy_ts_set)
            for c in cohort_members
        }
        cohort_bars[benchmark_symbol] = spy_df

        engine = BacktestEngine(app_config, cohort_universe)
        run_series = engine.run_with_series(
            cohort_bars,
            macro_points=macro_points,
            start_date=start_date,
            end_date=end_date,
            label=label,
            allow_shorting=True,
            use_news=False,
            execution_timing=eff_timing,
            n_trials=n_cohorts,
        )
        res = run_series.result
        if not res.equity_curve:
            raise ValueError(f"{label} produced an empty equity curve")

        timestamps = [ensure_utc(pt.timestamp) for pt in res.equity_curve]
        if ref_timestamps is None:
            ref_timestamps = timestamps
        elif timestamps != ref_timestamps:
            raise ValueError(
                f"{label} trading calendar ({len(timestamps)} bars) does not align "
                f"with Cohort 01 ({len(ref_timestamps)} bars)"
            )

        strat_eq = np.asarray([pt.equity for pt in res.equity_curve], dtype=float)
        ew_eq = np.asarray(run_series.ew_universe_equities, dtype=float)
        spy_closes = _aligned_close_series(spy_df, timestamps)
        spy_eq = initial_capital * (spy_closes / spy_closes[0])

        rel_paths = np.vstack(
            [
                _aligned_close_series(cohort_bars[c.symbol], timestamps)
                for c in cohort_members
            ]
        )
        rel_paths = rel_paths / rel_paths[:, [0]]
        ew_bh_eq = initial_capital * np.mean(rel_paths, axis=0)

        m = res.metrics
        strat_sm = SeriesRiskMetrics(
            total_return_pct=m.total_return_pct,
            cagr_pct=m.cagr_pct,
            ann_volatility_pct=m.ann_volatility_pct,
            sharpe_ratio=m.sharpe_ratio,
            excess_sharpe_ratio=m.excess_sharpe_ratio,
            max_drawdown_pct=m.max_drawdown_pct,
            calmar_ratio=m.calmar_ratio,
        )
        ew_sm = _compute_series_risk_metrics(ew_eq, run_series.daily_rf_rates)
        ew_bh_sm = _compute_series_risk_metrics(ew_bh_eq, run_series.daily_rf_rates)
        spy_sm = _compute_series_risk_metrics(spy_eq, run_series.daily_rf_rates)

        annual_returns = _compute_annual_returns(
            timestamps,
            {
                "strategy": strat_eq,
                "ew_cohort": ew_eq,
                "ew_buy_hold": ew_bh_eq,
                "spy": spy_eq,
            },
            initial_capital=initial_capital,
        )

        sector_counts = dict(sorted(Counter(c.sector for c in cohort_members).items()))
        cohort_summaries.append(
            CohortRunSummary(
                cohort_id=cohort_idx,
                label=label,
                symbols=[c.symbol for c in cohort_members],
                sector_counts=sector_counts,
                metrics=m,
                ew_buy_hold_return_pct=ew_bh_sm.total_return_pct,
                ew_buy_hold_cagr_pct=ew_bh_sm.cagr_pct,
                ew_buy_hold_sharpe=ew_bh_sm.sharpe_ratio,
                ew_buy_hold_excess_sharpe=ew_bh_sm.excess_sharpe_ratio,
                ew_buy_hold_ann_volatility_pct=ew_bh_sm.ann_volatility_pct,
                ew_buy_hold_calmar_ratio=ew_bh_sm.calmar_ratio,
                ew_buy_hold_max_drawdown_pct=ew_bh_sm.max_drawdown_pct,
                ew_cagr_pct=ew_sm.cagr_pct,
                ew_excess_sharpe=ew_sm.excess_sharpe_ratio,
                ew_ann_volatility_pct=ew_sm.ann_volatility_pct,
                ew_calmar_ratio=ew_sm.calmar_ratio,
                benchmark_cagr_pct=spy_sm.cagr_pct,
                benchmark_excess_sharpe=spy_sm.excess_sharpe_ratio,
                benchmark_ann_volatility_pct=spy_sm.ann_volatility_pct,
                benchmark_calmar_ratio=spy_sm.calmar_ratio,
                annual_returns=annual_returns,
            )
        )

        strat_series_metrics.append(strat_sm)
        ew_series_metrics.append(ew_sm)
        ew_bh_series_metrics.append(ew_bh_sm)
        spy_series_metrics.append(spy_sm)
        strat_matrix.append(strat_eq)
        ew_matrix.append(ew_eq)
        ew_bh_matrix.append(ew_bh_eq)

    assert ref_timestamps is not None
    spy_closes_ref = _aligned_close_series(spy_df, ref_timestamps)
    spy_equities_ref = initial_capital * (spy_closes_ref / spy_closes_ref[0])

    spy_ts = pd.to_datetime(spy_df["timestamp"], utc=True).sort_values()
    raw_bar_start = spy_ts.iloc[0].date().isoformat()
    raw_bar_end = spy_ts.iloc[-1].date().isoformat()
    unique_syms = {sym for cs in cohort_summaries for sym in cs.symbols}

    win_rates = WinRateStats(
        total_cohorts=len(cohort_summaries),
        beat_spy_return=sum(
            1
            for s, b in zip(strat_series_metrics, spy_series_metrics, strict=True)
            if s.total_return_pct > b.total_return_pct
        ),
        beat_spy_cagr=sum(
            1
            for s, b in zip(strat_series_metrics, spy_series_metrics, strict=True)
            if s.cagr_pct > b.cagr_pct
        ),
        beat_spy_sharpe=sum(
            1
            for s, b in zip(strat_series_metrics, spy_series_metrics, strict=True)
            if s.sharpe_ratio > b.sharpe_ratio
        ),
        beat_spy_excess_sharpe=sum(
            1
            for s, b in zip(strat_series_metrics, spy_series_metrics, strict=True)
            if s.excess_sharpe_ratio > b.excess_sharpe_ratio
        ),
        beat_spy_max_drawdown=sum(
            1
            for s, b in zip(strat_series_metrics, spy_series_metrics, strict=True)
            if s.max_drawdown_pct > b.max_drawdown_pct
        ),
        beat_ew_return=sum(
            1
            for s, e in zip(strat_series_metrics, ew_series_metrics, strict=True)
            if s.total_return_pct > e.total_return_pct
        ),
        beat_ew_cagr=sum(
            1
            for s, e in zip(strat_series_metrics, ew_series_metrics, strict=True)
            if s.cagr_pct > e.cagr_pct
        ),
        beat_ew_sharpe=sum(
            1
            for s, e in zip(strat_series_metrics, ew_series_metrics, strict=True)
            if s.sharpe_ratio > e.sharpe_ratio
        ),
        beat_ew_excess_sharpe=sum(
            1
            for s, e in zip(strat_series_metrics, ew_series_metrics, strict=True)
            if s.excess_sharpe_ratio > e.excess_sharpe_ratio
        ),
        beat_ew_max_drawdown=sum(
            1
            for s, e in zip(strat_series_metrics, ew_series_metrics, strict=True)
            if s.max_drawdown_pct > e.max_drawdown_pct
        ),
        beat_ew_bh_return=sum(
            1
            for s, e in zip(strat_series_metrics, ew_bh_series_metrics, strict=True)
            if s.total_return_pct > e.total_return_pct
        ),
        beat_ew_bh_cagr=sum(
            1
            for s, e in zip(strat_series_metrics, ew_bh_series_metrics, strict=True)
            if s.cagr_pct > e.cagr_pct
        ),
        beat_ew_bh_sharpe=sum(
            1
            for s, e in zip(strat_series_metrics, ew_bh_series_metrics, strict=True)
            if s.sharpe_ratio > e.sharpe_ratio
        ),
        beat_ew_bh_excess_sharpe=sum(
            1
            for s, e in zip(strat_series_metrics, ew_bh_series_metrics, strict=True)
            if s.excess_sharpe_ratio > e.excess_sharpe_ratio
        ),
        beat_ew_bh_max_drawdown=sum(
            1
            for s, e in zip(strat_series_metrics, ew_bh_series_metrics, strict=True)
            if s.max_drawdown_pct > e.max_drawdown_pct
        ),
    )

    first_m = cohort_summaries[0].metrics
    report = Sp500CohortStudyReport(
        start_date=first_m.start_date,
        end_date=first_m.end_date,
        raw_bar_start=raw_bar_start,
        raw_bar_end=raw_bar_end,
        trading_days=first_m.trading_days,
        cutoff_date=cutoff_date.isoformat(),
        total_wikipedia_constituents=constituents_total_count,
        pre_cutoff_constituents=pre_cutoff_count,
        deduped_issuer_constituents=deduped_count,
        active_eligible_pool_size=len(eligible),
        unique_symbols_across_cohorts=len(unique_syms),
        n_cohorts=n_cohorts,
        cohort_size=cohort_size,
        seed=seed,
        execution_timing=eff_timing,
        max_drawdown_halt_pct=round(app_config.risk.max_drawdown_halt_pct * 100.0, 2),
        top_k_short=app_config.strategy.top_k_short,
        cohorts=cohort_summaries,
        strategy_stats=_aggregate_series_metrics(strat_series_metrics),
        ew_cohort_stats=_aggregate_series_metrics(ew_series_metrics),
        ew_buy_hold_stats=_aggregate_series_metrics(ew_bh_series_metrics),
        spy_stats=_aggregate_series_metrics([spy_series_metrics[0]]),
        win_rates=win_rates,
    )

    ts_bundle = CohortTimeSeriesBundle(
        timestamps=ref_timestamps,
        strategy_equities=np.vstack(strat_matrix),
        ew_cohort_equities=np.vstack(ew_matrix),
        ew_buy_hold_equities=np.vstack(ew_bh_matrix),
        spy_equities=spy_equities_ref,
        cohort_labels=[c.label for c in cohort_summaries],
    )
    return report, ts_bundle


def _drawdown_matrix(equities: np.ndarray) -> np.ndarray:
    peaks = np.maximum.accumulate(equities, axis=-1)
    safe_peaks = np.where(peaks > 1e-9, peaks, 1.0)
    return ((equities - safe_peaks) / safe_peaks) * 100.0


def _style_axis(ax: Any) -> None:
    ax.set_facecolor("#FAFAFC")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.55, color="#CBD5E1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")


def _format_year_labels(
    years: Sequence[str],
    start_date_str: str,
    end_date_str: str,
) -> list[str]:
    if not years:
        return []
    start_dt = date.fromisoformat(start_date_str)
    end_dt = date.fromisoformat(end_date_str)
    is_partial_first = (start_dt.month, start_dt.day) > (1, 7)
    is_partial_last = (end_dt.month, end_dt.day) < (12, 28)

    labels: list[str] = []
    for idx, yr in enumerate(years):
        if len(years) == 1 and is_partial_first and is_partial_last:
            labels.append(f"{yr}* ({start_dt.strftime('%b %d')}–{end_dt.strftime('%b %d')})")
        elif idx == 0 and is_partial_first:
            labels.append(f"{yr}* (from {start_dt.strftime('%b %d')})")
        elif idx == len(years) - 1 and is_partial_last:
            labels.append(f"{yr} YTD")
        else:
            labels.append(yr)
    return labels


def generate_sp500_cohort_plots(
    report: Sp500CohortStudyReport,
    ts_bundle: CohortTimeSeriesBundle,
    output_dir: Path,
) -> list[Path]:
    """Generate 4 institutional summary PNG charts and close every matplotlib figure."""
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    output_dir.mkdir(parents=True, exist_ok=True)
    c_strat = "#0F4C81"
    c_strat_band = "#1D70B8"
    c_ew = "#D97706"
    c_ew_bh = "#059669"
    c_spy = "#DC2626"
    c_halt = "#475569"
    bg_color = "#FAFAFC"

    dates = [ts.astimezone(UTC).date() for ts in ts_bundle.timestamps]
    strat_eq = ts_bundle.strategy_equities
    ew_eq = ts_bundle.ew_cohort_equities
    ew_bh_eq = ts_bundle.ew_buy_hold_equities
    spy_eq = ts_bundle.spy_equities

    strat_dd = _drawdown_matrix(strat_eq)
    ew_dd = _drawdown_matrix(ew_eq)
    spy_dd = _drawdown_matrix(spy_eq)

    saved_paths: list[Path] = []

    # 1. Equity & Drawdown Fan
    p1 = output_dir / "01_equity_and_drawdown_fan.png"
    fig, (ax_eq, ax_dd) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.14},
    )
    try:
        fig.patch.set_facecolor(bg_color)
        _style_axis(ax_eq)
        _style_axis(ax_dd)

        for row in ew_eq:
            ax_eq.plot(dates, row, color=c_ew, alpha=0.12, linewidth=0.8)
        for row in strat_eq:
            ax_eq.plot(dates, row, color=c_strat, alpha=0.18, linewidth=0.9)

        strat_med = np.median(strat_eq, axis=0)
        strat_p25 = np.percentile(strat_eq, 25, axis=0)
        strat_p75 = np.percentile(strat_eq, 75, axis=0)
        ew_med = np.median(ew_eq, axis=0)
        ew_bh_med = np.median(ew_bh_eq, axis=0)

        ax_eq.fill_between(
            dates,
            strat_p25,
            strat_p75,
            color=c_strat_band,
            alpha=0.18,
            label="Strategy IQR [P25, P75]",
        )
        ax_eq.plot(
            dates,
            strat_med,
            color=c_strat,
            linewidth=2.5,
            label=f"Strategy Median (${strat_med[-1]:,.0f})",
            zorder=5,
        )
        ax_eq.plot(
            dates,
            ew_med,
            color=c_ew,
            linestyle="--",
            linewidth=2.2,
            label=f"EW-{report.cohort_size} Daily Median (${ew_med[-1]:,.0f})",
            zorder=4,
        )
        ax_eq.plot(
            dates,
            ew_bh_med,
            color=c_ew_bh,
            linestyle=":",
            linewidth=1.8,
            label=f"EW-{report.cohort_size} B&H Median (${ew_bh_med[-1]:,.0f})",
            zorder=4,
        )
        ax_eq.plot(
            dates,
            spy_eq,
            color=c_spy,
            linewidth=2.4,
            label=f"SPY Buy & Hold (${spy_eq[-1]:,.0f})",
            zorder=5,
        )

        # De-collided right-edge annotations
        end_items = sorted(
            [
                (float(strat_med[-1]), c_strat, f"Strategy: ${strat_med[-1]:,.0f}"),
                (float(ew_med[-1]), c_ew, f"EW-{report.cohort_size}: ${ew_med[-1]:,.0f}"),
                (float(ew_bh_med[-1]), c_ew_bh, f"EW B&H: ${ew_bh_med[-1]:,.0f}"),
                (float(spy_eq[-1]), c_spy, f"SPY: ${spy_eq[-1]:,.0f}"),
            ],
            key=lambda x: x[0],
        )
        y_lo, y_hi = ax_eq.get_ylim()
        min_gap = max((y_hi - y_lo) * 0.048, 1.0)
        placed_y: list[float] = []
        span_days = max((dates[-1] - dates[0]).days, 30)
        label_pad_days = max(2, int(span_days * 0.012))
        right_pad_days = max(18, int(span_days * 0.14))
        for val, color, text in end_items:
            target_y = val
            if placed_y and target_y - placed_y[-1] < min_gap:
                target_y = placed_y[-1] + min_gap
            placed_y.append(target_y)
            ax_eq.annotate(
                text,
                xy=(dates[-1], val),
                xytext=(dates[-1] + timedelta(days=label_pad_days), target_y),
                color=color,
                fontsize=8.5,
                fontweight="bold",
                va="center",
                arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.45, "lw": 0.8},
            )

        ax_eq.set_xlim(dates[0], dates[-1] + timedelta(days=right_pad_days))
        ax_eq.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        ax_eq.set_ylabel("Portfolio Equity ($)", fontsize=10, fontweight="bold")
        ax_eq.set_title(
            f"S&P 500 {report.n_cohorts}-Cohort Validation ({report.cohort_size} Stocks/Cohort, "
            f"{report.start_date} to {report.end_date}) — Cumulative Equity & Drawdowns",
            fontsize=12.5,
            fontweight="bold",
            pad=10,
        )
        ax_eq.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.5)

        strat_dd_med = np.median(strat_dd, axis=0)
        strat_dd_p10 = np.percentile(strat_dd, 10, axis=0)
        strat_dd_p90 = np.percentile(strat_dd, 90, axis=0)
        ew_dd_med = np.median(ew_dd, axis=0)
        ew_dd_p10 = np.percentile(ew_dd, 10, axis=0)
        ew_dd_p90 = np.percentile(ew_dd, 90, axis=0)

        ax_dd.fill_between(
            dates,
            strat_dd_p10,
            strat_dd_p90,
            color=c_strat_band,
            alpha=0.18,
            label="Strategy [P10, P90] DD",
        )
        ax_dd.fill_between(
            dates,
            ew_dd_p10,
            ew_dd_p90,
            color=c_ew,
            alpha=0.12,
            label=f"EW-{report.cohort_size} [P10, P90] DD",
        )
        ax_dd.plot(dates, strat_dd_med, color=c_strat, linewidth=2.2, label="Strategy Median DD")
        ax_dd.plot(
            dates,
            ew_dd_med,
            color=c_ew,
            linestyle="--",
            linewidth=1.9,
            label=f"EW-{report.cohort_size} Median DD",
        )
        ax_dd.plot(dates, spy_dd, color=c_spy, linewidth=2.0, label="SPY Drawdown")
        halt_pct = abs(report.max_drawdown_halt_pct)
        ax_dd.axhline(
            -halt_pct,
            color=c_halt,
            linestyle="-.",
            linewidth=1.3,
            label=f"Hard Max-DD Halt (-{halt_pct:.1f}%)",
        )
        ax_dd.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax_dd.set_ylabel("Underwater Drawdown (%)", fontsize=10, fontweight="bold")
        ax_dd.xaxis.set_major_locator(mdates.YearLocator())
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_dd.legend(
            loc="lower left",
            ncol=3,
            frameon=True,
            facecolor="white",
            framealpha=0.92,
            fontsize=8.2,
        )

        fig.savefig(p1, dpi=200, bbox_inches="tight")
        saved_paths.append(p1)
    finally:
        plt.close(fig)

    # 2. Per-Cohort Paired Metrics
    p2 = output_dir / "02_per_cohort_paired_metrics.png"
    sorted_cohorts = sorted(report.cohorts, key=lambda c: c.metrics.sharpe_ratio, reverse=True)
    y_pos = np.arange(len(sorted_cohorts))
    labels = [c.label for c in sorted_cohorts]
    spy_ref = sorted_cohorts[0]

    fig, axes = plt.subplots(1, 3, figsize=(17, 9), sharey=True)
    try:
        fig.patch.set_facecolor(bg_color)
        for ax in axes:
            _style_axis(ax)

        h = 0.25
        strat_cagr = [c.metrics.cagr_pct for c in sorted_cohorts]
        ew_cagr = [c.ew_cagr_pct for c in sorted_cohorts]
        bh_cagr = [c.ew_buy_hold_cagr_pct for c in sorted_cohorts]

        axes[0].barh(y_pos - h, strat_cagr, height=h, color=c_strat, label="Strategy")
        axes[0].barh(y_pos, ew_cagr, height=h, color=c_ew, alpha=0.88, label=f"EW-{report.cohort_size} Daily")
        axes[0].barh(y_pos + h, bh_cagr, height=h, color=c_ew_bh, alpha=0.82, label=f"EW-{report.cohort_size} B&H")
        axes[0].axvline(
            spy_ref.benchmark_cagr_pct,
            color=c_spy,
            linestyle="--",
            linewidth=1.8,
            label=f"SPY ({spy_ref.benchmark_cagr_pct:.1f}%)",
        )
        axes[0].set_xlabel("CAGR (%)", fontweight="bold")
        axes[0].set_title("CAGR (%) by Cohort", fontweight="bold")
        axes[0].legend(loc="lower right", fontsize=8, frameon=True, facecolor="white")

        strat_shp = [c.metrics.sharpe_ratio for c in sorted_cohorts]
        ew_shp = [c.metrics.ew_universe_sharpe for c in sorted_cohorts]
        bh_shp = [c.ew_buy_hold_sharpe for c in sorted_cohorts]
        med_ex_shp = report.strategy_stats.excess_sharpe_ratio.median

        axes[1].barh(y_pos - h, strat_shp, height=h, color=c_strat, label="Strategy")
        axes[1].barh(y_pos, ew_shp, height=h, color=c_ew, alpha=0.88, label=f"EW-{report.cohort_size} Daily")
        axes[1].barh(y_pos + h, bh_shp, height=h, color=c_ew_bh, alpha=0.82, label=f"EW-{report.cohort_size} B&H")
        axes[1].axvline(
            spy_ref.metrics.benchmark_sharpe,
            color=c_spy,
            linestyle="--",
            linewidth=1.8,
            label=f"SPY ({spy_ref.metrics.benchmark_sharpe:.2f})",
        )
        axes[1].set_xlabel("Sharpe Ratio (rf=0)", fontweight="bold")
        axes[1].set_title(
            f"Sharpe Ratio (Median Excess Sharpe: {med_ex_shp:.2f})",
            fontweight="bold",
        )
        axes[1].legend(loc="lower right", fontsize=8, frameon=True, facecolor="white")

        strat_mdd = [c.metrics.max_drawdown_pct for c in sorted_cohorts]
        ew_mdd = [c.metrics.ew_universe_max_drawdown_pct for c in sorted_cohorts]
        bh_mdd = [c.ew_buy_hold_max_drawdown_pct for c in sorted_cohorts]

        axes[2].barh(y_pos - h, strat_mdd, height=h, color=c_strat, label="Strategy")
        axes[2].barh(y_pos, ew_mdd, height=h, color=c_ew, alpha=0.88, label=f"EW-{report.cohort_size} Daily")
        axes[2].barh(y_pos + h, bh_mdd, height=h, color=c_ew_bh, alpha=0.82, label=f"EW-{report.cohort_size} B&H")
        axes[2].axvline(
            spy_ref.metrics.benchmark_max_drawdown_pct,
            color=c_spy,
            linestyle="--",
            linewidth=1.8,
            label=f"SPY ({spy_ref.metrics.benchmark_max_drawdown_pct:.1f}%)",
        )
        axes[2].axvline(-halt_pct, color=c_halt, linestyle="-.", linewidth=1.3, label=f"Hard Halt (-{halt_pct:.1f}%)")
        axes[2].set_xlabel("Max Drawdown (%)", fontweight="bold")
        axes[2].set_title("Max Drawdown (%)", fontweight="bold")
        axes[2].legend(loc="lower left", fontsize=8, frameon=True, facecolor="white")

        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(labels, fontsize=9)
        axes[0].invert_yaxis()

        fig.suptitle(
            f"Per-Cohort Paired Performance (Sorted by Strategy Sharpe) — "
            f"Beat SPY CAGR {report.win_rates.beat_spy_cagr}/{report.n_cohorts}, "
            f"Beat EW-{report.cohort_size} Sharpe {report.win_rates.beat_ew_sharpe}/{report.n_cohorts}",
            fontsize=12.5,
            fontweight="bold",
            y=0.98,
        )
        fig.subplots_adjust(top=0.92, wspace=0.10)
        fig.savefig(p2, dpi=200, bbox_inches="tight")
        saved_paths.append(p2)
    finally:
        plt.close(fig)

    # 3. Risk-Return Scatter & Alpha Spreads
    p3 = output_dir / "03_risk_return_and_alpha_spreads.png"
    fig, (ax_sc, ax_sp) = plt.subplots(1, 2, figsize=(15, 6.8))
    try:
        fig.patch.set_facecolor(bg_color)
        _style_axis(ax_sc)
        _style_axis(ax_sp)

        cohorts_list = report.cohorts
        s_vol = np.array([c.metrics.ann_volatility_pct for c in cohorts_list])
        s_cagr = np.array([c.metrics.cagr_pct for c in cohorts_list])
        e_vol = np.array([c.ew_ann_volatility_pct for c in cohorts_list])
        e_cagr = np.array([c.ew_cagr_pct for c in cohorts_list])
        b_vol = np.array([c.ew_buy_hold_ann_volatility_pct for c in cohorts_list])
        b_cagr = np.array([c.ew_buy_hold_cagr_pct for c in cohorts_list])

        for xv, yv, xe, ye in zip(s_vol, s_cagr, e_vol, e_cagr, strict=True):
            ax_sc.plot([xe, xv], [ye, yv], color="#94A3B8", alpha=0.45, linewidth=0.9, zorder=1)

        ax_sc.scatter(e_vol, e_cagr, marker="^", s=55, color=c_ew, alpha=0.85, label=f"EW-{report.cohort_size} Daily", zorder=3)
        ax_sc.scatter(b_vol, b_cagr, marker="D", s=34, color=c_ew_bh, alpha=0.75, label=f"EW-{report.cohort_size} B&H", zorder=3)
        ax_sc.scatter(s_vol, s_cagr, marker="o", s=64, color=c_strat, alpha=0.92, edgecolors="white", linewidth=0.6, label="Strategy", zorder=4)
        ax_sc.scatter(
            [spy_ref.benchmark_ann_volatility_pct],
            [spy_ref.benchmark_cagr_pct],
            marker="*",
            s=230,
            color=c_spy,
            edgecolors="black",
            linewidth=0.6,
            label="SPY",
            zorder=5,
        )

        min_vol_x = min(
            float(np.min(s_vol)),
            float(np.min(e_vol)),
            float(np.min(b_vol)),
            spy_ref.benchmark_ann_volatility_pct,
        )
        max_vol_x = (
            max(
                float(np.max(e_vol)),
                float(np.max(s_vol)),
                float(np.max(b_vol)),
                spy_ref.benchmark_ann_volatility_pct,
                10.0,
            )
            * 1.15
        )
        x_grid = np.linspace(0.0, max_vol_x, 100)
        for ray_shp in (0.75, 1.0, 1.25, 1.5):
            ax_sc.plot(x_grid, ray_shp * x_grid, linestyle="--", linewidth=0.8, color="#94A3B8", alpha=0.6, zorder=0)
            ax_sc.text(
                max_vol_x * 0.96,
                ray_shp * max_vol_x * 0.96,
                f"SR={ray_shp:.2f}",
                fontsize=7.5,
                color="#64748B",
                ha="right",
                va="bottom",
            )

        ax_sc.set_xlim(left=max(0.0, min_vol_x * 0.85), right=max_vol_x)
        ax_sc.set_xlabel("Annualized Volatility (%)", fontweight="bold")
        ax_sc.set_ylabel("CAGR (%)", fontweight="bold")
        ax_sc.set_title("Risk-Return Efficiency vs. Cohort Baselines", fontweight="bold")
        ax_sc.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)

        spread_specs = [
            (
                f"CAGR Spread vs SPY\n({report.win_rates.beat_spy_cagr}/{report.n_cohorts} beat)",
                [c.metrics.cagr_pct - c.benchmark_cagr_pct for c in cohorts_list],
                c_spy,
            ),
            (
                f"CAGR Spread vs EW-{report.cohort_size}\n({report.win_rates.beat_ew_cagr}/{report.n_cohorts} beat)",
                [c.metrics.cagr_pct - c.ew_cagr_pct for c in cohorts_list],
                c_ew,
            ),
            (
                f"CAGR Spread vs EW B&H\n({report.win_rates.beat_ew_bh_cagr}/{report.n_cohorts} beat)",
                [c.metrics.cagr_pct - c.ew_buy_hold_cagr_pct for c in cohorts_list],
                c_ew_bh,
            ),
            (
                f"Max DD Reduction vs SPY\n({report.win_rates.beat_spy_max_drawdown}/{report.n_cohorts} shallower)",
                [c.metrics.max_drawdown_pct - c.metrics.benchmark_max_drawdown_pct for c in cohorts_list],
                c_strat_band,
            ),
            (
                f"Max DD Reduction vs EW-{report.cohort_size}\n({report.win_rates.beat_ew_max_drawdown}/{report.n_cohorts} shallower)",
                [c.metrics.max_drawdown_pct - c.metrics.ew_universe_max_drawdown_pct for c in cohorts_list],
                c_strat,
            ),
        ]

        spread_vals = [vals for _, vals, _ in spread_specs]
        spread_labels = [lbl for lbl, _, _ in spread_specs]
        bp = ax_sp.boxplot(
            spread_vals,
            orientation="horizontal",
            patch_artist=True,
            widths=0.48,
            medianprops={"color": "#111827", "linewidth": 1.8},
            boxprops={"linewidth": 1.1, "color": "#475569"},
            whiskerprops={"linewidth": 1.0, "color": "#475569"},
            capprops={"linewidth": 1.0, "color": "#475569"},
            showfliers=False,
        )
        rng_jitter = np.random.default_rng(42)
        for idx, (patch, (_, vals, color)) in enumerate(zip(bp["boxes"], spread_specs, strict=True), start=1):
            patch.set_facecolor(color)
            patch.set_alpha(0.28)
            jitter = rng_jitter.uniform(-0.14, 0.14, size=len(vals))
            ax_sp.scatter(vals, np.full(len(vals), idx) + jitter, color=color, s=28, alpha=0.9, zorder=4)

        ax_sp.axvline(0.0, color="#111827", linestyle="--", linewidth=1.2)
        ax_sp.set_yticks(range(1, len(spread_labels) + 1))
        ax_sp.set_yticklabels(spread_labels, fontsize=8.5)
        ax_sp.invert_yaxis()
        ax_sp.set_xlabel("Improvement (% points, > 0 favors Strategy)", fontweight="bold")
        ax_sp.set_title("Paired Alpha & Drawdown-Reduction Distributions", fontweight="bold")

        fig.tight_layout()
        fig.savefig(p3, dpi=200, bbox_inches="tight")
        saved_paths.append(p3)
    finally:
        plt.close(fig)

    # 4. Calendar Year Regime Comparison
    p4 = output_dir / "04_calendar_year_regime_comparison.png"
    years = sorted(report.cohorts[0].annual_returns.keys())
    year_labels = _format_year_labels(years, report.start_date, report.end_date)

    fig, ax = plt.subplots(figsize=(13, 6.2))
    try:
        fig.patch.set_facecolor(bg_color)
        _style_axis(ax)

        x = np.arange(len(years), dtype=float)
        width = 0.19
        series_cfg = [
            ("strategy", "Strategy", c_strat, -1.5 * width),
            ("ew_cohort", f"EW-{report.cohort_size} Daily", c_ew, -0.5 * width),
            ("ew_buy_hold", f"EW-{report.cohort_size} B&H", c_ew_bh, 0.5 * width),
            ("spy", "SPY", c_spy, 1.5 * width),
        ]
        rng_dot = np.random.default_rng(7)
        for key, lbl, color, offset in series_cfg:
            medians: list[float] = []
            err_lo: list[float] = []
            err_hi: list[float] = []
            for y_idx, yr in enumerate(years):
                yr_vals = np.array([c.annual_returns[yr][key] for c in report.cohorts], dtype=float)
                med = float(np.median(yr_vals))
                p25 = float(np.percentile(yr_vals, 25))
                p75 = float(np.percentile(yr_vals, 75))
                medians.append(med)
                err_lo.append(max(0.0, med - p25))
                err_hi.append(max(0.0, p75 - med))
                if key != "spy":
                    jx = x[y_idx] + offset + rng_dot.uniform(-width * 0.18, width * 0.18, size=len(yr_vals))
                    ax.scatter(jx, yr_vals, color=color, s=12, alpha=0.45, zorder=4)

            bars = ax.bar(
                x + offset,
                medians,
                width=width,
                color=color,
                alpha=0.85,
                label=lbl,
                zorder=3,
            )
            if key != "spy":
                ax.errorbar(
                    x + offset,
                    medians,
                    yerr=[err_lo, err_hi],
                    fmt="none",
                    ecolor="#1E293B",
                    elinewidth=1.2,
                    capsize=3,
                    zorder=5,
                )
            for bar_rect, med, lo_e, hi_e in zip(bars, medians, err_lo, err_hi, strict=True):
                va = "bottom" if med >= 0 else "top"
                y_txt = (med + (hi_e if key != "spy" else 0.0) + 0.8) if med >= 0 else (med - (lo_e if key != "spy" else 0.0) - 0.8)
                ax.text(
                    bar_rect.get_x() + bar_rect.get_width() / 2.0,
                    y_txt,
                    f"{med:+.1f}%",
                    ha="center",
                    va=va,
                    fontsize=7.5,
                    fontweight="bold",
                    color="#0F172A",
                )

        ax.margins(y=0.18)

        ax.axhline(0.0, color="#334155", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(year_labels, fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.set_ylabel("Calendar-Year Return (%)", fontsize=10, fontweight="bold")
        ax.set_title(
            f"Calendar-Year Regime Performance (Median & Cohort IQR across {report.n_cohorts} Cohorts)",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="upper left", frameon=True, facecolor="white", fontsize=9)
        fig.tight_layout()
        fig.savefig(p4, dpi=200, bbox_inches="tight")
        saved_paths.append(p4)
    finally:
        plt.close(fig)

    return saved_paths


def render_sp500_cohort_markdown(report: Sp500CohortStudyReport) -> str:
    """Render a clean institutional Markdown summary of the S&P 500 cohort study."""
    lines: list[str] = [
        f"# S&P 500 {report.n_cohorts}-Cohort Validation Report ({report.cohort_size} Stocks/Cohort)",
        "",
        "## Methodology & Point-in-Time Safeguards",
        "",
        f"- **Evaluation Window:** `{report.start_date}` to `{report.end_date}` ({report.trading_days} trading days; raw bars `{report.raw_bar_start}` to `{report.raw_bar_end}`)",
        f"- **Index-Addition Anti-Lookahead Cutoff:** `{report.cutoff_date}` (Wikipedia constituents: `{report.total_wikipedia_constituents}` total -> `{report.pre_cutoff_constituents}` added on/before cutoff -> `{report.deduped_issuer_constituents}` unique issuers -> `{report.active_eligible_pool_size}` continuously active eligible equities)",
        f"- **Sampling:** `{report.n_cohorts}` distinct uniform cohorts of `{report.cohort_size}` equities without replacement (`seed={report.seed}`, covering `{report.unique_symbols_across_cohorts}` unique symbols)",
        f"- **Execution & Signal Configuration:** `execution_timing={report.execution_timing}`, `allow_shorting=True` (matches `top_k_short={report.top_k_short}`), `max_drawdown_halt_pct={report.max_drawdown_halt_pct:.1f}%`, `use_news=False` (symmetric price+vol+PIT-macro across all S&P 500 equities), `n_trials={report.n_cohorts}` for Deflated Sharpe Ratio",
        f"- **Survivor-Conditioning Disclosure:** Filtering current S&P 500 constituents by `date_added <= {report.cutoff_date}` plus active-bar continuity eliminates post-cutoff index-addition lookahead (e.g., `SMCI`, `PLTR`), while post-cutoff index deletions/bankruptcies (e.g., `SIVB`, `FRC`) are absent from the continuous survivor pool. Consequently, `Strategy` vs `EW-{report.cohort_size} Daily` / `EW-{report.cohort_size} B&H` is a strictly matched-universe comparison (zero differential survivorship bias), whereas `SPY` reflects the unconditioned point-in-time index.",
        "",
        "## Cross-Cohort Distribution Summary",
        "",
        "| Series | Metric | Median | Mean | Std | IQR [P25, P75] | Min .. Max |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]

    series_rows = [
        ("Strategy", report.strategy_stats),
        (f"EW-{report.cohort_size} Daily", report.ew_cohort_stats),
        (f"EW-{report.cohort_size} Buy & Hold", report.ew_buy_hold_stats),
        ("SPY Buy & Hold", report.spy_stats),
    ]
    metric_specs = [
        ("CAGR (%)", "cagr_pct", "{:+.2f}%"),
        ("Total Return (%)", "total_return_pct", "{:+.2f}%"),
        ("Sharpe (rf=0)", "sharpe_ratio", "{:.3f}"),
        ("Excess Sharpe", "excess_sharpe_ratio", "{:.3f}"),
        ("Ann Vol (%)", "ann_volatility_pct", "{:.2f}%"),
        ("Max Drawdown (%)", "max_drawdown_pct", "{:.2f}%"),
        ("Calmar Ratio", "calmar_ratio", "{:.3f}"),
    ]
    for series_name, agg in series_rows:
        for metric_label, attr, fmt in metric_specs:
            d: DistributionStats = getattr(agg, attr)
            lines.append(
                f"| {series_name} | {metric_label} | {fmt.format(d.median)} | "
                f"{fmt.format(d.mean)} | {d.std:.3f} | "
                f"[{fmt.format(d.p25)}, {fmt.format(d.p75)}] | "
                f"{fmt.format(d.min)} .. {fmt.format(d.max)} |"
            )

    w = report.win_rates
    n = w.total_cohorts
    lines.extend(
        [
            "",
            "## Strategy Win Rates Across Cohorts",
            "",
            "| Comparison Metric | Strategy > SPY | Strategy > EW-Daily | Strategy > EW-B&H |",
            "| :--- | ---: | ---: | ---: |",
            f"| Total Return | {w.beat_spy_return}/{n} ({w.beat_spy_return / n * 100:.1f}%) | {w.beat_ew_return}/{n} ({w.beat_ew_return / n * 100:.1f}%) | {w.beat_ew_bh_return}/{n} ({w.beat_ew_bh_return / n * 100:.1f}%) |",
            f"| CAGR | {w.beat_spy_cagr}/{n} ({w.beat_spy_cagr / n * 100:.1f}%) | {w.beat_ew_cagr}/{n} ({w.beat_ew_cagr / n * 100:.1f}%) | {w.beat_ew_bh_cagr}/{n} ({w.beat_ew_bh_cagr / n * 100:.1f}%) |",
            f"| Sharpe (rf=0) | {w.beat_spy_sharpe}/{n} ({w.beat_spy_sharpe / n * 100:.1f}%) | {w.beat_ew_sharpe}/{n} ({w.beat_ew_sharpe / n * 100:.1f}%) | {w.beat_ew_bh_sharpe}/{n} ({w.beat_ew_bh_sharpe / n * 100:.1f}%) |",
            f"| Excess Sharpe | {w.beat_spy_excess_sharpe}/{n} ({w.beat_spy_excess_sharpe / n * 100:.1f}%) | {w.beat_ew_excess_sharpe}/{n} ({w.beat_ew_excess_sharpe / n * 100:.1f}%) | {w.beat_ew_bh_excess_sharpe}/{n} ({w.beat_ew_bh_excess_sharpe / n * 100:.1f}%) |",
            f"| Shallower Max Drawdown | {w.beat_spy_max_drawdown}/{n} ({w.beat_spy_max_drawdown / n * 100:.1f}%) | {w.beat_ew_max_drawdown}/{n} ({w.beat_ew_max_drawdown / n * 100:.1f}%) | {w.beat_ew_bh_max_drawdown}/{n} ({w.beat_ew_bh_max_drawdown / n * 100:.1f}%) |",
            "",
            "## Calendar-Year Sub-Period Returns (Cross-Cohort Medians)",
            "",
            f"| Year | Strategy Median | EW-{report.cohort_size} Daily Median | EW-{report.cohort_size} B&H Median | SPY |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )

    years = sorted(report.cohorts[0].annual_returns.keys())
    year_labels = _format_year_labels(years, report.start_date, report.end_date)
    for yr, yr_label in zip(years, year_labels, strict=True):
        s_med = float(np.median([c.annual_returns[yr]["strategy"] for c in report.cohorts]))
        e_med = float(np.median([c.annual_returns[yr]["ew_cohort"] for c in report.cohorts]))
        b_med = float(np.median([c.annual_returns[yr]["ew_buy_hold"] for c in report.cohorts]))
        spy_yr = report.cohorts[0].annual_returns[yr]["spy"]
        lines.append(
            f"| {yr_label} | {s_med:+.2f}% | {e_med:+.2f}% | {b_med:+.2f}% | {spy_yr:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Per-Cohort Results",
            "",
            f"| Cohort | Sample Tickers | Strat Ret | Strat CAGR | Strat Vol | Sharpe | Ex-Sharpe | DSR | Strat MaxDD | EW-{report.cohort_size} CAGR | EW-{report.cohort_size} MaxDD | SPY CAGR | SPY MaxDD |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for c in report.cohorts:
        m = c.metrics
        sample_syms = ", ".join(c.symbols[:5]) + ", ..."
        dsr_str = f"{m.deflated_sharpe_prob:.3f}" if m.deflated_sharpe_prob is not None else "N/A"
        lines.append(
            f"| {c.label} | `{sample_syms}` | {m.total_return_pct:+.2f}% | "
            f"{m.cagr_pct:+.2f}% | {m.ann_volatility_pct:.2f}% | {m.sharpe_ratio:.3f} | "
            f"{m.excess_sharpe_ratio:.3f} | {dsr_str} | {m.max_drawdown_pct:.2f}% | "
            f"{c.ew_cagr_pct:+.2f}% | {m.ew_universe_max_drawdown_pct:.2f}% | "
            f"{c.benchmark_cagr_pct:+.2f}% | {m.benchmark_max_drawdown_pct:.2f}% |"
        )

    if report.plot_paths:
        lines.extend(["", "## Charts", ""])
        for p in report.plot_paths:
            lines.append(f"- `{p}`")

    lines.append("")
    return "\n".join(lines)
