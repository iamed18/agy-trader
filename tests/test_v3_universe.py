from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd
import pytest

from agy_trader.backtest.v3_universe import (
    _DUAL_CLASS_DROP,
    HISTORICAL_DELISTED_SECTORS,
    SECTOR_TO_ETF,
    V3_CORE_ETFS,
    Sp400Constituent,
    V3EquityCandidate,
    atomic_write_text,
    build_dynamic_pit_universe_schedule,
    build_live_dynamic_universe,
    fetch_and_cache_fred_macro,
    fetch_and_cache_v3_sip_bars,
    is_in_sp500_on,
    load_or_fetch_v3_equity_pool,
    normalize_equity_symbol,
    parse_sp500_start_end_csv,
    parse_wikipedia_sp400_html,
)
from agy_trader.core.models import AssetType, MacroPoint, UniverseSymbol
from agy_trader.data.alpaca_market import AlpacaMarketDataProvider
from agy_trader.data.macro_and_calendar import FREDMacroProvider
from agy_trader.storage.parquet_store import ParquetBarStore

NY_TZ = ZoneInfo("America/New_York")


def _trading_dates(n_days: int, start: date = date(2022, 1, 3)) -> list[date]:
    """Generate n_days of weekday trading dates starting from start."""
    out: list[date] = []
    cur = start
    while len(out) < n_days:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def _make_bars_df(
    dates: list[date],
    closes: np.ndarray | list[float],
    *,
    volume: float = 2_000_000.0,
    daily_spread: float = 0.01,
) -> pd.DataFrame:
    """Create canonical daily OHLCV DataFrame for testing."""
    c = np.asarray(closes, dtype=float)
    o = c * (1.0 - 0.002)
    h = np.maximum(o, c) * (1.0 + daily_spread)
    lows = np.minimum(o, c) * (1.0 - daily_spread)
    timestamps = [
        datetime(d.year, d.month, d.day, 16, 0, tzinfo=NY_TZ).astimezone(UTC) for d in dates
    ]
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(timestamps, tz="UTC"),
            "open": o,
            "high": h,
            "low": lows,
            "close": c,
            "volume": np.full(len(dates), volume, dtype=float),
            "vwap": c,
        }
    )


# ---------------------------------------------------------------------------
# 1. Constants, Helpers, Normalization
# ---------------------------------------------------------------------------


def test_constants_and_mappings() -> None:
    assert len(V3_CORE_ETFS) == 16
    assert "SPY" in V3_CORE_ETFS
    assert "QQQ" in V3_CORE_ETFS
    assert "XLK" in V3_CORE_ETFS
    assert "XLRE" in V3_CORE_ETFS

    assert len(SECTOR_TO_ETF) == 11
    assert SECTOR_TO_ETF["Technology"] == "XLK"
    assert SECTOR_TO_ETF["Financials"] == "XLF"
    assert SECTOR_TO_ETF["Real Estate"] == "XLRE"

    # Historical delisted mappings
    assert HISTORICAL_DELISTED_SECTORS["SIVB"] == "Financials"
    assert HISTORICAL_DELISTED_SECTORS["FRC"] == "Financials"
    assert HISTORICAL_DELISTED_SECTORS["ATVI"] == "Communication Services"
    assert HISTORICAL_DELISTED_SECTORS["VMW"] == "Technology"
    assert HISTORICAL_DELISTED_SECTORS["PXD"] == "Energy"
    assert HISTORICAL_DELISTED_SECTORS["CELG"] == "Healthcare"
    assert HISTORICAL_DELISTED_SECTORS["KSU"] == "Industrials"
    assert HISTORICAL_DELISTED_SECTORS["TIF"] == "Consumer Discretionary"
    assert HISTORICAL_DELISTED_SECTORS["DRE"] == "Real Estate"

    # Dual class drop map
    assert _DUAL_CLASS_DROP["GOOG"] == "GOOGL"
    assert _DUAL_CLASS_DROP["BRK.A"] == "BRK.B"
    assert _DUAL_CLASS_DROP["FOX"] == "FOXA"
    assert _DUAL_CLASS_DROP["NWS"] == "NWSA"
    assert _DUAL_CLASS_DROP["BF.A"] == "BF.B"


def test_normalize_equity_symbol() -> None:
    assert normalize_equity_symbol("BRK-B") == "BRK.B"
    assert normalize_equity_symbol("BF-B") == "BF.B"
    assert normalize_equity_symbol("UHAL-B") == "UHAL.B"
    assert normalize_equity_symbol("AAPL[1]") == "AAPL"
    assert normalize_equity_symbol(" msft ") == "MSFT"
    assert normalize_equity_symbol("") == ""


# ---------------------------------------------------------------------------
# 2. V3EquityCandidate & is_in_sp500_on
# ---------------------------------------------------------------------------


def test_is_in_sp500_on() -> None:
    cand_active = V3EquityCandidate(
        symbol="AAPL",
        security="Apple Inc.",
        sector="Technology",
        sources=("sp500_curr",),
        sp500_intervals=(("1982-11-30", None),),
    )
    assert not is_in_sp500_on(cand_active, date(1982, 11, 29))
    assert is_in_sp500_on(cand_active, date(1982, 11, 30))
    assert is_in_sp500_on(cand_active, date(2024, 6, 1))

    cand_delisted = V3EquityCandidate(
        symbol="SIVB",
        security="SVB Financial Group",
        sector="Financials",
        sources=("sp500_hist",),
        sp500_intervals=(("2006-03-31", "2023-03-10"),),
    )
    assert not is_in_sp500_on(cand_delisted, date(2005, 1, 1))
    assert is_in_sp500_on(cand_delisted, date(2006, 3, 31))
    assert is_in_sp500_on(cand_delisted, date(2023, 3, 10))
    assert not is_in_sp500_on(cand_delisted, date(2023, 3, 11))

    cand_multi = V3EquityCandidate(
        symbol="MULTI",
        security="Multi Re-enter Corp",
        sector="Industrials",
        sources=("sp500_hist",),
        sp500_intervals=(("2010-01-01", "2015-12-31"), ("2020-01-01", None)),
    )
    assert is_in_sp500_on(cand_multi, date(2012, 6, 1))
    assert not is_in_sp500_on(cand_multi, date(2018, 1, 1))
    assert is_in_sp500_on(cand_multi, date(2022, 1, 1))

    cand_none = V3EquityCandidate(symbol="MID", sector="Technology")
    assert not is_in_sp500_on(cand_none, date(2023, 1, 1))


# ---------------------------------------------------------------------------
# 3. CSV Parsing (parse_sp500_start_end_csv)
# ---------------------------------------------------------------------------


def test_parse_sp500_start_end_csv() -> None:
    csv_text = """ticker,start_date,end_date
AAPL,1982-11-30,
SIVB,2006-03-31,2023-03-10
OLD_DELISTED,1990-01-01,2010-05-01
FUTURE_IPO,2027-01-01,
BRK-B,2010-02-16,
MULTI,2010-01-01,2015-01-01
MULTI,2020-01-01,2024-01-01
"""
    intervals = parse_sp500_start_end_csv(
        csv_text,
        min_overlap_date=date(2016, 1, 1),
        max_overlap_date=date(2026, 11, 20),
    )
    assert "AAPL" in intervals
    assert intervals["AAPL"] == [("1982-11-30", None)]

    assert "SIVB" in intervals
    assert intervals["SIVB"] == [("2006-03-31", "2023-03-10")]

    # Delisted in 2010: does not overlap with [2016-01-01, 2026-11-20]
    assert "OLD_DELISTED" not in intervals

    # Starts in 2027: does not overlap
    assert "FUTURE_IPO" not in intervals

    # Hyphen normalization
    assert "BRK.B" in intervals

    # Multi intervals: first interval 2010-2015 dropped, second kept
    assert "MULTI" in intervals
    assert intervals["MULTI"] == [("2020-01-01", "2024-01-01")]


# ---------------------------------------------------------------------------
# 4. Wikipedia S&P 400 Parsing (parse_wikipedia_sp400_html)
# ---------------------------------------------------------------------------


def test_parse_wikipedia_sp400_html() -> None:
    html = """
    <html><body>
      <table id="constituents">
        <tr>
          <th>Ticker symbol</th><th>Security</th><th>GICS Sector</th><th>HQ</th>
        </tr>
        <tr>
          <td>AA</td><td>Alcoa Corp</td><td>Materials</td><td>Pittsburgh, PA</td>
        </tr>
        <tr>
          <td>SAIC[1]</td><td>Science Applications</td><td>Information Technology</td><td>Reston, VA</td>
        </tr>
        <tr>
          <td>BRK-B</td><td>Berkshire Mid</td><td>Financials</td><td>Omaha, NE</td>
        </tr>
      </table>
    </body></html>
    """
    parsed = parse_wikipedia_sp400_html(html)
    assert len(parsed) == 3

    # NamedTuple supports tuple unpacking and attribute access
    sym, sec, sector = parsed[0]
    assert sym == "AA"
    assert sec == "Alcoa Corp"
    assert sector == "Materials"

    assert parsed[1].symbol == "SAIC"
    assert parsed[1].sector == "Technology"  # Mapped via map_gics_sector

    assert parsed[2].symbol == "BRK.B"  # Normalized hyphen


def test_parse_wikipedia_sp400_html_wikitable_fallback() -> None:
    html = """
    <html><body>
      <table class="wikitable">
        <tr>
          <th>Symbol</th><th>Company</th><th>Sector</th>
        </tr>
        <tr>
          <td>TEX</td><td>Terex Corp</td><td>Industrials</td>
        </tr>
      </table>
    </body></html>
    """
    parsed = parse_wikipedia_sp400_html(html)
    assert len(parsed) == 1
    assert parsed[0] == Sp400Constituent(symbol="TEX", security="Terex Corp", sector="Industrials")


def test_parse_wikipedia_sp400_html_missing_table_raises() -> None:
    with pytest.raises(ValueError, match="Could not find S&P 400 constituents table"):
        parse_wikipedia_sp400_html("<html><body><p>No table</p></body></html>")


# ---------------------------------------------------------------------------
# 5. V3 Equity Pool Discovery & Cache (load_or_fetch_v3_equity_pool)
# ---------------------------------------------------------------------------


def test_load_or_fetch_v3_equity_pool_cache_hit(tmp_path: Path) -> None:
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    cache_path = research_dir / "v3_equity_pool.json"
    dummy_pool = [
        {
            "symbol": "AAPL",
            "security": "Apple Inc.",
            "sector": "Technology",
            "sources": ["sp500_curr"],
            "sp500_intervals": [["1982-11-30", None]],
        }
    ]
    cache_path.write_text(json.dumps(dummy_pool), encoding="utf-8")

    pool = load_or_fetch_v3_equity_pool(research_dir, refresh=False)
    assert len(pool) == 1
    assert pool[0].symbol == "AAPL"
    assert pool[0].sector == "Technology"


def test_load_or_fetch_v3_equity_pool_fetch_and_merge(tmp_path: Path) -> None:
    research_dir = tmp_path / "research"
    research_dir.mkdir()

    csv_data = """ticker,start_date,end_date
GOOG,2006-04-03,
SIVB,2006-03-31,2023-03-10
UNK_TICKER,2018-01-01,2022-01-01
"""
    sp500_html = """
    <html><body>
      <table id="constituents">
        <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>Date added</th><th>CIK</th></tr>
        <tr><td>GOOGL</td><td>Alphabet Class A</td><td>Communication Services</td><td>2014-04-03</td><td>0001652044</td></tr>
        <tr><td>MSFT</td><td>Microsoft Corp</td><td>Information Technology</td><td>1994-06-01</td><td>0000789019</td></tr>
      </table>
    </body></html>
    """
    sp400_html = """
    <html><body>
      <table id="constituents">
        <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
        <tr><td>AA</td><td>Alcoa Corp</td><td>Materials</td></tr>
      </table>
    </body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "sp500_ticker_start_end.csv" in url:
            return httpx.Response(200, text=csv_data)
        if "List_of_S%26P_500_companies" in url:
            return httpx.Response(200, text=sp500_html)
        if "List_of_S%26P_400_companies" in url:
            return httpx.Response(200, text=sp400_html)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        pool = load_or_fetch_v3_equity_pool(research_dir, refresh=True, client=client)

    symbols = [c.symbol for c in pool]
    by_sym = {c.symbol: c for c in pool}

    # Verify sorting
    assert symbols == sorted(symbols)

    # 1. Dual class drop: GOOG dropped, GOOGL kept with merged intervals and sources
    assert "GOOG" not in by_sym
    assert "GOOGL" in by_sym
    assert "sp500_hist" in by_sym["GOOGL"].sources
    assert "sp500_curr" in by_sym["GOOGL"].sources
    assert by_sym["GOOGL"].sector == "Communication Services"

    # 2. Historical delisted: SIVB gets sector from HISTORICAL_DELISTED_SECTORS
    assert "SIVB" in by_sym
    assert by_sym["SIVB"].sector == "Financials"
    assert "sp500_hist" in by_sym["SIVB"].sources

    # 3. Unknown delisted ticker falls back to Unknown
    assert "UNK_TICKER" in by_sym
    assert by_sym["UNK_TICKER"].sector == "Unknown"

    # 4. S&P 400 constituent
    assert "AA" in by_sym
    assert by_sym["AA"].sector == "Materials"
    assert by_sym["AA"].sources == ("sp400_curr",)

    # 5. Parquet/json file saved on disk
    cache_path = research_dir / "v3_equity_pool.json"
    assert cache_path.exists()


def test_validate_research_dir_safety() -> None:
    # Reject protected data/parquet writes
    with pytest.raises(ValueError, match="must not target or live inside"):
        load_or_fetch_v3_equity_pool(Path("data/parquet"), refresh=False)


# ---------------------------------------------------------------------------
# 6. Alpaca SIP Bar Fetcher (fetch_and_cache_v3_sip_bars)
# ---------------------------------------------------------------------------


def test_fetch_and_cache_v3_sip_bars_manifest_and_cache_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    store = ParquetBarStore(research_dir)

    # Pre-populate AAPL bars
    dates = _trading_dates(10)
    aapl_df = _make_bars_df(dates, [150.0] * 10)
    store.save_bars("AAPL", aapl_df, timeframe="1d")

    manifest = {
        "AAPL": {
            "status": "ok",
            "bars": 10,
            "first": str(aapl_df["timestamp"].min()),
            "last": str(aapl_df["timestamp"].max()),
            "fetched_at": "2026-11-20T00:00:00Z",
        },
        "DELISTED_EMPTY": {
            "status": "empty",
            "bars": 0,
            "first": None,
            "last": None,
            "fetched_at": "2026-11-20T00:00:00Z",
        },
    }
    atomic_write_text(research_dir / "sip_fetch_manifest.json", json.dumps(manifest))

    # Should not call provider because AAPL has cache + ok, and DELISTED_EMPTY is empty
    class _FailingProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Provider should not be called on cache hit!")

    monkeypatch.setattr(
        "agy_trader.backtest.v3_universe.AlpacaMarketDataProvider",
        _FailingProvider,
    )

    bars = fetch_and_cache_v3_sip_bars(
        ["AAPL", "DELISTED_EMPTY"],
        research_dir,
        api_key="fake",
        secret_key="fake",
    )
    assert "AAPL" in bars
    assert len(bars["AAPL"]) == 10
    assert "DELISTED_EMPTY" not in bars


def test_fetch_and_cache_v3_sip_bars_rate_limit_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    research_dir = tmp_path / "research"
    research_dir.mkdir()

    dates = _trading_dates(5)
    msft_df = _make_bars_df(dates, [250.0] * 5)

    attempts = {"batch": 0}
    slept: list[float] = []

    class _MockClient:
        def __init__(self) -> None:
            self._session = self
            self.closed = False

        def close(self) -> None:
            self.closed = True

    mock_client = _MockClient()

    class _MockProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._client = mock_client

        def fetch_bars(
            self, symbols: list[str], start: datetime, end: datetime, timeframe: str = "1d"
        ) -> dict[str, pd.DataFrame]:
            if len(symbols) > 1:
                attempts["batch"] += 1
                if attempts["batch"] == 1:
                    raise RuntimeError("429 rate limit exceeded")
                # Non-rate failure to test batch fallback to single-symbol
                raise RuntimeError("Batch internal error")
            sym = symbols[0]
            if sym == "MSFT":
                return {"MSFT": msft_df}
            return {sym: pd.DataFrame()}

    monkeypatch.setattr(
        "agy_trader.backtest.v3_universe.AlpacaMarketDataProvider",
        _MockProvider,
    )

    bars = fetch_and_cache_v3_sip_bars(
        ["MSFT", "INVALID_TICKER"],
        research_dir,
        api_key="fake",
        secret_key="fake",
        batch_size=25,
        sleep_fn=slept.append,
    )

    assert "MSFT" in bars
    assert len(bars["MSFT"]) == 5
    assert len(slept) == 1  # 429 backoff was called
    assert mock_client.closed is True  # Session cleanly closed

    # Check manifest updated
    manifest_path = research_dir / "sip_fetch_manifest.json"
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["MSFT"]["status"] == "ok"
    assert manifest_data["INVALID_TICKER"]["status"] == "empty"


# ---------------------------------------------------------------------------
# 7. FRED Macro Fetcher & Cache (fetch_and_cache_fred_macro)
# ---------------------------------------------------------------------------


def test_fetch_and_cache_fred_macro(tmp_path: Path) -> None:
    research_dir = tmp_path / "research"
    research_dir.mkdir()

    class _MockFREDProvider(FREDMacroProvider):
        def __init__(self) -> None:
            pass

        def fetch_macro(self, start_date: date, end_date: date) -> list[MacroPoint]:
            return [
                MacroPoint(as_of_date=date(2020, 1, 1), series_id="T10Y2Y", value=0.5),
                MacroPoint(as_of_date=date(2020, 1, 1), series_id="VIXCLS", value=14.0),
            ]

    points = fetch_and_cache_fred_macro(
        research_dir,
        fred_api_key="dummy",
        provider=_MockFREDProvider(),
    )
    assert len(points) == 2
    cache_path = research_dir / "fred_macro_2015_2026.json"
    assert cache_path.exists()

    # Cache hit offline with dummy provider that raises
    class _FailingFREDProvider(FREDMacroProvider):
        def __init__(self) -> None:
            pass

        def fetch_macro(self, start_date: date, end_date: date) -> list[MacroPoint]:
            raise RuntimeError("Should not be called")

    cached_points = fetch_and_cache_fred_macro(
        research_dir,
        fred_api_key="dummy",
        refresh=False,
        provider=_FailingFREDProvider(),
    )
    assert len(cached_points) == 2
    assert cached_points[0].series_id == "T10Y2Y"


# ---------------------------------------------------------------------------
# 8. Point-in-Time Dynamic Universe Schedule (build_dynamic_pit_universe_schedule)
# ---------------------------------------------------------------------------


def test_build_dynamic_pit_universe_schedule_full() -> None:
    # Generate 300 trading days (~14 months) so min_history_bars=252 is satisfied
    dates = _trading_dates(300, start=date(2022, 1, 3))
    spy_df = _make_bars_df(dates, [400.0] * 300)

    # Candidate 1: High MDDV, active throughout
    sym1_df = _make_bars_df(dates, [100.0] * 300, volume=5_000_000.0)

    # Candidate 2: Moderate MDDV, active throughout
    sym2_df = _make_bars_df(dates, [50.0] * 300, volume=2_000_000.0)

    # Candidate 3: Delisted halfway (stops trading at bar 260)
    sym3_df = _make_bars_df(dates[:260], [80.0] * 260, volume=3_000_000.0)

    # Candidate 4: Penny stock (price < 10)
    sym4_df = _make_bars_df(dates, [5.0] * 300, volume=10_000_000.0)

    bars_by_symbol = {
        "SPY": spy_df,
        "SYM1": sym1_df,
        "SYM2": sym2_df,
        "SYM3": sym3_df,
        "SYM4": sym4_df,
    }

    pool = [
        V3EquityCandidate(symbol="SYM1", sector="Technology"),
        V3EquityCandidate(symbol="SYM2", sector="Financials"),
        V3EquityCandidate(symbol="SYM3", sector="Energy"),
        V3EquityCandidate(symbol="SYM4", sector="Healthcare"),
    ]

    schedule = build_dynamic_pit_universe_schedule(
        bars_by_symbol,
        pool,
        benchmark_symbol="SPY",
        top_n=2,
        hysteresis_buffer=1,
        min_price=10.0,
        min_history_bars=250,
        mddv_window=60,
        max_stale_calendar_days=5,
        mode="broad_mddv",
    )

    assert len(schedule) > 0
    # Check that penny stock SYM4 is never selected
    for members in schedule.values():
        assert "SYM4" not in members
        assert "SPY" not in members
        assert "SYM1" in members  # Highest MDDV

    # Reconstitution dates are first trading days of calendar months
    r_dates = sorted(schedule.keys())
    assert len(r_dates) >= 2

    # In later months after bar 260, SYM3 should drop out due to stale delta (> 5 days)
    last_r_date = r_dates[-1]
    assert "SYM3" not in schedule[last_r_date]


def test_build_dynamic_pit_universe_schedule_turnover_hysteresis() -> None:
    # 2 reconstitution dates
    dates = _trading_dates(65, start=date(2023, 1, 3))
    # Month 1 is Jan 3 (bar 0), Month 2 is Feb 1 (bar 22), Month 3 is Mar 1 (bar 42)
    # With min_history_bars=10, Month 2 (Feb) and Month 3 (Mar) will be evaluated
    spy_df = _make_bars_df(dates, [400.0] * 65)

    # Let SYM_A be #1 in month 1, and drop to #3 in month 2
    # Let SYM_B be #2 in month 1, and become #1 in month 2
    # Let SYM_C be #3 in month 1, and become #2 in month 2
    # With top_n=2 and hysteresis_buffer=1 (cutoff rank < 3):
    # In month 2, SYM_A has rank 2 (< 3), so hysteresis keeps SYM_A! SYM_C does not displace SYM_A!

    # A: high in Feb, slips in Mar
    closes = np.full(65, 50.0)
    vols_a = np.array([30_000.0] * 35 + [10_000.0] * 30)
    df_a = _make_bars_df(dates, closes)
    df_a["volume"] = vols_a

    # B: constant high
    vols_b = np.array([25_000.0] * 35 + [25_000.0] * 30)
    df_b = _make_bars_df(dates, closes)
    df_b["volume"] = vols_b

    # C: low in Feb, surges in Mar
    vols_c = np.array([5_000.0] * 35 + [20_000.0] * 30)
    df_c = _make_bars_df(dates, closes)
    df_c["volume"] = vols_c

    bars = {"SPY": spy_df, "SYM_A": df_a, "SYM_B": df_b, "SYM_C": df_c}
    pool = [
        V3EquityCandidate(symbol="SYM_A"),
        V3EquityCandidate(symbol="SYM_B"),
        V3EquityCandidate(symbol="SYM_C"),
    ]

    schedule = build_dynamic_pit_universe_schedule(
        bars,
        pool,
        benchmark_symbol="SPY",
        top_n=2,
        hysteresis_buffer=1,  # rank < 3 retains existing members
        min_price=10.0,
        min_history_bars=10,
        mddv_window=10,
        mode="broad_mddv",
    )

    r_dates = sorted(schedule.keys())
    assert len(r_dates) >= 2
    feb_members = schedule[r_dates[0]]
    mar_members = schedule[r_dates[1]]

    assert feb_members == frozenset({"SYM_A", "SYM_B"})
    # Even though C has higher MDDV in Mar than A, A is kept by hysteresis (rank 2 < 2 + 1)
    assert mar_members == frozenset({"SYM_A", "SYM_B"})


def test_build_dynamic_pit_universe_schedule_pit_sp500_mode() -> None:
    dates = _trading_dates(30, start=date(2023, 1, 3))
    spy_df = _make_bars_df(dates, [400.0] * 30)
    sym1_df = _make_bars_df(dates, [100.0] * 30, volume=1_000_000.0)
    sym2_df = _make_bars_df(dates, [100.0] * 30, volume=2_000_000.0)

    bars = {"SPY": spy_df, "IN_SP500": sym1_df, "OUT_SP500": sym2_df}
    pool = [
        V3EquityCandidate(
            symbol="IN_SP500",
            sp500_intervals=(("2020-01-01", None),),
        ),
        V3EquityCandidate(
            symbol="OUT_SP500",
            sp500_intervals=(),  # Not in S&P 500
        ),
    ]

    sched_pit = build_dynamic_pit_universe_schedule(
        bars,
        pool,
        benchmark_symbol="SPY",
        top_n=2,
        min_history_bars=5,
        mddv_window=5,
        mode="pit_sp500_mddv",
    )
    first_date = min(sched_pit.keys())
    assert "IN_SP500" in sched_pit[first_date]
    assert "OUT_SP500" not in sched_pit[first_date]

    sched_broad = build_dynamic_pit_universe_schedule(
        bars,
        pool,
        benchmark_symbol="SPY",
        top_n=2,
        min_history_bars=5,
        mddv_window=5,
        mode="broad_mddv",
    )
    assert "OUT_SP500" in sched_broad[first_date]


# ---------------------------------------------------------------------------
# 9. Live Dynamic Universe Construction (build_live_dynamic_universe)
# ---------------------------------------------------------------------------


def test_build_live_dynamic_universe() -> None:
    class _MockAsset:
        def __init__(
            self,
            symbol: str,
            exchange: str = "NASDAQ",
            status: str = "active",
            tradable: bool = True,
            name: str = "Common Stock",
            attributes: list[str] | None = None,
        ) -> None:
            self.symbol = symbol
            self.exchange = exchange
            self.status = status
            self.tradable = tradable
            self.name = name
            self.attributes = attributes or []

    mock_assets = [
        _MockAsset("NVDA", "NASDAQ", "active"),
        _MockAsset("BRK.B", "NYSE", "active"),
        _MockAsset("SPY", "NYSE", "active"),  # Excluded (core ETF)
        _MockAsset("INACTIVE", "NASDAQ", "inactive"),  # Excluded
        _MockAsset("OTCBB", "OTC", "active"),  # Excluded (exchange)
        _MockAsset("TESTW", "NASDAQ", "active", name="Test Warrant"),  # Excluded warrant
        _MockAsset("TOO_LONG", "NYSE", "active"),  # Excluded length > 5
        _MockAsset("LONG.A", "NYSE", "active"),  # Excluded non-canonical dot
    ]

    class _MockTradingClient:
        def get_all_assets(self, *args: Any, **kwargs: Any) -> list[_MockAsset]:
            return mock_assets

    dates = _trading_dates(70)
    nvda_df = _make_bars_df(dates, [120.0] * 70, volume=10_000_000.0)
    brkb_df = _make_bars_df(dates, [350.0] * 70, volume=2_000_000.0)

    class _MockDataProvider(AlpacaMarketDataProvider):
        def __init__(self) -> None:
            pass

        def fetch_bars(
            self, symbols: list[str], start: datetime, end: datetime, timeframe: str = "1d"
        ) -> dict[str, pd.DataFrame]:
            return {"NVDA": nvda_df, "BRK.B": brkb_df}

    tc = _MockTradingClient()
    dp = _MockDataProvider()

    as_of = datetime(dates[-1].year, dates[-1].month, dates[-1].day, 20, 0, tzinfo=UTC)
    universe = build_live_dynamic_universe(
        tc,
        dp,
        top_n=10,
        min_price=10.0,
        as_of=as_of,
        sector_lookup={"NVDA": "Technology", "BRK.B": "Financials"},
    )

    assert len(universe) == 2
    assert universe[0] == UniverseSymbol(
        symbol="NVDA",
        asset_type=AssetType.equity,
        sector="Technology",
        shortable_default=True,
    )
    assert universe[1] == UniverseSymbol(
        symbol="BRK.B",
        asset_type=AssetType.equity,
        sector="Financials",
        shortable_default=True,
    )


# ---------------------------------------------------------------------------
# 10. Torvalds Audit Findings Regression Tests
# ---------------------------------------------------------------------------


def test_build_live_dynamic_universe_regression_filters_freshness_dedup() -> None:
    class _MockAsset:
        def __init__(
            self,
            symbol: str,
            name: str,
            exchange: str = "NASDAQ",
            status: str = "active",
            tradable: bool = True,
        ) -> None:
            self.symbol = symbol
            self.name = name
            self.exchange = exchange
            self.status = status
            self.tradable = tradable
            self.attributes: list[str] = []

    assets = [
        # Legitimate common stocks that had false-positive substring collisions
        _MockAsset("NFLX", "Netflix, Inc."),
        _MockAsset("UNH", "UnitedHealth Group Incorporated"),
        _MockAsset("UPS", "United Parcel Service, Inc."),
        _MockAsset("CW", "Curtiss-Wright Corporation"),
        # Dual-class pair: GOOG dropped, GOOGL kept
        _MockAsset("GOOG", "Alphabet Inc. Class C"),
        _MockAsset("GOOGL", "Alphabet Inc. Class A"),
        # True non-common instruments to exclude via regex
        _MockAsset("ACMW", "Acme Corp Warrants"),
        _MockAsset("ACMU", "Acme Acquisition Unit"),
        _MockAsset("MGAR", "Mega Corp Rights"),
        _MockAsset("PRFS", "Bank of America Preferred Stock"),
        _MockAsset("TETF", "Technology Sector ETF"),
        # Stale ticker (> 5 calendar days old)
        _MockAsset("HALT", "Halted Tech Corp"),
    ]

    dates = _trading_dates(70, start=date(2023, 1, 3))
    as_of = datetime(dates[-1].year, dates[-1].month, dates[-1].day, 20, 0, tzinfo=UTC)

    fresh_df = _make_bars_df(dates, [100.0] * 70, volume=5_000_000.0)
    stale_dates = dates[:55]
    stale_df = _make_bars_df(stale_dates, [50.0] * len(stale_dates), volume=1_000_000.0)

    bars_map = {
        "NFLX": fresh_df,
        "UNH": fresh_df,
        "UPS": fresh_df,
        "CW": fresh_df,
        "GOOG": fresh_df,
        "GOOGL": fresh_df,
        "HALT": stale_df,
    }

    class _MockTC:
        def get_all_assets(self, *args: Any, **kwargs: Any) -> list[_MockAsset]:
            return assets

    class _MockDP(AlpacaMarketDataProvider):
        def __init__(self) -> None:
            pass

        def fetch_bars(
            self, symbols: list[str], start: datetime, end: datetime, timeframe: str = "1d"
        ) -> dict[str, pd.DataFrame]:
            return {s: bars_map[s] for s in symbols if s in bars_map}

    universe = build_live_dynamic_universe(
        _MockTC(),
        _MockDP(),
        top_n=20,
        min_price=10.0,
        as_of=as_of,
    )
    result_syms = {u.symbol for u in universe}

    # Verify retained legitimate equities
    assert "NFLX" in result_syms
    assert "UNH" in result_syms
    assert "UPS" in result_syms
    assert "CW" in result_syms
    assert "GOOGL" in result_syms

    # Verify dual-class dropped
    assert "GOOG" not in result_syms

    # Verify warrants/units/rights/preferred/etf dropped
    assert "ACMW" not in result_syms
    assert "ACMU" not in result_syms
    assert "MGAR" not in result_syms
    assert "PRFS" not in result_syms
    assert "TETF" not in result_syms

    # Verify stale ticker dropped
    assert "HALT" not in result_syms


def test_fetch_and_cache_v3_sip_bars_error_status_poisoning_and_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    manifest_path = research_dir / "sip_fetch_manifest.json"

    dates = _trading_dates(10)
    good_df = _make_bars_df(dates, [100.0] * 10)

    call_counts: dict[str, int] = {"batch": 0, "OK_SYM": 0, "FAIL_SYM": 0, "EMPTY_SYM": 0}

    class _MockProvider1:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def fetch_bars(
            self, symbols: list[str], start: datetime, end: datetime, timeframe: str = "1d"
        ) -> dict[str, pd.DataFrame]:
            if len(symbols) > 1:
                call_counts["batch"] += 1
                raise RuntimeError("Batch internal error forcing fallback")
            sym = symbols[0]
            call_counts[sym] += 1
            if sym == "FAIL_SYM":
                raise RuntimeError("HTTP 500 Alpaca Internal Server Error")
            if sym == "EMPTY_SYM":
                return {sym: pd.DataFrame()}
            if sym == "OK_SYM":
                return {sym: good_df}
            return {}

    monkeypatch.setattr(
        "agy_trader.backtest.v3_universe.AlpacaMarketDataProvider",
        _MockProvider1,
    )

    bars = fetch_and_cache_v3_sip_bars(
        ["OK_SYM", "FAIL_SYM", "EMPTY_SYM"],
        research_dir,
        api_key="fake",
        secret_key="fake",
    )

    assert "OK_SYM" in bars
    assert "FAIL_SYM" not in bars
    assert "EMPTY_SYM" not in bars

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["OK_SYM"]["status"] == "ok"
    assert manifest["EMPTY_SYM"]["status"] == "empty"
    assert manifest["FAIL_SYM"]["status"] == "error"
    assert "HTTP 500 Alpaca Internal Server Error" in manifest["FAIL_SYM"]["error"]

    class _MockProvider2:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def fetch_bars(
            self, symbols: list[str], start: datetime, end: datetime, timeframe: str = "1d"
        ) -> dict[str, pd.DataFrame]:
            assert symbols == ["FAIL_SYM"], f"Expected only FAIL_SYM retried, got {symbols}"
            return {"FAIL_SYM": good_df}

    monkeypatch.setattr(
        "agy_trader.backtest.v3_universe.AlpacaMarketDataProvider",
        _MockProvider2,
    )

    bars2 = fetch_and_cache_v3_sip_bars(
        ["OK_SYM", "FAIL_SYM", "EMPTY_SYM"],
        research_dir,
        api_key="fake",
        secret_key="fake",
        refresh=False,
    )
    assert "FAIL_SYM" in bars2
    assert "OK_SYM" in bars2

    manifest2 = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest2["FAIL_SYM"]["status"] == "ok"


def test_fetch_and_cache_v3_sip_bars_manifest_dirty_flush_on_cache_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    manifest_path = research_dir / "sip_fetch_manifest.json"

    store = ParquetBarStore(research_dir)
    dates = _trading_dates(10)
    df = _make_bars_df(dates, [100.0] * 10)
    store.save_bars("PRE_CACHED", df, timeframe="1d")

    assert not manifest_path.exists()

    class _FailingProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Provider should not be called!")

    monkeypatch.setattr(
        "agy_trader.backtest.v3_universe.AlpacaMarketDataProvider",
        _FailingProvider,
    )

    bars = fetch_and_cache_v3_sip_bars(
        ["PRE_CACHED"],
        research_dir,
        api_key="fake",
        secret_key="fake",
        refresh=False,
    )
    assert "PRE_CACHED" in bars

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["PRE_CACHED"]["status"] == "ok"
    assert manifest["PRE_CACHED"]["bars"] == 10


def test_parse_sp500_start_end_csv_invalid_and_inverted_end_dates() -> None:
    csv_text = """ticker,start_date,end_date
BAD_SLASH,2020-01-01,2023/03/10
BAD_TEXT,2020-01-01,bad
INVERTED,2023-01-01,2020-01-01
VALID_ACTIVE,2020-01-01,
VALID_EXPLICIT_ACTIVE,2020-01-01,active
VALID_CLOSED,2020-01-01,2022-01-01
"""
    intervals = parse_sp500_start_end_csv(
        csv_text,
        min_overlap_date=date(2016, 1, 1),
        max_overlap_date=date(2026, 11, 20),
    )
    assert "BAD_SLASH" not in intervals
    assert "BAD_TEXT" not in intervals
    assert "INVERTED" not in intervals

    assert "VALID_ACTIVE" in intervals
    assert intervals["VALID_ACTIVE"] == [("2020-01-01", None)]
    assert "VALID_EXPLICIT_ACTIVE" in intervals
    assert intervals["VALID_EXPLICIT_ACTIVE"] == [("2020-01-01", None)]
    assert "VALID_CLOSED" in intervals
    assert intervals["VALID_CLOSED"] == [("2020-01-01", "2022-01-01")]


def test_build_dynamic_pit_universe_schedule_mid_month_start_date() -> None:
    dates = _trading_dates(350, start=date(2019, 1, 2))
    spy_df = _make_bars_df(dates, [300.0] * 350)
    stock_df = _make_bars_df(dates, [50.0] * 350, volume=2_000_000.0)

    bars = {"SPY": spy_df, "STOCK_A": stock_df}
    pool = [V3EquityCandidate(symbol="STOCK_A")]

    mid_month_start = date(2020, 1, 15)
    schedule = build_dynamic_pit_universe_schedule(
        bars,
        pool,
        benchmark_symbol="SPY",
        top_n=5,
        min_history_bars=250,
        mddv_window=60,
        start_date=mid_month_start,
    )

    assert len(schedule) > 0
    earliest_date = min(schedule.keys())
    assert earliest_date <= mid_month_start
    for d in sorted(schedule.keys())[1:]:
        assert d > mid_month_start


def test_build_dynamic_pit_universe_schedule_duplicate_pool_and_nat_timestamps() -> None:
    dates = _trading_dates(80, start=date(2023, 1, 3))
    spy_df = _make_bars_df(dates, [400.0] * 80)
    spy_df.iloc[5, spy_df.columns.get_loc("timestamp")] = pd.NaT
    spy_df = spy_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    stock_df = _make_bars_df(dates, [100.0] * 80, volume=1_000_000.0)
    stock_df.iloc[10, stock_df.columns.get_loc("timestamp")] = pd.NaT
    stock_df = stock_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    bars = {"SPY": spy_df, "STOCK_1": stock_df}
    pool = [
        V3EquityCandidate(symbol="STOCK_1"),
        V3EquityCandidate(symbol="STOCK_1"),
        V3EquityCandidate(symbol="stock-1"),
    ]

    schedule = build_dynamic_pit_universe_schedule(
        bars,
        pool,
        benchmark_symbol="SPY",
        top_n=5,
        min_history_bars=20,
        mddv_window=20,
    )

    assert len(schedule) > 0
    for members in schedule.values():
        assert "STOCK_1" in members
        assert len(members) == 1
