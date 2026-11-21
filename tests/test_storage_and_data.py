from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd
import pytest
import respx

from agy_trader.core.config import LLMQuotaConfig
from agy_trader.core.models import (
    AssetType,
    EarningsEvent,
    LLMUsageRecord,
    MacroPoint,
    NewsArticle,
    UniverseSymbol,
    compute_headline_hash,
)
from agy_trader.data.alpaca_news import AlpacaNewsProvider
from agy_trader.data.macro_and_calendar import FREDMacroProvider
from agy_trader.data.synthetic import generate_synthetic_dataset
from agy_trader.storage.db import StateDB
from agy_trader.storage.parquet_store import CorporateActionMismatchError, ParquetBarStore

NY_TZ = ZoneInfo("America/New_York")


def _make_bars_df(
    timestamps: list[datetime] | pd.DatetimeIndex,
    closes: list[float],
    *,
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
    vwaps: list[float] | None = None,
) -> pd.DataFrame:
    n = len(closes)
    op = opens or closes
    hi = highs or [max(o, c) * 1.01 for o, c in zip(op, closes, strict=True)]
    lo = lows or [min(o, c) * 0.99 for o, c in zip(op, closes, strict=True)]
    vol = volumes or [1_000_000.0] * n
    data: dict[str, object] = {
        "timestamp": timestamps,
        "open": op,
        "high": hi,
        "low": lo,
        "close": closes,
        "volume": vol,
    }
    if vwaps is not None:
        data["vwap"] = vwaps
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# ParquetBarStore tests
# ---------------------------------------------------------------------------


def test_parquet_store_utc_enforcement_rejects_naive_and_nat(tmp_path) -> None:
    store = ParquetBarStore(tmp_path)

    # Naive datetime64 column
    naive_dt64 = pd.date_range("2025-01-02 21:00:00", periods=2, freq="1D")
    with pytest.raises(ValueError, match="timezone-aware"):
        store.save_bars("AAPL", _make_bars_df(naive_dt64, [100.0, 101.0]))

    # Naive Python datetime objects
    naive_py = [datetime(2025, 1, 2, 16, 0), datetime(2025, 1, 3, 16, 0)]
    with pytest.raises(ValueError, match="timezone-aware"):
        store.save_bars("AAPL", _make_bars_df(naive_py, [100.0, 101.0]))

    # NaT inside a DatetimeTZDtype(tz='UTC')
    tz_with_nat = pd.DatetimeIndex(
        [datetime(2025, 1, 2, 21, 0, tzinfo=UTC), pd.NaT],
        tz="UTC",
    )
    with pytest.raises(ValueError, match="NaN/NaT"):
        store.save_bars("AAPL", _make_bars_df(tz_with_nat, [100.0, 101.0]))

    # Non-UTC tz-aware timestamps should be normalized cleanly to UTC
    et_ts = [
        datetime(2025, 1, 2, 16, 0, tzinfo=NY_TZ),
        datetime(2025, 1, 3, 16, 0, tzinfo=NY_TZ),
    ]
    count = store.save_bars("AAPL", _make_bars_df(et_ts, [100.0, 101.0]))
    assert count == 2
    loaded = store.load_bars("AAPL")
    assert isinstance(loaded["timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(loaded["timestamp"].dt.tz) == "UTC"
    assert loaded["timestamp"].iloc[0].to_pydatetime() == datetime(2025, 1, 2, 21, 0, tzinfo=UTC)


def test_parquet_store_rejects_inf_and_invalid_ohlc(tmp_path) -> None:
    store = ParquetBarStore(tmp_path)
    ts = [
        datetime(2025, 1, 2, 21, 0, tzinfo=UTC),
        datetime(2025, 1, 3, 21, 0, tzinfo=UTC),
    ]

    for bad_val in (float("inf"), float("-inf"), float("nan")):
        df_bad = _make_bars_df(ts, [100.0, 101.0], highs=[105.0, bad_val])
        with pytest.raises(ValueError):
            store.save_bars("AAPL", df_bad)

    # high < low
    df_hl = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0, 100.0],
            "high": [98.0, 101.0],
            "low": [99.0, 99.0],
            "close": [98.5, 100.0],
            "volume": [1000.0, 1000.0],
        }
    )
    with pytest.raises(ValueError, match="high"):
        store.save_bars("AAPL", df_hl)

    # high < open
    df_ho = _make_bars_df(ts, [100.0, 100.0], opens=[105.0, 100.0], highs=[103.0, 101.0], lows=[99.0, 99.0])
    with pytest.raises(ValueError, match="high"):
        store.save_bars("AAPL", df_ho)

    # low > close
    df_lc = _make_bars_df(ts, [95.0, 100.0], opens=[100.0, 100.0], highs=[102.0, 101.0], lows=[97.0, 99.0])
    with pytest.raises(ValueError, match="low"):
        store.save_bars("AAPL", df_lc)

    # volume < 0
    df_neg_vol = _make_bars_df(ts, [100.0, 101.0], volumes=[1000.0, -1.0])
    with pytest.raises(ValueError, match="Volume"):
        store.save_bars("AAPL", df_neg_vol)

    # vwap <= 0
    for bad_vwap in (0.0, -10.0):
        df_bad_vwap = _make_bars_df(ts, [100.0, 101.0], vwaps=[100.0, bad_vwap])
        with pytest.raises(ValueError, match="vwap"):
            store.save_bars("AAPL", df_bad_vwap)


def test_parquet_store_atomic_save_load_and_strict_end_filtering(tmp_path) -> None:
    store = ParquetBarStore(tmp_path)
    t1 = datetime(2025, 1, 2, 21, 0, tzinfo=UTC)
    t2 = datetime(2025, 1, 3, 21, 0, tzinfo=UTC)
    t3 = datetime(2025, 1, 6, 21, 0, tzinfo=UTC)

    # Unsorted input with duplicate timestamp
    df = _make_bars_df(
        [t3, t1, t2, t1],
        [103.0, 99.0, 102.0, 100.0],
    )
    saved = store.save_bars("aapl", df)
    assert saved == 3
    assert store.list_symbols() == ["AAPL"]

    # No lingering .tmp artifacts
    leftovers = list((tmp_path / "bars_1d").glob("*.tmp"))
    assert leftovers == []

    # Strict end filtering (inclusive on t2, excludes t3)
    filtered = store.load_bars("AAPL", start=t1, end=t2)
    assert len(filtered) == 2
    assert list(filtered["close"]) == [100.0, 102.0]
    assert filtered["timestamp"].is_monotonic_increasing

    # Strict end excludes t2 when end is 1 microsecond earlier
    strict_before = store.load_bars("AAPL", end=t2 - timedelta(microseconds=1))
    assert len(strict_before) == 1
    assert strict_before["close"].iloc[0] == 100.0

    # Naive start/end raises
    with pytest.raises(ValueError, match="timezone-aware"):
        store.load_bars("AAPL", end=datetime(2025, 1, 3, 21, 0))


def test_parquet_store_corporate_action_mismatch_and_overwrite(tmp_path) -> None:
    store = ParquetBarStore(tmp_path)
    t1 = datetime(2025, 1, 2, 21, 0, tzinfo=UTC)
    t2 = datetime(2025, 1, 3, 21, 0, tzinfo=UTC)
    t3 = datetime(2025, 1, 6, 21, 0, tzinfo=UTC)

    store.save_bars("NVDA", _make_bars_df([t1, t2], [400.0, 410.0]))

    # Within 1e-4 tolerance on overlap succeeds cleanly and appends t3
    store.save_bars("NVDA", _make_bars_df([t2, t3], [410.01, 415.0]))
    assert len(store.load_bars("NVDA")) == 3

    # Split-adjusted cliff (>1e-4 relative diff on overlapping close) raises without allow_overwrite
    split_df = _make_bars_df([t2, t3], [102.5, 103.75])
    with pytest.raises(CorporateActionMismatchError):
        store.save_bars("NVDA", split_df, allow_overwrite=False)

    # Existing bars remain untouched after CorporateActionMismatchError
    unchanged = store.load_bars("NVDA")
    assert list(unchanged["close"]) == [400.0, 410.01, 415.0]

    # allow_overwrite=True replaces with split-adjusted history
    new_count = store.save_bars("NVDA", split_df, allow_overwrite=True)
    assert new_count == 2
    replaced = store.load_bars("NVDA")
    assert list(replaced["close"]) == [102.5, 103.75]


# ---------------------------------------------------------------------------
# StateDB tests
# ---------------------------------------------------------------------------


def test_state_db_wal_mode_enabled(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    StateDB(db_path)
    with closing(sqlite3.connect(str(db_path))) as conn:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert str(mode).lower() == "wal"


def test_state_db_news_dedup_across_utc_midnight_and_retag_preserves_published_at(
    tmp_path,
) -> None:
    db = StateDB(tmp_path / "state.db")

    # 18:30 ET = 23:30 UTC (same UTC date 2025-01-15)
    # 21:30 ET = 02:30 UTC on 2025-01-16 (crosses UTC midnight, STILL 2025-01-15 in ET!)
    et_evening_1 = datetime(2025, 1, 15, 18, 30, tzinfo=NY_TZ).astimezone(UTC)
    et_evening_2 = datetime(2025, 1, 15, 21, 30, tzinfo=NY_TZ).astimezone(UTC)
    assert et_evening_1.date() != et_evening_2.date()  # Different UTC dates
    assert et_evening_1.astimezone(NY_TZ).date() == et_evening_2.astimezone(NY_TZ).date()

    headline = "AAPL Beats Estimates & Raises Guidance!"
    assert compute_headline_hash(headline, et_evening_1) == compute_headline_hash(
        "  aapl beats estimates raises guidance ", et_evening_2
    )

    art1 = NewsArticle(
        article_id="art-1",
        published_at=et_evening_1,
        headline=headline,
        symbols=["AAPL"],
    )
    # Same headline on same NY date across UTC midnight, different article_id, re-tagged with MSFT
    art2 = NewsArticle(
        article_id="art-2",
        published_at=et_evening_2,
        headline=headline,
        symbols=["MSFT"],
    )
    # Same article_id re-upserted with NVDA and shifted published_at
    art3 = NewsArticle(
        article_id="art-1",
        published_at=et_evening_2 + timedelta(hours=5),
        headline=headline,
        symbols=["NVDA"],
    )
    # Next NY calendar date should NOT collide
    next_ny_day = datetime(2025, 1, 16, 8, 0, tzinfo=NY_TZ).astimezone(UTC)
    art4 = NewsArticle(
        article_id="art-4",
        published_at=next_ny_day,
        headline=headline,
        symbols=["AAPL"],
    )

    inserted_1 = db.upsert_news_articles([art1])
    inserted_2 = db.upsert_news_articles([art2, art3, art4])
    assert inserted_1 == 1
    assert inserted_2 == 1  # Only art4 is a distinct article

    # Re-tagged symbols MSFT and NVDA must preserve canonical art1 published_at (et_evening_1)
    msft_news = db.get_news_for_symbol_window(
        "MSFT", et_evening_1 - timedelta(minutes=1), et_evening_1 + timedelta(minutes=1)
    )
    assert len(msft_news) == 1
    assert msft_news[0].article_id == "art-1"
    assert msft_news[0].published_at == et_evening_1
    assert set(msft_news[0].symbols) == {"AAPL", "MSFT", "NVDA"}

    # Strict end_utc boundary excludes art4 when end_utc < next_ny_day
    aapl_before_next_day = db.get_news_for_symbol_window(
        "AAPL",
        et_evening_1 - timedelta(hours=1),
        next_ny_day - timedelta(seconds=1),
    )
    assert [a.article_id for a in aapl_before_next_day] == ["art-1"]

    aapl_inclusive = db.get_news_for_symbol_window(
        "AAPL", et_evening_1 - timedelta(hours=1), next_ny_day
    )
    assert [a.article_id for a in aapl_inclusive] == ["art-1", "art-4"]


def test_state_db_update_news_scores_validation_and_scored_window(tmp_path) -> None:
    db = StateDB(tmp_path / "state.db")
    t1 = datetime(2025, 2, 3, 15, 0, tzinfo=UTC)
    t2 = datetime(2025, 2, 3, 16, 0, tzinfo=UTC)
    db.upsert_news_articles(
        [
            NewsArticle(article_id="a1", published_at=t1, headline="Positive beat", symbols=["SPY"]),
            NewsArticle(article_id="a2", published_at=t2, headline="Later warning", symbols=["SPY"]),
        ]
    )

    for bad_score in (-1.01, 1.01, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="sentiment score"):
            db.update_news_scores([("a1", bad_score, "positive", 0.8, "lexicon-v1")])

    with pytest.raises(ValueError, match="confidence"):
        db.update_news_scores([("a1", 0.5, "positive", 1.2, "lexicon-v1")])

    with pytest.raises(ValueError, match="label"):
        db.update_news_scores([("a1", 0.5, "bullish", 0.8, "lexicon-v1")])

    with pytest.raises(ValueError, match="scorer_model"):
        db.update_news_scores([("a1", 0.5, "positive", 0.8, "   ")])

    updated = db.update_news_scores(
        [
            ("a1", 0.7, "positive", 0.9, "lexicon-v1"),
            ("a2", -0.6, "negative", 0.8, "lexicon-v1"),
        ]
    )
    assert updated == 2

    scored_upto_t1 = db.get_all_scored_news(end_utc=t1)
    assert [a.article_id for a in scored_upto_t1] == ["a1"]
    scored_all = db.get_all_scored_news(start_utc=t1, end_utc=t2)
    assert [a.article_id for a in scored_all] == ["a1", "a2"]


def test_state_db_llm_usage_and_budget_caps(tmp_path) -> None:
    db = StateDB(tmp_path / "state.db")
    quota = LLMQuotaConfig(daily_usd_cap=1.00, monthly_usd_cap=2.00)
    now = datetime(2025, 3, 15, 14, 0, tzinfo=UTC)

    # Prior month usage does not count toward March budget
    db.log_llm_usage(
        LLMUsageRecord(
            timestamp=datetime(2025, 2, 28, 23, 0, tzinfo=UTC),
            tier=1,
            model="flash",
            task="news",
            prompt_tokens=100,
            completion_tokens=50,
            estimated_usd=5.00,
        )
    )
    allowed, info = db.check_llm_budget(quota, now)
    assert allowed is True
    assert info["daily_spend"] == pytest.approx(0.0)
    assert info["monthly_spend"] == pytest.approx(0.0)

    # Earlier in March (different day): $1.50 -> counts toward monthly, not daily
    db.log_llm_usage(
        LLMUsageRecord(
            timestamp=datetime(2025, 3, 10, 12, 0, tzinfo=UTC),
            tier=2,
            model="pro",
            task="ops",
            prompt_tokens=1000,
            completion_tokens=500,
            estimated_usd=1.50,
        )
    )
    allowed, info = db.check_llm_budget(quota, now)
    assert allowed is True
    assert info["daily_spend"] == pytest.approx(0.0)
    assert info["monthly_spend"] == pytest.approx(1.50)

    # Today +$0.60 -> monthly = 2.10 >= 2.00 -> blocked by monthly cap even though daily < 1.00
    db.log_llm_usage(
        LLMUsageRecord(
            timestamp=datetime(2025, 3, 15, 9, 0, tzinfo=UTC),
            tier=1,
            model="flash",
            task="news",
            prompt_tokens=500,
            completion_tokens=100,
            estimated_usd=0.60,
        )
    )
    allowed, info = db.check_llm_budget(quota, now)
    assert allowed is False
    assert info["daily_spend"] == pytest.approx(0.60)
    assert info["monthly_spend"] == pytest.approx(2.10)

    # Fresh month with higher monthly cap -> hit daily cap directly
    quota_high_month = LLMQuotaConfig(daily_usd_cap=1.00, monthly_usd_cap=10.00)
    db.log_llm_usage(
        LLMUsageRecord(
            timestamp=datetime(2025, 3, 15, 11, 0, tzinfo=UTC),
            tier=1,
            model="flash",
            task="news",
            prompt_tokens=500,
            completion_tokens=100,
            estimated_usd=0.40,
        )
    )
    allowed, info = db.check_llm_budget(quota_high_month, now)
    assert allowed is False
    assert info["daily_spend"] == pytest.approx(1.00)


# ---------------------------------------------------------------------------
# AlpacaNewsProvider & FREDMacroProvider tests (mocked via respx)
# ---------------------------------------------------------------------------


@respx.mock
def test_alpaca_news_provider_pagination_null_payload_and_symbol_normalization() -> None:
    base_url = "https://data.alpaca.markets/v1beta1/news"
    start = datetime(2025, 4, 1, 0, 0, tzinfo=UTC)
    end = datetime(2025, 4, 2, 23, 59, tzinfo=UTC)

    route = respx.get(base_url).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "news": [
                        {
                            "id": "101",
                            "created_at": "2025-04-01T14:30:00Z",
                            "headline": "Apple launches new chip",
                            "summary": None,
                            "source": None,
                            "symbols": ["aapl", "extra_unrequested"],
                        }
                    ],
                    "next_page_token": "page-2-token",
                },
            ),
            httpx.Response(
                200,
                json={
                    "news": [
                        {
                            "id": "102",
                            "created_at": "2025-04-02T10:00:00Z",
                            "headline": "Microsoft cloud surges",
                            "summary": "Strong quarter",
                            "source": "benzinga",
                            "symbols": [" msft ", "AAPL"],
                        }
                    ],
                    "next_page_token": None,
                },
            ),
        ]
    )

    provider = AlpacaNewsProvider(api_key="k", secret_key="s", base_url=base_url)
    articles = provider.fetch_news(["aapl", " msft "], start=start, end=end)

    assert route.call_count == 2
    assert "page_token" not in route.calls[0].request.url.params
    assert route.calls[1].request.url.params["page_token"] == "page-2-token"
    assert route.calls[0].request.url.params["symbols"] == "AAPL,MSFT"

    assert [a.article_id for a in articles] == ["101", "102"]
    assert articles[0].symbols == ["AAPL"]
    assert articles[0].summary == ""
    assert articles[0].source == "benzinga"
    assert articles[1].symbols == ["MSFT", "AAPL"]


@respx.mock
def test_alpaca_news_provider_null_json_resilience() -> None:
    base_url = "https://data.alpaca.markets/v1beta1/news"
    start = datetime(2025, 4, 1, 0, 0, tzinfo=UTC)
    end = datetime(2025, 4, 2, 0, 0, tzinfo=UTC)

    respx.get(base_url).mock(
        side_effect=[
            httpx.Response(200, json={"news": None, "next_page_token": None}),
            httpx.Response(200, content=b"null", headers={"content-type": "application/json"}),
        ]
    )

    provider = AlpacaNewsProvider(api_key="k", secret_key="s", base_url=base_url)
    assert provider.fetch_news(["SPY"], start=start, end=end) == []
    assert provider.fetch_news(["SPY"], start=start, end=end) == []


@respx.mock
def test_fred_macro_provider_skips_dots_and_handles_empty_api_key() -> None:
    url = "https://api.stlouisfed.org/fred/series/observations"

    # Empty API key returns [] without hitting HTTP
    empty_provider = FREDMacroProvider(api_key="   ", base_url=url)
    assert empty_provider.fetch_macro(date(2025, 1, 1), date(2025, 1, 10)) == []

    route = respx.get(url).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "observations": [
                        {"date": "2025-01-02", "value": "."},
                        {"date": "2025-01-03", "value": "0.35"},
                        {"date": "2025-01-04", "value": "NaN"},
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "observations": [
                        {"date": "2025-01-02", "value": "17.4"},
                        {"date": "2025-01-03", "value": "."},
                    ]
                },
            ),
        ]
    )

    provider = FREDMacroProvider(
        api_key="fred-test-key",
        base_url=url,
        series_ids=("T10Y2Y", "VIXCLS"),
    )
    points = provider.fetch_macro(date(2025, 1, 1), date(2025, 1, 10))
    assert route.call_count == 2
    assert points == [
        MacroPoint(as_of_date=date(2025, 1, 3), series_id="T10Y2Y", value=0.35),
        MacroPoint(as_of_date=date(2025, 1, 2), series_id="VIXCLS", value=17.4),
    ]


# ---------------------------------------------------------------------------
# Synthetic generator determinism tests
# ---------------------------------------------------------------------------


def test_synthetic_generator_determinism_and_small_days(tmp_path) -> None:
    universe = [
        UniverseSymbol(symbol="SPY", asset_type=AssetType.etf, sector="Broad Market"),
        UniverseSymbol(symbol="AAPL", asset_type=AssetType.equity, sector="Technology"),
        UniverseSymbol(symbol="TLT", asset_type=AssetType.etf, sector="Fixed Income"),
    ]

    store1 = ParquetBarStore(tmp_path / "run1" / "bars")
    db1 = StateDB(tmp_path / "run1" / "state.db")
    summary1 = generate_synthetic_dataset(
        universe, store1, db1, days=40, end_date=date(2025, 6, 30), seed=123
    )

    store2 = ParquetBarStore(tmp_path / "run2" / "bars")
    db2 = StateDB(tmp_path / "run2" / "state.db")
    summary2 = generate_synthetic_dataset(
        universe, store2, db2, days=40, end_date=date(2025, 6, 30), seed=123
    )

    assert summary1 == summary2
    for sym in ("SPY", "AAPL", "TLT"):
        df1 = store1.load_bars(sym)
        df2 = store2.load_bars(sym)
        pd.testing.assert_frame_equal(df1, df2)
        assert not df1.empty
        assert np.isfinite(df1[["open", "high", "low", "close", "volume"]].to_numpy()).all()

    start_utc = datetime(2024, 1, 1, tzinfo=UTC)
    end_utc = datetime(2026, 1, 1, tzinfo=UTC)
    for sym in ("SPY", "AAPL"):
        n1 = db1.get_news_for_symbol_window(sym, start_utc, end_utc)
        n2 = db2.get_news_for_symbol_window(sym, start_utc, end_utc)
        assert [(a.article_id, a.published_at, a.headline, a.headline_hash) for a in n1] == [
            (a.article_id, a.published_at, a.headline, a.headline_hash) for a in n2
        ]

    for sid in ("VIXCLS", "BAMLH0A0HYM2", "T10Y2Y"):
        assert db1.get_macro_series(sid) == db2.get_macro_series(sid)

    assert db1.get_all_earnings_map() == db2.get_all_earnings_map()
    assert isinstance(db1.get_earnings_dates("AAPL"), list)

    # Small days=3 must not crash
    store_small = ParquetBarStore(tmp_path / "small" / "bars")
    db_small = StateDB(tmp_path / "small" / "state.db")
    small_summary = generate_synthetic_dataset(
        universe, store_small, db_small, days=3, end_date=date(2025, 6, 30), seed=7
    )
    assert small_summary["bars_saved"] == 9
    assert len(store_small.load_bars("SPY")) == 3
    assert isinstance(EarningsEvent(symbol="AAPL", earnings_date=date(2025, 6, 30)), EarningsEvent)
    assert math.isfinite(store_small.load_bars("AAPL")["close"].iloc[-1])


def test_yfinance_earnings_provider_resilience(monkeypatch: pytest.MonkeyPatch) -> None:
    from agy_trader.data.macro_and_calendar import YFinanceEarningsProvider

    class _FakeTicker:
        def __init__(self, sym: str) -> None:
            self.sym = sym

        def get_earnings_dates(self, limit: int = 40) -> pd.DataFrame | None:
            if self.sym == "FAIL":
                raise RuntimeError("yfinance HTTP 404")
            if self.sym == "EMPTY":
                return None
            idx = pd.DatetimeIndex(
                [
                    pd.Timestamp("2025-05-01 16:00:00", tz="America/New_York"),
                    pd.NaT,
                    pd.Timestamp("2025-02-01 16:00:00", tz="America/New_York"),
                    pd.Timestamp("2025-05-01 08:00:00", tz="America/New_York"),  # duplicate date
                ]
            )
            return pd.DataFrame({"EPS Estimate": [1.5, None, 1.4, 1.5]}, index=idx)

    monkeypatch.setattr("agy_trader.data.macro_and_calendar.yf.Ticker", _FakeTicker)

    provider = YFinanceEarningsProvider(limit=20)
    events = provider.fetch_earnings([" aapl ", "FAIL", "EMPTY", "aapl"])
    assert events == [
        EarningsEvent(symbol="AAPL", earnings_date=date(2025, 2, 1)),
        EarningsEvent(symbol="AAPL", earnings_date=date(2025, 5, 1)),
    ]

