from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
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
def test_alpaca_news_provider_429_retry_and_rate_limit_pacing() -> None:
    base_url = "https://data.alpaca.markets/v1beta1/news"
    start = datetime(2025, 4, 1, 0, 0, tzinfo=UTC)
    end = datetime(2025, 4, 2, 23, 59, tzinfo=UTC)

    route = respx.get(base_url).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "3"}),
            httpx.Response(
                200,
                headers={"X-RateLimit-Remaining": "1", "X-RateLimit-Reset": "1010"},
                json={
                    "news": [
                        {
                            "id": "201",
                            "created_at": "2025-04-01T12:00:00Z",
                            "headline": "First page after retry",
                            "symbols": ["SPY"],
                        }
                    ],
                    "next_page_token": "tok-2",
                },
            ),
            httpx.Response(429, headers={"X-RateLimit-Reset": "1025"}),
            httpx.Response(
                200,
                json={
                    "news": [
                        {
                            "id": "202",
                            "created_at": "2025-04-02T12:00:00Z",
                            "headline": "Second page after reset wait",
                            "symbols": ["SPY"],
                        }
                    ],
                    "next_page_token": None,
                },
            ),
        ]
    )

    sleeps: list[float] = []
    provider = AlpacaNewsProvider(
        api_key="k",
        secret_key="s",
        base_url=base_url,
        sleep_fn=sleeps.append,
        now_fn=lambda: 1000.0,
    )
    articles = provider.fetch_news(["SPY"], start=start, end=end)

    assert route.call_count == 4
    assert [a.article_id for a in articles] == ["201", "202"]
    assert sleeps == [3.0, 10.5, 25.5]


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


# ---------------------------------------------------------------------------
# Workstreams 3, 4, 5: News filters, Finnhub, SEC EDGAR, RSS/yfinance, DB
# ---------------------------------------------------------------------------


def test_news_filters_aliases_and_institutional_noise_regex() -> None:
    from agy_trader.data.news_filters import (
        canonicalize_article_symbols,
        expand_query_symbols,
        is_institutional_noise_headline,
    )

    assert expand_query_symbols(["aapl", "googl", "msft", "GOOGL", " "]) == [
        "AAPL",
        "GOOGL",
        "GOOG",
        "MSFT",
    ]
    assert canonicalize_article_symbols(
        ["goog", "aapl", "GOOGL", "tsla"], {"GOOGL", "AAPL"}
    ) == ["GOOGL", "AAPL"]
    assert canonicalize_article_symbols(["goog", "aapl", "googl"], set()) == ["GOOGL", "AAPL"]

    noise_samples = [
        "BlackRock Discloses Stake in Apple Inc.",
        "Vanguard Discloses Increased Position in MSFT",
        "Fidelity Discloses Decreased Holding in NVDA",
        "Form 13F: Citadel Reports Q1 Holdings",
        "Form 13G Filing for Alphabet",
        "Form 13D Activist Stake Update",
        "Schedule 13G/A Amendment Filed",
        "TR-1 Notification of Major Holdings",
        "Notification of Major Holdings: Total Voting Rights",
        "Total Voting Rights and Capital Update",
        "Major Shareholding Announcement",
        "SEC Filing: Form 4 Transaction Notice",
    ]
    for h in noise_samples:
        assert is_institutional_noise_headline(h) is True, f"Expected noise for {h!r}"

    signal_samples = [
        "Apple beats Q3 earnings estimates and raises guidance",
        "Alphabet unveils Gemini model update at developer conference",
        "Federal Reserve holds interest rates steady",
    ]
    for h in signal_samples:
        assert is_institutional_noise_headline(h) is False, f"Unexpected noise for {h!r}"


@respx.mock
def test_alpaca_news_provider_aliases_and_noise_filtering() -> None:
    base_url = "https://data.alpaca.markets/v1beta1/news"
    start = datetime(2025, 4, 1, 0, 0, tzinfo=UTC)
    end = datetime(2025, 4, 2, 23, 59, tzinfo=UTC)

    route = respx.get(base_url).mock(
        return_value=httpx.Response(
            200,
            json={
                "news": [
                    {
                        "id": "goog-1",
                        "created_at": "2025-04-01T15:00:00Z",
                        "headline": "Alphabet announces cloud expansion",
                        "symbols": ["GOOG", "UNREQUESTED"],
                    },
                    {
                        "id": "noise-1",
                        "created_at": "2025-04-01T16:00:00Z",
                        "headline": "BlackRock Discloses Stake in AAPL via Form 13G",
                        "symbols": ["AAPL"],
                    },
                ],
                "next_page_token": None,
            },
        )
    )

    provider = AlpacaNewsProvider(api_key="k", secret_key="s", base_url=base_url)
    articles = provider.fetch_news(["GOOGL", "AAPL"], start=start, end=end)

    assert route.call_count == 1
    assert route.calls[0].request.url.params["symbols"] == "GOOGL,GOOG,AAPL"
    assert [a.article_id for a in articles] == ["goog-1"]
    assert articles[0].symbols == ["GOOGL"]


@respx.mock
def test_finnhub_news_provider_chunking_pacing_retry_alias_and_noise() -> None:
    from agy_trader.data.finnhub_news import FinnhubNewsProvider

    base_url = "https://finnhub.io/api/v1/company-news"
    start = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    end = datetime(2025, 2, 14, 18, 0, tzinfo=UTC)  # 45 calendar days -> 2 chunks (30 + 15)

    # Empty API key short-circuits cleanly
    assert FinnhubNewsProvider(api_key="  ", base_url=base_url).fetch_news(["AAPL"], start, end) == []

    shared_ts = int(datetime(2025, 1, 15, 14, 0, tzinfo=UTC).timestamp())
    out_of_window_ts = int(datetime(2025, 1, 1, 10, 0, tzinfo=UTC).timestamp())  # before start_utc

    route = respx.get(base_url).mock(
        side_effect=[
            # GOOGL slice 1 -> 429 retry, then ok
            httpx.Response(429, headers={"Retry-After": "2"}),
            httpx.Response(
                200,
                json=[
                    {
                        "id": 9001,
                        "datetime": shared_ts,
                        "headline": "Alphabet posts record search revenue",
                        "summary": "Strong quarter",
                        "related": "GOOGL",
                    },
                    {
                        "id": 9002,
                        "datetime": shared_ts,
                        "headline": "Vanguard Discloses Increased Stake in GOOGL",
                        "related": "GOOGL",
                    },
                    {
                        "id": 9003,
                        "datetime": out_of_window_ts,
                        "headline": "Too early before start timestamp",
                        "related": "GOOGL",
                    },
                ],
            ),
            # GOOGL slice 2
            httpx.Response(200, json=[]),
            # GOOG slice 1 -> returns same 9001 plus unique GOOG item
            httpx.Response(
                200,
                json=[
                    {
                        "id": 9001,
                        "datetime": shared_ts,
                        "headline": "Alphabet posts record search revenue",
                        "related": "GOOG,OTHER",
                    },
                    {
                        "id": 9004,
                        "datetime": shared_ts + 60,
                        "headline": "Google Cloud signs enterprise deal",
                        "related": "GOOG",
                    },
                ],
            ),
            # GOOG slice 2 -> 503 then ok
            httpx.Response(503),
            httpx.Response(200, json=[]),
        ]
    )

    sleeps: list[float] = []
    provider = FinnhubNewsProvider(
        api_key="fh-key",
        base_url=base_url,
        chunk_days=30,
        min_interval_seconds=1.25,
        backoff_base_seconds=1.0,
        sleep_fn=sleeps.append,
        now_fn=lambda: 1000.0,
    )
    articles = provider.fetch_news(["GOOGL"], start=start, end=end)

    assert route.call_count == 6
    assert route.calls[0].request.headers["X-Finnhub-Token"] == "fh-key"
    assert "token" not in route.calls[0].request.url.params
    assert route.calls[0].request.url.params["symbol"] == "GOOGL"
    assert route.calls[0].request.url.params["from"] == "2025-01-01"
    assert route.calls[0].request.url.params["to"] == "2025-01-30"
    assert route.calls[2].request.url.params["from"] == "2025-01-31"
    assert route.calls[2].request.url.params["to"] == "2025-02-14"
    assert [a.article_id for a in articles] == ["finnhub:9001", "finnhub:9004"]
    assert all(a.source == "finnhub" and a.symbols == ["GOOGL"] for a in articles)
    assert 2.0 in sleeps  # Retry-After
    assert 1.25 in sleeps  # Pacing


@respx.mock
def test_sec_edgar_provider_cik_form4_xml_8k_and_pit_timestamps(tmp_path) -> None:
    from agy_trader.data.sec_edgar import SecEdgarProvider

    with pytest.raises(ValueError, match="user_agent"):
        SecEdgarProvider(user_agent="   ")

    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    sub_base = "https://data.sec.gov/submissions"
    arch_base = "https://www.sec.gov/Archives/edgar/data"

    respx.get(tickers_url).mock(
        return_value=httpx.Response(
            200,
            json={"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}},
        )
    )
    respx.get(f"{sub_base}/CIK0000320193.json").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(
                200,
                json={
                    "cik": "0000320193",
                    "filings": {
                        "recent": {
                            "accessionNumber": [
                                "0000320193-25-000010",
                                "0000320193-25-000011",
                                "0000320193-25-000012",
                                "0000320193-25-000013",
                            ],
                            "filingDate": [
                                "2025-03-01",
                                "2025-03-05",
                                "2025-03-06",
                                "2025-03-10",
                            ],
                            "acceptanceDateTime": [
                                "2025-03-01T21:15:30.000Z",
                                "",  # missing -> conservative 21:00 UTC fallback
                                "2025-03-06T14:00:00.000Z",
                                "2025-03-10T18:00:00.000Z",
                            ],
                            "form": ["4", "8-K", "4", "4"],
                            "primaryDocument": [
                                "xslF345X05/form4_buy.xml",
                                "f8k.htm",
                                "xslF345X05/form4_bad.xml",
                                "form4_non_officer.xml",
                            ],
                            "items": [[], ["4.02", "9.01"], [], []],
                        },
                        "files": [
                            {
                                "name": "CIK0000320193-submissions-001.json",
                                "filingFrom": "2024-01-01",
                                "filingTo": "2025-02-28",
                            },
                            {
                                "name": "CIK0000320193-submissions-old.json",
                                "filingFrom": "2010-01-01",
                                "filingTo": "2015-12-31",
                            },
                        ],
                    },
                },
            ),
        ]
    )
    respx.get(f"{sub_base}/CIK0000320193-submissions-001.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "accessionNumber": ["0000320193-25-000001"],
                "filingDate": ["2025-02-15"],
                "acceptanceDateTime": ["2025-02-15T16:30:00.000Z"],
                "form": ["8-K"],
                "primaryDocument": ["old8k.htm"],
                "items": ["2.02, 9.01"],
            },
        )
    )

    form4_buy_xml = """<?xml version="1.0"?>
    <ownershipDocument>
      <reportingOwner>
        <reportingOwnerId><rptOwnerCik>0001214156</rptOwnerCik><rptOwnerName>Cook Tim</rptOwnerName></reportingOwnerId>
        <reportingOwnerRelationship><isDirector>1</isDirector><isOfficer>1</isOfficer></reportingOwnerRelationship>
      </reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
          <transactionAmounts>
            <transactionShares><value>1000</value></transactionShares>
            <transactionPricePerShare><value>200.0</value></transactionPricePerShare>
          </transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
          <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
          <transactionAmounts>
            <transactionShares><value>200</value></transactionShares>
            <transactionPricePerShare><value>210.0</value></transactionPricePerShare>
          </transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
          <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
          <transactionAmounts>
            <transactionShares><value>5000</value></transactionShares>
            <transactionPricePerShare><value>0</value></transactionPricePerShare>
          </transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
    </ownershipDocument>"""

    form4_non_officer_xml = """<?xml version="1.0"?>
    <ownershipDocument>
      <reportingOwner>
        <reportingOwnerId><rptOwnerCik>0009999999</rptOwnerCik></reportingOwnerId>
        <reportingOwnerRelationship><isDirector>0</isDirector><isOfficer>0</isOfficer></reportingOwnerRelationship>
      </reportingOwner>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
          <transactionAmounts>
            <transactionShares><value>9999</value></transactionShares>
            <transactionPricePerShare><value>200.0</value></transactionPricePerShare>
          </transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
    </ownershipDocument>"""

    # Note: XSLT prefix xslF345X05/ must be stripped!
    respx.get(f"{arch_base}/320193/000032019325000010/form4_buy.xml").mock(
        return_value=httpx.Response(200, text=form4_buy_xml)
    )
    respx.get(f"{arch_base}/320193/000032019325000012/form4_bad.xml").mock(
        return_value=httpx.Response(200, text="<not-valid-xml")
    )
    respx.get(f"{arch_base}/320193/000032019325000013/form4_non_officer.xml").mock(
        return_value=httpx.Response(200, text=form4_non_officer_xml)
    )

    sleeps: list[float] = []
    cache_dir = tmp_path / "sec_cache"
    provider = SecEdgarProvider(
        user_agent="agy-trader test@example.com",
        cache_dir=cache_dir,
        min_interval_seconds=0.13,
        sleep_fn=sleeps.append,
        now_fn=lambda: 500.0,
    )
    events = provider.fetch_filings(
        ["AAPL", "SPY"],
        start=datetime(2025, 2, 1, 0, 0, tzinfo=UTC),
        end=datetime(2025, 3, 20, 0, 0, tzinfo=UTC),
    )

    assert (cache_dir / "company_tickers.json").exists()
    assert 1.0 in sleeps  # Retry-After
    assert 0.13 in sleeps  # Rate pacing

    by_acc = {ev.accession_number: ev for ev in events}
    # Malformed XML skipped; 4 valid events remain (2x 8-K, 2x Form 4)
    assert set(by_acc.keys()) == {
        "0000320193-25-000001",
        "0000320193-25-000010",
        "0000320193-25-000011",
        "0000320193-25-000013",
    }
    assert by_acc["0000320193-25-000001"].items == ["2.02", "9.01"]
    assert by_acc["0000320193-25-000011"].items == ["4.02", "9.01"]
    # Conservative 22:05 ET fallback (2025-03-05 EST -> 2025-03-06T03:05:00Z) when acceptanceDateTime is missing
    assert by_acc["0000320193-25-000011"].accepted_at == datetime(2025, 3, 6, 3, 5, tzinfo=UTC)

    f4_buy = by_acc["0000320193-25-000010"]
    # 1000*200 - 200*210 = 200,000 - 42,000 = 158,000 (ignores 'A' grant)
    assert f4_buy.net_insider_buy_usd == pytest.approx(158_000.0)
    assert f4_buy.insider_buyer_count == 1
    assert f4_buy.insider_seller_count == 1

    f4_non_off = by_acc["0000320193-25-000013"]
    assert f4_non_off.net_insider_buy_usd == 0.0
    assert f4_non_off.insider_buyer_count == 0
    assert f4_non_off.insider_seller_count == 0


def test_state_db_sec_filings_and_get_all_scored_news_filters(tmp_path) -> None:
    from agy_trader.core.models import SecFilingEvent

    db = StateDB(tmp_path / "state.db")
    t1 = datetime(2025, 3, 1, 15, 0, tzinfo=UTC)
    t2 = datetime(2025, 3, 2, 21, 0, tzinfo=UTC)

    events = [
        SecFilingEvent(
            accession_number="acc-1",
            symbol="aapl",
            cik="0000320193",
            form_type="4",
            accepted_at=t1,
            net_insider_buy_usd=250_000.0,
            insider_buyer_count=2,
            insider_seller_count=0,
        ),
        SecFilingEvent(
            accession_number="acc-2",
            symbol="AAPL",
            cik="0000320193",
            form_type="8-K",
            accepted_at=t2,
            items=["4.02", "9.01"],
        ),
    ]
    assert db.upsert_sec_filings(events) == 2
    assert db.upsert_sec_filings(events) == 2  # idempotent upsert

    loaded = db.get_sec_filings_map()
    assert list(loaded.keys()) == ["AAPL"]
    assert [e.accession_number for e in loaded["AAPL"]] == ["acc-1", "acc-2"]
    assert loaded["AAPL"][1].items == ["4.02", "9.01"]
    assert db.get_summary_counts()["sec_filings"] == 2

    # News filtering by scorer_model & sources
    db.upsert_news_articles(
        [
            NewsArticle(
                article_id="n-bz-lex",
                published_at=t1,
                headline="AAPL launches product",
                source="Benzinga",
                symbols=["AAPL"],
                sentiment_score=0.6,
                sentiment_label="positive",
                confidence=0.9,
                scorer_model="lexicon-v1",
            ),
            NewsArticle(
                article_id="n-fh-fb",
                published_at=t2,
                headline="AAPL beats expectations",
                source="finnhub",
                symbols=["AAPL"],
                sentiment_score=0.8,
                sentiment_label="positive",
                confidence=0.95,
                scorer_model="ProsusAI/finbert",
            ),
            NewsArticle(
                article_id="n-rss-lex",
                published_at=t2,
                headline="Fed FOMC minutes released",
                source="rss:fed_fomc",
                symbols=["SPY"],
                sentiment_score=-0.2,
                sentiment_label="negative",
                confidence=0.8,
                scorer_model="lexicon-v1",
            ),
        ]
    )

    assert db.get_all_scored_news(sources=[]) == []
    # Forward-only rss:fed_fomc excluded by default even with sources=None
    default_scored = db.get_all_scored_news()
    assert [a.article_id for a in default_scored] == ["n-bz-lex", "n-fh-fb"]
    # Explicit include_forward_only=True returns all scored rows
    with_fwd = db.get_all_scored_news(include_forward_only=True)
    assert [a.article_id for a in with_fwd] == ["n-bz-lex", "n-fh-fb", "n-rss-lex"]
    bz_fh = db.get_all_scored_news(sources=["BENZINGA", "finnhub"])
    assert [a.article_id for a in bz_fh] == ["n-bz-lex", "n-fh-fb"]
    only_lex_bz = db.get_all_scored_news(scorer_model="lexicon-v1", sources=["benzinga"])
    assert [a.article_id for a in only_lex_bz] == ["n-bz-lex"]

    # Cross-contamination guard: forward-only yfinance_news & model-eligible finnhub
    # sharing the exact same headline & date do NOT collide or leak symbols
    dup_ts = datetime(2025, 4, 10, 15, 0, tzinfo=UTC)
    yf_art = NewsArticle(
        article_id="yf:dup-1",
        published_at=dup_ts,
        headline="AAPL announces spring hardware event",
        source="yfinance_news",
        symbols=["SPY"],
        sentiment_score=-0.5,
        sentiment_label="negative",
        confidence=0.9,
        scorer_model="lexicon-v1",
    )
    fh_art = NewsArticle(
        article_id="finnhub:dup-1",
        published_at=dup_ts,
        headline="AAPL announces spring hardware event",
        source="finnhub",
        symbols=["AAPL"],
        sentiment_score=0.5,
        sentiment_label="positive",
        confidence=0.9,
        scorer_model="lexicon-v1",
    )
    assert db.upsert_news_articles([yf_art, fh_art]) == 2
    fh_rows = db.get_all_scored_news(sources=["finnhub"])
    fh_dup = next(a for a in fh_rows if a.article_id == "finnhub:dup-1")
    assert fh_dup.symbols == ["AAPL"]
    assert "SPY" not in fh_dup.symbols
    assert fh_dup.headline_hash == fh_art.headline_hash


def test_sec_edgar_acceptance_dt_edt_and_winter_est_fallback() -> None:
    from agy_trader.data.sec_edgar import _parse_sec_acceptance_dt

    # HIGH #1: 14-digit & naive ISO SEC acceptanceDateTime parsed as America/New_York -> UTC
    dt_14 = _parse_sec_acceptance_dt("20240501161500", "2024-05-01")
    assert dt_14 == datetime(2024, 5, 1, 20, 15, tzinfo=UTC)
    assert dt_14 > datetime(2024, 5, 1, 20, 0, tzinfo=UTC)  # strictly after 16:00 EDT

    dt_naive_iso = _parse_sec_acceptance_dt("2024-05-01T16:15:00", "2024-05-01")
    assert dt_naive_iso == datetime(2024, 5, 1, 20, 15, tzinfo=UTC)

    # HIGH #2: Winter EST fallback on 2024-01-10 -> 22:05 EST = 2024-01-11T03:05:00Z > 16:00 EST (21:00Z)
    dt_winter_fb = _parse_sec_acceptance_dt("", "2024-01-10")
    assert dt_winter_fb == datetime(2024, 1, 11, 3, 5, tzinfo=UTC)
    assert dt_winter_fb > datetime(2024, 1, 10, 21, 0, tzinfo=UTC)


@respx.mock
def test_feeds_collector_rss_and_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    from agy_trader.data.feeds import (
        RssFeedSpec,
        collect_official_rss_feeds,
        collect_yfinance_news,
    )

    rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <item>
          <title>Federal Reserve issues FOMC statement</title>
          <description>&lt;p&gt;Rates held steady.&lt;/p&gt;</description>
          <pubDate>Wed, 07 May 2025 18:00:00 GMT</pubDate>
          <guid>https://www.federalreserve.gov/newsevents/pressreleases/monetary20250507a.htm</guid>
        </item>
      </channel>
    </rss>"""

    atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>urn:eia:12345</id>
        <title>US crude inventories decline</title>
        <summary>Weekly petroleum report</summary>
        <published>2025-05-07T14:30:00Z</published>
      </entry>
    </feed>"""

    specs = (
        RssFeedSpec("Fed", "https://example.com/fed.xml", "rss:fed_fomc", ("SPY", "TLT")),
        RssFeedSpec("EIA", "https://example.com/eia.xml", "rss:eia", ("XLE",)),
        RssFeedSpec("FDA", "https://example.com/fda.xml", "rss:fda", ("XLV",)),
    )
    respx.get("https://example.com/fed.xml").mock(return_value=httpx.Response(200, text=rss_xml))
    respx.get("https://example.com/eia.xml").mock(return_value=httpx.Response(200, text=atom_xml))
    respx.get("https://example.com/fda.xml").mock(return_value=httpx.Response(500))

    rss_arts = collect_official_rss_feeds(feeds=specs)
    assert len(rss_arts) == 2
    assert {a.source for a in rss_arts} == {"rss:fed_fomc", "rss:eia"}
    fed_art = next(a for a in rss_arts if a.source == "rss:fed_fomc")
    assert fed_art.summary == "Rates held steady."
    assert fed_art.published_at == datetime(2025, 5, 7, 18, 0, tzinfo=UTC)
    assert fed_art.symbols == ["SPY", "TLT"]

    class _FakeNewsTicker:
        def __init__(self, sym: str) -> None:
            self.sym = sym

        @property
        def news(self) -> list[dict[str, object]]:
            if self.sym == "FAIL":
                raise RuntimeError("boom")
            if self.sym == "GOOG":
                return [
                    {
                        "id": "new-schema-1",
                        "content": {
                            "title": "Alphabet expands AI infrastructure",
                            "summary": "Cloud capex update",
                            "pubDate": "2025-05-08T13:00:00Z",
                        },
                    }
                ]
            return [
                {
                    "uuid": "legacy-1",
                    "title": "SPY hits new monthly volume high",
                    "providerPublishTime": int(datetime(2025, 5, 8, 12, 0, tzinfo=UTC).timestamp()),
                },
                {
                    "uuid": "noise-yf",
                    "title": "Form 13F: Institution Discloses Stake in SPY",
                    "providerPublishTime": int(datetime(2025, 5, 8, 12, 5, tzinfo=UTC).timestamp()),
                },
            ]

    monkeypatch.setattr("agy_trader.data.feeds.yf.Ticker", _FakeNewsTicker)
    yf_arts = collect_yfinance_news(["SPY", "GOOG", "FAIL"], max_per_symbol=10)
    assert [a.article_id for a in yf_arts] == ["yf:legacy-1", "yf:new-schema-1"]
    assert all(a.source == "yfinance_news" for a in yf_arts)
    assert yf_arts[1].symbols == ["GOOGL"]


def test_state_db_read_only_mode_wal_integrity_and_live_tables(tmp_path) -> None:
    # 1. read_only=True must NOT create parent dirs or DB files
    missing_dir = tmp_path / "nonexistent_dir"
    ro_missing = StateDB(missing_dir / "state.db", read_only=True)
    assert not missing_dir.exists()
    with pytest.raises(sqlite3.OperationalError):
        ro_missing.get_live_positions()

    # 2. Populate all 4 live tables + heartbeats via writable StateDB
    db_path = tmp_path / "live" / "state.db"
    db = StateDB(db_path)
    ok, msg = db.check_wal_integrity()
    assert ok is True
    assert msg == "ok"

    t0 = datetime(2026, 11, 20, 15, 40, tzinfo=UTC)
    db.upsert_live_order(
        client_order_id="agy:20261120:1540:AAPL:B:main",
        broker_order_id="brk-1",
        run_id="run-1",
        symbol="aapl",
        side="buy",
        order_type="limit",
        qty=10.0,
        limit_price=200.30,
        status="filled",
        filled_qty=10.0,
        filled_avg_price=200.10,
        reference_price=200.00,
        realized_slippage_bps=5.0,
        is_risk_reducing=False,
        submitted_at=t0,
        updated_at=t0,
    )
    fetched_ord = db.get_live_order("agy:20261120:1540:AAPL:B:main")
    assert fetched_ord is not None
    assert fetched_ord["symbol"] == "AAPL"
    assert fetched_ord["realized_slippage_bps"] == pytest.approx(5.0)
    assert fetched_ord["is_risk_reducing"] is False
    assert len(db.get_recent_live_orders(limit=10)) == 1

    db.replace_live_positions(
        [
            {
                "symbol": "AAPL",
                "qty": 10.0,
                "avg_entry_price": 200.10,
                "market_price": 201.00,
                "unrealized_pnl": 9.0,
                "stop_order_id": "stp-1",
                "stop_price": 190.00,
                "updated_at": t0,
            },
            {
                "symbol": "TSLA",
                "qty": -5.0,
                "avg_entry_price": 250.00,
                "market_price": 248.00,
                "unrealized_pnl": 10.0,
                "stop_order_id": "stp-2",
                "stop_price": 265.00,
                "updated_at": t0,
            },
        ]
    )
    # Second replace atomically replaces stale positions
    db.replace_live_positions(
        [
            {
                "symbol": "AAPL",
                "qty": 10.0,
                "avg_entry_price": 200.10,
                "market_price": 202.00,
                "unrealized_pnl": 19.0,
                "stop_order_id": "stp-1",
                "stop_price": 190.00,
                "updated_at": t0,
            }
        ]
    )
    positions = db.get_live_positions()
    assert [p["symbol"] for p in positions] == ["AAPL"]
    assert positions[0]["stop_order_id"] == "stp-1"

    db.insert_portfolio_snapshot(
        timestamp=t0,
        equity=100_500.0,
        cash=50_000.0,
        buying_power=100_000.0,
        gross_exposure=0.505,
        net_exposure=0.505,
        daily_pnl_pct=0.005,
        drawdown_pct=0.0,
        high_water_mark=100_500.0,
        spy_close=595.25,
        ew50_index=101.2,
    )
    latest_snap = db.get_latest_portfolio_snapshot()
    assert latest_snap is not None
    assert latest_snap["equity"] == pytest.approx(100_500.0)
    assert len(db.get_portfolio_snapshots(limit=100)) == 1

    assert db.has_session_rebalance_completed("2026-11-20", "15:40") is False
    db.record_daemon_event(
        "agy-trader",
        "rebalance",
        {"session_date": "2026-11-20", "checkpoint": "15:40", "status": "started"},
        timestamp=t0,
    )
    assert db.has_session_rebalance_completed("2026-11-20", "15:40") is False
    db.record_daemon_event(
        "agy-trader",
        "rebalance",
        {"session_date": "2026-11-20", "checkpoint": "15:40", "status": "completed"},
        timestamp=t0 + timedelta(seconds=15),
    )
    assert db.has_session_rebalance_completed("2026-11-20", "15:40") is True
    assert len(db.get_daemon_events(daemon_name="agy-trader", event_type="rebalance")) == 2

    db.upsert_daemon_heartbeat("agy-trader", "ok", {"loop_ms": 42}, now_utc=t0)
    heartbeats = db.get_daemon_heartbeats()
    assert heartbeats["agy-trader"]["status"] == "ok"
    assert heartbeats["agy-trader"]["details"]["loop_ms"] == 42

    # 3. read_only=True reads all tables cleanly & blocks all writes at SQLite layer
    ro_db = StateDB(db_path, read_only=True)
    assert len(ro_db.get_live_positions()) == 1
    assert ro_db.get_latest_portfolio_snapshot() is not None
    assert ro_db.has_session_rebalance_completed("2026-11-20", "15:40") is True
    with pytest.raises(sqlite3.OperationalError):
        ro_db.record_daemon_event("dashboard", "boot", {})
    with pytest.raises(sqlite3.OperationalError):
        ro_db.upsert_daemon_heartbeat("dashboard", "ok")

    # 4. Corrupted DB fails check_wal_integrity()
    corrupt_path = tmp_path / "corrupt.db"
    corrupt_path.write_bytes(b"not a valid sqlite database file at all")
    ro_corrupt = StateDB(corrupt_path, read_only=True)
    corrupt_ok, corrupt_msg = ro_corrupt.check_wal_integrity()
    assert corrupt_ok is False
    assert corrupt_msg

    # 5. HIGH #6, MEDIUM #11, LOW #14: telemetry pruning, index, open statuses, busy_timeout
    with closing(db._connect()) as conn:
        busy = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        assert int(busy) == 30000
        indexes = {
            str(r[1])
            for r in conn.execute("PRAGMA index_list('daemon_events');").fetchall()
        }
        assert "idx_daemon_events_type_ts" in indexes

    # Insert >500 non-rebalance events after an older rebalance to prove no 500-row scan bug
    old_ts = t0 - timedelta(days=120)
    db.record_daemon_event(
        "agy-trader",
        "rebalance",
        {"session_date": "2026-07-23", "checkpoint": "15:40", "status": "completed"},
        timestamp=old_ts,
    )
    for i in range(520):
        db.record_daemon_event(
            "agy-trader",
            "reconcile",
            {"seq": i},
            timestamp=old_ts + timedelta(minutes=i),
        )
    assert db.has_session_rebalance_completed("2026-07-23", "15:40") is True

    db.insert_portfolio_snapshot(
        timestamp=old_ts,
        equity=90_000.0,
        cash=90_000.0,
        buying_power=90_000.0,
        gross_exposure=0.0,
        net_exposure=0.0,
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
        high_water_mark=90_000.0,
    )
    pruned = db.prune_old_telemetry(keep_days=90, now_utc=t0)
    assert pruned["daemon_events"] == 520
    assert pruned["portfolio_snapshots"] == 1
    # Rebalance checkpoints & recent snapshots are preserved
    assert db.has_session_rebalance_completed("2026-07-23", "15:40") is True
    assert len(db.get_portfolio_snapshots(limit=100)) == 1

    # MEDIUM #11: get_open_live_orders includes accepted_for_bidding and calculated
    for idx, st in enumerate(("accepted_for_bidding", "calculated")):
        db.upsert_live_order(
            client_order_id=f"agy:20261120:1540:SPY:B:m{idx}",
            run_id="r-open",
            symbol="SPY",
            side="buy",
            order_type="limit",
            qty=1.0,
            status=st,
            reference_price=500.0,
            is_risk_reducing=False,
        )
    open_cids = {o["client_order_id"] for o in db.get_open_live_orders()}
    assert "agy:20261120:1540:SPY:B:m0" in open_cids
    assert "agy:20261120:1540:SPY:B:m1" in open_cids


def test_portfolio_snapshots_spmo_qmom_schema_migration_and_backfill(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy_state.db"
    # Create legacy 12-column portfolio_snapshots table without spmo_close / qmom_close
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                cash REAL NOT NULL,
                buying_power REAL NOT NULL,
                gross_exposure REAL NOT NULL,
                net_exposure REAL NOT NULL,
                daily_pnl_pct REAL NOT NULL,
                drawdown_pct REAL NOT NULL,
                high_water_mark REAL NOT NULL,
                spy_close REAL,
                ew50_index REAL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO portfolio_snapshots (
                timestamp, equity, cash, buying_power, gross_exposure, net_exposure,
                daily_pnl_pct, drawdown_pct, high_water_mark, spy_close, ew50_index
            ) VALUES
                ('2026-11-20T20:40:00.000000Z', 100000.0, 40000.0, 140000.0, 0.6, 0.6, 0.0, 0.0, 100000.0, 500.0, 100.0),
                ('2026-11-23T20:40:00.000000Z', 101000.0, 40000.0, 141000.0, 0.6, 0.6, 0.01, 0.0, 101000.0, 505.0, 101.0)
            """
        )

    # Opening pre-migration DB in read_only mode must succeed without IndexError
    ro_db = StateDB(db_path, read_only=True)
    ro_snaps = ro_db.get_portfolio_snapshots(limit=10)
    assert len(ro_snaps) == 2
    assert ro_snaps[0]["spmo_close"] is None
    assert ro_snaps[0]["qmom_close"] is None

    # Opening writable StateDB idempotently adds spmo_close and qmom_close
    rw_db = StateDB(db_path)
    with closing(rw_db._connect()) as conn:
        cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(portfolio_snapshots)").fetchall()}
    assert {"spmo_close", "qmom_close"}.issubset(cols)

    id3 = rw_db.insert_portfolio_snapshot(
        timestamp=datetime(2026, 11, 24, 20, 40, tzinfo=UTC),
        equity=102_000.0,
        cash=40_000.0,
        buying_power=142_000.0,
        gross_exposure=0.6,
        net_exposure=0.6,
        daily_pnl_pct=0.01,
        drawdown_pct=0.0,
        high_water_mark=102_000.0,
        spy_close=510.0,
        ew50_index=102.0,
        spmo_close=110.0,
        qmom_close=60.0,
    )
    updated = rw_db.backfill_portfolio_snapshot_benchmarks(
        [
            (1, 100.0, 50.0),
            (2, 105.0, None),
            (id3, 999.0, 999.0),  # must NOT overwrite existing non-null values
        ]
    )
    assert updated == 3
    snaps = rw_db.get_portfolio_snapshots(limit=10)
    assert snaps[0]["spmo_close"] == pytest.approx(100.0)
    assert snaps[0]["qmom_close"] == pytest.approx(50.0)
    assert snaps[1]["spmo_close"] == pytest.approx(105.0)
    assert snaps[1]["qmom_close"] is None
    assert snaps[2]["spmo_close"] == pytest.approx(110.0)
    assert snaps[2]["qmom_close"] == pytest.approx(60.0)



