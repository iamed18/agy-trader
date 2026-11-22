from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from agy_trader.core.config import StrategyConfig
from agy_trader.core.models import MacroPoint, NewsArticle
from agy_trader.nlp.sentiment import (
    FinBERTScorer,
    LexiconSentimentScorer,
    SentimentAggregator,
)
from agy_trader.strategy.features import (
    FeatureEngine,
    compute_macro_regime_scale,
    macro_point_available_at_utc,
)

NY_TZ = ZoneInfo("America/New_York")


def _make_price_df(
    dates: list[date],
    closes: np.ndarray,
    *,
    spread: float = 0.01,
) -> pd.DataFrame:
    timestamps = [
        datetime.combine(d, datetime.min.time(), tzinfo=NY_TZ).replace(hour=16).astimezone(UTC)
        for d in dates
    ]
    opens = closes * (1.0 - spread * 0.25)
    highs = np.maximum(opens, closes) * (1.0 + spread)
    lows = np.minimum(opens, closes) * (1.0 - spread)
    volumes = np.full(len(closes), 5_000_000.0)
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(timestamps, tz="UTC"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "vwap": (highs + lows + 2.0 * closes) / 4.0,
        }
    )


def _scored_article(
    article_id: str,
    published_at: datetime,
    headline: str,
    score: float,
    confidence: float = 0.9,
) -> NewsArticle:
    label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
    return NewsArticle(
        article_id=article_id,
        published_at=published_at,
        headline=headline,
        symbols=["AAPL"],
        sentiment_score=score,
        sentiment_label=label,
        confidence=confidence,
        scorer_model="lexicon-v1",
    )


# ---------------------------------------------------------------------------
# LexiconSentimentScorer tests
# ---------------------------------------------------------------------------


def test_lexicon_scorer_polarity_negation_and_neutral() -> None:
    scorer = LexiconSentimentScorer()
    preds = scorer.score_texts(
        [
            "Company beats estimates and raises guidance on strong demand",
            "Accelerating growth drives record profit",
            "Management cuts guidance amid weak revenue",
            "SEC investigation launched into accounting probe",
            "Company did not beats estimates this quarter",
            "Board schedules annual shareholder meeting for Tuesday",
        ]
    )

    assert preds[0].score > 0.0 and preds[0].label == "positive"
    assert preds[1].score > 0.0 and preds[1].label == "positive"
    assert preds[2].score < 0.0 and preds[2].label == "negative"
    assert preds[3].score < 0.0 and preds[3].label == "negative"
    # Negation reversal: "not beats estimates" must NOT be positive
    assert preds[4].score <= 0.0
    assert preds[4].label != "positive"
    # Neutral text
    assert preds[5].score == 0.0
    assert preds[5].label == "neutral"


# ---------------------------------------------------------------------------
# SentimentAggregator tests
# ---------------------------------------------------------------------------


def test_sentiment_aggregator_half_life_decay_and_dedup_and_velocity_polarity() -> None:
    agg = SentimentAggregator(half_life_hours=24.0, lookback_hours=72.0, baseline_days=30)
    as_of = datetime(2025, 5, 10, 20, 0, tzinfo=UTC)

    # Case A: recent positive (+0.8 at 1h ago) vs older negative (-0.8 at 48h ago)
    # Decay weight at 1h > weight at 48h -> net raw_polarity must be strictly positive.
    recent_pos = _scored_article("p1", as_of - timedelta(hours=1), "Beats estimates", 0.8, 0.8)
    older_neg = _scored_article("n1", as_of - timedelta(hours=48), "Cuts guidance", -0.8, 0.8)
    res_recent_pos = agg.aggregate_for_symbol([recent_pos, older_neg], as_of=as_of)
    assert res_recent_pos.raw_polarity > 0.0
    assert res_recent_pos.news_score > 0.0

    # Flip ages: older positive vs recent negative -> strictly negative
    older_pos = _scored_article("p2", as_of - timedelta(hours=48), "Beats estimates", 0.8, 0.8)
    recent_neg = _scored_article("n2", as_of - timedelta(hours=1), "Cuts guidance", -0.8, 0.8)
    res_recent_neg = agg.aggregate_for_symbol([older_pos, recent_neg], as_of=as_of)
    assert res_recent_neg.raw_polarity < 0.0
    assert res_recent_neg.news_score < 0.0

    # Duplicate headline_hash on same NY date must NOT double-count velocity or alter sentiment
    art_base = _scored_article("d1", as_of - timedelta(hours=2), "AAPL raises guidance", 0.7, 0.9)
    art_dup = _scored_article(
        "d2", as_of - timedelta(hours=1), "aapl raises guidance!", 0.7, 0.9
    )
    assert art_base.headline_hash == art_dup.headline_hash
    res_single = agg.aggregate_for_symbol([art_base], as_of=as_of)
    res_with_dup = agg.aggregate_for_symbol([art_base, art_dup], as_of=as_of)
    assert res_with_dup == res_single
    assert res_with_dup.article_count_window == 1

    # Negative velocity_z (busy prior 29 days, quiet last 24h) must NOT flip sentiment polarity
    prior_articles = [
        _scored_article(
            f"hist-{d}-{k}",
            as_of - timedelta(days=d, hours=k + 1),
            f"Prior positive update {d}-{k}",
            0.5,
            0.8,
        )
        for d in range(4, 18)
        for k in range(3)
    ]
    quiet_recent = _scored_article(
        "recent-neg", as_of - timedelta(hours=36), "Recent guidance cut", -0.7, 0.9
    )
    res_neg_vel = agg.aggregate_for_symbol([*prior_articles, quiet_recent], as_of=as_of)
    assert res_neg_vel.velocity_z < 0.0
    assert res_neg_vel.raw_polarity < 0.0
    assert res_neg_vel.news_score < 0.0
    assert res_neg_vel.news_score == pytest.approx(res_neg_vel.raw_polarity)


def test_sentiment_aggregator_strict_zero_lookahead() -> None:
    agg = SentimentAggregator(half_life_hours=24.0, lookback_hours=72.0, baseline_days=30)
    as_of = datetime(2025, 5, 10, 20, 0, tzinfo=UTC)

    past_articles = [
        _scored_article("p1", as_of - timedelta(hours=5), "Strong demand", 0.65, 0.85),
        _scored_article("p2", as_of, "Exact boundary headline", 0.40, 0.80),
    ]
    baseline = agg.aggregate_for_symbol(past_articles, as_of=as_of)

    future_articles = [
        _scored_article(
            "fut1",
            as_of + timedelta(seconds=1),
            "Catastrophic fraud and bankruptcy",
            -1.0,
            1.0,
        ),
        _scored_article(
            "fut2",
            as_of + timedelta(hours=12),
            "Massive regulatory probe",
            -1.0,
            1.0,
        ),
    ]
    with_future = agg.aggregate_for_symbol([*past_articles, *future_articles], as_of=as_of)
    assert with_future == baseline


# ---------------------------------------------------------------------------
# FeatureEngine strict zero-lookahead test
# ---------------------------------------------------------------------------


def test_feature_engine_strict_zero_lookahead_120_bars() -> None:
    rng = np.random.default_rng(99)
    bdays = [ts.date() for ts in pd.bdate_range(start=date(2025, 1, 2), periods=120)]
    symbols = ["SPY", "AAPL", "MSFT", "NVDA", "XOM"]

    bars: dict[str, pd.DataFrame] = {}
    for idx, sym in enumerate(symbols):
        rets = rng.normal(0.0005 * (idx + 1), 0.015, size=120)
        closes = 100.0 * np.exp(np.cumsum(rets))
        bars[sym] = _make_price_df(bdays, closes)

    day80_idx = 79  # 80th bar (0-indexed)
    as_of_day80 = datetime.combine(bdays[day80_idx], datetime.min.time(), tzinfo=NY_TZ).replace(
        hour=16
    ).astimezone(UTC)

    past_news = {
        "AAPL": [
            _scored_article(
                "n-past-1",
                as_of_day80 - timedelta(hours=6),
                "AAPL beats estimates",
                0.75,
                0.9,
            )
        ],
        "MSFT": [
            _scored_article(
                "n-past-2",
                as_of_day80 - timedelta(days=2),
                "MSFT steady cloud growth",
                0.35,
                0.8,
            )
        ],
    }
    past_macro = [
        MacroPoint(as_of_date=bdays[day80_idx - 5], series_id="VIXCLS", value=16.0),
        MacroPoint(as_of_date=bdays[day80_idx - 5], series_id="T10Y2Y", value=0.25),
    ]

    engine = FeatureEngine(StrategyConfig())
    baseline_signals = engine.build_signals(
        as_of=as_of_day80,
        bars_by_symbol=bars,
        news_by_symbol=past_news,
        macro_points=past_macro,
    )
    assert len(baseline_signals) == len(symbols)

    # Violently mutate days 81..120 (+500% / -90% prices), inject extreme future news & macro
    mutated_bars: dict[str, pd.DataFrame] = {}
    for idx, (sym, df) in enumerate(bars.items()):
        df_mut = df.copy()
        mult = 6.0 if idx % 2 == 0 else 0.10
        for col in ("open", "high", "low", "close", "vwap"):
            df_mut.loc[day80_idx + 1 :, col] = df_mut.loc[day80_idx + 1 :, col] * mult
        mutated_bars[sym] = df_mut

    mutated_news = {
        sym: [
            *past_news.get(sym, []),
            _scored_article(
                f"fut-{sym}",
                as_of_day80 + timedelta(seconds=1),
                f"{sym} extreme future shock",
                -1.0 if sym == "AAPL" else 1.0,
                1.0,
            ),
        ]
        for sym in symbols
    }
    mutated_macro = [
        *past_macro,
        # Same day as day 80, but FRED only releases next business day!
        MacroPoint(as_of_date=bdays[day80_idx], series_id="VIXCLS", value=55.0),
        MacroPoint(as_of_date=bdays[day80_idx], series_id="BAMLH0A0HYM2", value=8.5),
        MacroPoint(as_of_date=bdays[day80_idx], series_id="T10Y2Y", value=-1.0),
        # Future date
        MacroPoint(as_of_date=bdays[day80_idx + 1], series_id="VIXCLS", value=60.0),
    ]

    after_mutation_signals = engine.build_signals(
        as_of=as_of_day80,
        bars_by_symbol=mutated_bars,
        news_by_symbol=mutated_news,
        macro_points=mutated_macro,
    )

    assert after_mutation_signals == baseline_signals


# ---------------------------------------------------------------------------
# FRED macro release-lag & holiday tests
# ---------------------------------------------------------------------------


def test_fred_macro_release_lag_friday_monday_and_presidents_day_holiday() -> None:
    # Regular Friday -> Monday 16:30 ET release for T10Y2Y
    fri_normal = date(2025, 2, 7)
    pt_normal = MacroPoint(as_of_date=fri_normal, series_id="T10Y2Y", value=-0.8)
    avail_normal = macro_point_available_at_utc(pt_normal)
    assert avail_normal.astimezone(NY_TZ) == datetime(2025, 2, 10, 16, 30, tzinfo=NY_TZ)

    vix_pt = MacroPoint(as_of_date=fri_normal, series_id="VIXCLS", value=40.0)
    assert macro_point_available_at_utc(vix_pt).astimezone(NY_TZ) == datetime(
        2025, 2, 10, 11, 30, tzinfo=NY_TZ
    )

    # Monday 10:00 ET: neither VIXCLS (11:30 ET) nor T10Y2Y (16:30 ET) from Friday is visible
    mon_1000 = datetime(2025, 2, 10, 10, 0, tzinfo=NY_TZ).astimezone(UTC)
    assert compute_macro_regime_scale(mon_1000, None, [vix_pt, pt_normal]) == 1.0

    # Monday 12:00 ET: VIXCLS=40.0 is visible (scale -> 0.45), T10Y2Y (-0.8) is NOT yet visible
    # Verify via VIXCLS=26 (scale 0.70) + T10Y2Y=-0.8 (would drop to 0.50 once visible)
    vix_26 = MacroPoint(as_of_date=fri_normal, series_id="VIXCLS", value=26.0)
    mon_1200 = datetime(2025, 2, 10, 12, 0, tzinfo=NY_TZ).astimezone(UTC)
    assert compute_macro_regime_scale(mon_1200, None, [vix_26, pt_normal]) == 0.70

    # Monday 16:30 ET: T10Y2Y=-0.8 IS now visible -> drops scale to 0.50
    mon_1630 = datetime(2025, 2, 10, 16, 30, tzinfo=NY_TZ).astimezone(UTC)
    assert compute_macro_regime_scale(mon_1630, None, [vix_26, pt_normal]) == 0.50

    # Presidents' Day weekend: Friday 2025-02-14 -> Monday 2025-02-17 is holiday -> Tuesday 2025-02-18!
    fri_before_pres_day = date(2025, 2, 14)
    pt_holiday = MacroPoint(as_of_date=fri_before_pres_day, series_id="T10Y2Y", value=-0.8)
    vix_holiday = MacroPoint(as_of_date=fri_before_pres_day, series_id="VIXCLS", value=28.0)

    avail_t10 = macro_point_available_at_utc(pt_holiday).astimezone(NY_TZ)
    avail_vix = macro_point_available_at_utc(vix_holiday).astimezone(NY_TZ)
    assert avail_t10 == datetime(2025, 2, 18, 16, 30, tzinfo=NY_TZ)
    assert avail_vix == datetime(2025, 2, 18, 11, 30, tzinfo=NY_TZ)

    # Monday (Presidents' Day) 17:00 ET must NOT reveal Friday's macro prints
    mon_holiday_1700 = datetime(2025, 2, 17, 17, 0, tzinfo=NY_TZ).astimezone(UTC)
    assert compute_macro_regime_scale(mon_holiday_1700, None, [vix_holiday, pt_holiday]) == 1.0

    # Tuesday 16:30 ET reveals both
    tue_1630 = datetime(2025, 2, 18, 16, 30, tzinfo=NY_TZ).astimezone(UTC)
    assert compute_macro_regime_scale(tue_1630, None, [vix_holiday, pt_holiday]) == 0.50


# ---------------------------------------------------------------------------
# Corrupt SPY bar fallback & single-symbol cross-section tests
# ---------------------------------------------------------------------------


def test_corrupt_spy_fallback_and_single_symbol_zscore_zero() -> None:
    bdays = [ts.date() for ts in pd.bdate_range(start=date(2025, 1, 2), periods=80)]
    as_of = datetime.combine(bdays[-1], datetime.min.time(), tzinfo=NY_TZ).replace(
        hour=16
    ).astimezone(UTC)

    # Healthy universe with NaN/zero injected into SPY
    aapl_closes = np.linspace(100.0, 140.0, 80)
    msft_closes = np.linspace(200.0, 190.0, 80)
    spy_closes = np.linspace(400.0, 420.0, 80)

    spy_df = _make_price_df(bdays, spy_closes)
    spy_df.loc[70, "close"] = float("nan")
    spy_df.loc[72, "close"] = 0.0

    bars = {
        "SPY": spy_df,
        "AAPL": _make_price_df(bdays, aapl_closes),
        "MSFT": _make_price_df(bdays, msft_closes),
    }

    engine = FeatureEngine(StrategyConfig())
    signals = engine.build_signals(as_of=as_of, bars_by_symbol=bars)
    # Corrupt SPY is skipped, but AAPL and MSFT still produce valid finite signals
    assert [s.symbol for s in signals] == ["AAPL", "MSFT"]

    # Single-symbol cross-section: trend_score and pullback_score must be 0.0 (never leak raw z)
    single_signals = engine.build_signals(
        as_of=as_of,
        bars_by_symbol={"AAPL": _make_price_df(bdays, aapl_closes), "SPY": spy_df},
    )
    assert len(single_signals) == 1
    assert single_signals[0].symbol == "AAPL"
    assert single_signals[0].trend_score == 0.0
    assert single_signals[0].pullback_score == 0.0


# ---------------------------------------------------------------------------
# Missing-news renormalization & SPY SMA200 defensive scale tests
# ---------------------------------------------------------------------------


def test_feature_engine_missing_news_does_not_shrink_composite() -> None:
    bdays = [ts.date() for ts in pd.bdate_range(start=date(2025, 1, 2), periods=80)]
    as_of = datetime.combine(bdays[-1], datetime.min.time(), tzinfo=NY_TZ).replace(
        hour=16
    ).astimezone(UTC)

    bars = {
        "SPY": _make_price_df(bdays, np.linspace(400.0, 420.0, 80)),
        "AAPL": _make_price_df(bdays, np.linspace(100.0, 140.0, 80)),
        "MSFT": _make_price_df(bdays, np.linspace(200.0, 180.0, 80)),
    }
    # MSFT has 0 news articles; AAPL has 1 scored article.
    news = {
        "AAPL": [
            _scored_article("a1", as_of - timedelta(hours=3), "AAPL strong beat", 0.75, 0.9),
        ]
    }

    cfg_no_news_w = StrategyConfig(w_trend=0.45, w_pullback=0.25, w_news_sentiment=0.0)
    cfg_with_news_w = StrategyConfig(w_trend=0.45, w_pullback=0.25, w_news_sentiment=0.30)

    sig_w0 = {
        s.symbol: s
        for s in FeatureEngine(cfg_no_news_w).build_signals(
            as_of=as_of, bars_by_symbol=bars, news_by_symbol=news
        )
    }
    sig_w30 = {
        s.symbol: s
        for s in FeatureEngine(cfg_with_news_w).build_signals(
            as_of=as_of, bars_by_symbol=bars, news_by_symbol=news
        )
    }

    # MSFT (0 news) must have identical composite_score regardless of w_news_sentiment
    assert sig_w30["MSFT"].composite_score == sig_w0["MSFT"].composite_score
    # AAPL (>=1 news) blends news_score as expected
    assert sig_w30["AAPL"].composite_score != sig_w0["AAPL"].composite_score
    assert sig_w30["AAPL"].composite_score > sig_w0["AAPL"].composite_score

    # Pure-news config (w_trend=0, w_pullback=0, w_news_sentiment=1.0) on symbol with 0 news -> 0.0 cleanly
    cfg_pure_news = StrategyConfig(w_trend=0.0, w_pullback=0.0, w_news_sentiment=1.0)
    sig_pure = {
        s.symbol: s
        for s in FeatureEngine(cfg_pure_news).build_signals(
            as_of=as_of, bars_by_symbol=bars, news_by_symbol=news
        )
    }
    assert sig_pure["MSFT"].composite_score == 0.0
    assert sig_pure["AAPL"].composite_score > 0.0


def test_compute_macro_regime_scale_spy_sma200_defensive_scale() -> None:
    bdays = [ts.date() for ts in pd.bdate_range(start=date(2024, 1, 2), periods=210)]
    as_of = datetime.combine(bdays[-1], datetime.min.time(), tzinfo=NY_TZ).replace(
        hour=16
    ).astimezone(UTC)

    # Low-vol steady downtrend: last_close < sma_200, but 20d ann vol is tiny and close ~= ema_50
    closes_below = np.linspace(500.0, 420.0, 210)
    spy_below = _make_price_df(bdays, closes_below, spread=0.001)
    assert float(spy_below["close"].iloc[-1]) < float(spy_below["close"].tail(200).mean())

    # Without spy_sma200_defensive_scale (None), existing 0.4 floor and 1.0 scale apply
    assert compute_macro_regime_scale(as_of, spy_below, None, spy_sma200_defensive_scale=None) == 1.0
    # With VIX=40 -> 0.45, plus ema/vol drop would hit existing 0.40 floor when None
    vix_high = [MacroPoint(as_of_date=bdays[-3], series_id="VIXCLS", value=40.0)]
    assert (
        compute_macro_regime_scale(as_of, spy_below, vix_high, spy_sma200_defensive_scale=None)
        == 0.45
    )

    # With spy_sma200_defensive_scale=0.30 and last_close < sma_200 -> drops to 0.30
    assert (
        compute_macro_regime_scale(as_of, spy_below, None, spy_sma200_defensive_scale=0.30)
        == 0.30
    )

    # Steady uptrend: last_close >= sma_200 -> defensive scale does NOT trigger
    closes_above = np.linspace(400.0, 500.0, 210)
    spy_above = _make_price_df(bdays, closes_above, spread=0.001)
    assert float(spy_above["close"].iloc[-1]) >= float(spy_above["close"].tail(200).mean())
    assert (
        compute_macro_regime_scale(as_of, spy_above, None, spy_sma200_defensive_scale=0.30)
        == 1.0
    )

    # Fewer than 200 bars -> SMA200 check does NOT trigger
    spy_short = _make_price_df(bdays[-150:], closes_below[-150:], spread=0.001)
    assert (
        compute_macro_regime_scale(as_of, spy_short, None, spy_sma200_defensive_scale=0.30)
        == 1.0
    )


def test_finbert_scorer_rejects_non_finite_probabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    class _FakeTensor:
        def __init__(self, data: list[list[float]]) -> None:
            self._data = data

        def cpu(self) -> _FakeTensor:
            return self

        def tolist(self) -> list[list[float]]:
            return self._data

    class _FakeBatch(dict):
        def to(self, _device: str) -> _FakeBatch:
            return self

    class _FakeNoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    fake_torch = SimpleNamespace(
        no_grad=lambda: _FakeNoGrad(),
        softmax=lambda logits, dim=-1: _FakeTensor([[float("nan"), 0.5, 0.5]]),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    scorer = FinBERTScorer()
    scorer._tokenizer = lambda *args, **kwargs: _FakeBatch()
    scorer._model = lambda **kwargs: SimpleNamespace(logits=[[0.0, 0.0, 0.0]])
    scorer._label_map = {0: "positive", 1: "negative", 2: "neutral"}

    with pytest.raises(ValueError, match="non-finite probability"):
        scorer.score_texts(["AAPL beats estimates"])
