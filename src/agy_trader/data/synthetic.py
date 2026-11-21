from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from agy_trader.core.models import (
    AssetType,
    EarningsEvent,
    MacroPoint,
    NewsArticle,
    UniverseSymbol,
)
from agy_trader.storage.db import StateDB
from agy_trader.storage.parquet_store import ParquetBarStore

NY_TZ = ZoneInfo("America/New_York")

_POSITIVE_HEADLINES = (
    "{sym} beats earnings estimates and raises guidance on strong demand",
    "{sym} surges after upbeat revenue outlook and analyst upgrade",
    "{sym} posts record profit margins and expands share buyback",
    "{sym} outperforms expectations with accelerating growth and bullish guidance",
    "{sym} wins major enterprise contract boosting full-year forecast",
)

_NEGATIVE_HEADLINES = (
    "{sym} downgraded amid weak revenue outlook and margin pressure",
    "{sym} misses quarterly estimates and cuts guidance on slowing demand",
    "{sym} slides after cautious forecast and rising regulatory headwinds",
    "{sym} warns of supply chain delays and lower operating profit",
    "{sym} drops on disappointing subscriber growth and analyst cut",
)

_NEUTRAL_HEADLINES = (
    "{sym} schedules quarterly investor conference and operational update",
    "{sym} announces routine board appointment and annual meeting date",
    "{sym} completes previously disclosed debt refinancing transaction",
)


def _trading_days(end_date: date, days: int) -> list[date]:
    if days < 1:
        raise ValueError("days must be >= 1")
    all_bdays = pd.bdate_range(end=end_date, periods=days)
    return [ts.date() for ts in all_bdays]


def _market_close_utc(trading_day: date) -> datetime:
    close_et = datetime.combine(trading_day, time(16, 0), tzinfo=NY_TZ)
    return close_et.astimezone(UTC)


def _news_timestamp_utc(trading_day: date, rng: np.random.Generator) -> datetime:
    pre_or_intraday_minutes = int(rng.integers(7 * 60 + 30, 15 * 60 + 30))
    hour, minute = divmod(pre_or_intraday_minutes, 60)
    et_dt = datetime.combine(trading_day, time(hour, minute), tzinfo=NY_TZ)
    return et_dt.astimezone(UTC)


def generate_synthetic_dataset(
    universe: list[UniverseSymbol],
    bar_store: ParquetBarStore,
    db: StateDB,
    days: int = 730,
    end_date: date | None = None,
    seed: int = 42,
) -> dict[str, int]:
    """Generate a deterministic synthetic multi-asset dataset and persist it."""
    resolved_end = end_date or date(2025, 12, 31)
    rng = np.random.default_rng(seed)

    dates = _trading_days(resolved_end, days)
    n_days = len(dates)
    timestamps_utc = [_market_close_utc(d) for d in dates]

    # Volatility regime schedule: calm baseline + realistic stress windows
    stress_regime = np.zeros(n_days, dtype=bool)
    if n_days >= 120:
        s1_start = int(n_days * 0.28)
        s1_end = min(n_days, s1_start + max(20, int(n_days * 0.08)))
        stress_regime[s1_start:s1_end] = True
    if n_days >= 240:
        s2_start = int(n_days * 0.68)
        s2_end = min(n_days, s2_start + max(25, int(n_days * 0.10)))
        stress_regime[s2_start:s2_end] = True

    vol_mult = np.where(stress_regime, 1.95, 1.0)

    # Market factor (SPY) with drift, stress-regime drawdown/bounce, and AR(1) mean reversion
    mkt_drift = np.where(stress_regime, -0.0012, 0.00045)
    mkt_shock = rng.normal(0.0, 0.0085, size=n_days) * vol_mult
    mkt_ret = mkt_drift + mkt_shock
    for t in range(1, n_days):
        mkt_ret[t] -= 0.06 * mkt_ret[t - 1]
    mkt_ret = np.clip(mkt_ret, -0.075, 0.075)

    # Sector factors keyed by sector name
    sectors = sorted({item.sector for item in universe})
    sector_factors: dict[str, np.ndarray] = {}
    for sector in sectors:
        if sector == "Fixed Income":
            # Flight-to-safety negative equity correlation during stress
            sec_ret = -0.30 * mkt_ret + rng.normal(0.0001, 0.0055, size=n_days)
        elif sector == "Commodities":
            sec_ret = 0.10 * mkt_ret + rng.normal(0.0002, 0.0075, size=n_days)
        else:
            sec_ret = rng.normal(0.0, 0.0055, size=n_days) * vol_mult
        sector_factors[sector] = sec_ret

    # Macro series aligned with stress regimes
    vix_series = np.where(stress_regime, 29.5, 15.0) + rng.normal(0.0, 1.4, size=n_days)
    for t in range(1, n_days):
        vix_series[t] = 0.85 * vix_series[t - 1] + 0.15 * vix_series[t] - 120.0 * mkt_ret[t]
    vix_series = np.clip(vix_series, 10.5, 65.0)

    hy_oas = np.where(stress_regime, 5.2, 3.3) + rng.normal(0.0, 0.12, size=n_days)
    for t in range(1, n_days):
        hy_oas[t] = 0.90 * hy_oas[t - 1] + 0.10 * hy_oas[t]
    hy_oas = np.clip(hy_oas, 2.4, 9.5)

    t10y2y = 0.35 - 0.55 * stress_regime.astype(float) + np.cumsum(rng.normal(0.0, 0.015, size=n_days))
    t10y2y = np.clip(t10y2y, -1.1, 1.6)

    macro_points: list[MacroPoint] = []
    for i, d in enumerate(dates):
        macro_points.append(MacroPoint(as_of_date=d, series_id="VIXCLS", value=round(float(vix_series[i]), 4)))
        macro_points.append(MacroPoint(as_of_date=d, series_id="BAMLH0A0HYM2", value=round(float(hy_oas[i]), 4)))
        macro_points.append(MacroPoint(as_of_date=d, series_id="T10Y2Y", value=round(float(t10y2y[i]), 4)))

    # Select subset of equities with mild news lead/lag signal
    equity_symbols = [u.symbol for u in universe if u.asset_type == AssetType.equity]
    signal_tickers = set(equity_symbols[: max(1, len(equity_symbols) // 2)])

    earnings_events: list[EarningsEvent] = []
    news_articles: list[NewsArticle] = []
    bars_saved = 0
    article_counter = 0

    for item in universe:
        sym = item.symbol
        is_etf = item.asset_type == AssetType.etf

        # Quarterly earnings dates for equities (~every 63 trading days, deterministic offset)
        earnings_day_indices: dict[int, float] = {}
        if not is_etf:
            offset = int(rng.integers(15, 55))
            for idx_e in range(offset, n_days, 63):
                earnings_events.append(EarningsEvent(symbol=sym, earnings_date=dates[idx_e]))
                surprise = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.018, 0.055))
                earnings_day_indices[idx_e] = surprise

        # Idiosyncratic slow-switching trend regime + mean-reverting noise
        trend_regime = np.zeros(n_days, dtype=float)
        block_len = max(35, n_days // 10)
        for b_start in range(0, n_days, block_len):
            b_end = min(n_days, b_start + block_len)
            trend_regime[b_start:b_end] = float(rng.normal(0.0, 0.00055 if is_etf else 0.00095))

        if sym == "SPY":
            beta = 1.0
            idio_scale = 0.0005
            sec_weight = 0.0
        elif item.sector == "Fixed Income":
            beta = 0.0
            idio_scale = 0.0035
            sec_weight = 1.0
        elif item.sector == "Commodities":
            beta = 0.0
            idio_scale = 0.0045
            sec_weight = 1.0
        elif is_etf:
            beta = float(rng.uniform(0.75, 1.15))
            idio_scale = 0.0030
            sec_weight = 0.70
        else:
            beta = float(rng.uniform(0.80, 1.45))
            idio_scale = float(rng.uniform(0.0075, 0.0140))
            sec_weight = 0.55

        idio_noise = rng.normal(0.0, idio_scale, size=n_days) * vol_mult
        news_lead_alpha = np.zeros(n_days, dtype=float)

        # Schedule discrete news catalysts (earnings + sporadic catalysts)
        scheduled_news: list[tuple[int, int]] = []  # (day_idx, polarity: +1, -1, 0)
        for idx_e, surprise in earnings_day_indices.items():
            polarity = 1 if surprise > 0 else -1
            scheduled_news.append((idx_e, polarity))

        if n_days > 7:
            n_extra_news = max(6, n_days // 45) if not is_etf else max(2, n_days // 150)
            extra_days = rng.choice(
                np.arange(5, n_days - 2),
                size=min(n_extra_news, n_days - 7),
                replace=False,
            )
            for d_idx_raw in extra_days:
                d_idx = int(d_idx_raw)
                if d_idx in earnings_day_indices:
                    continue
                polarity = int(rng.choice([-1, 0, 1], p=[0.35, 0.25, 0.40]))
                scheduled_news.append((d_idx, polarity))
                if polarity != 0 and sym in signal_tickers and rng.random() < 0.70:
                    # Mild realistic lead/lag drift over next 1-3 sessions (~60% directional hit rate via noise)
                    drift_val = polarity * float(rng.uniform(0.004, 0.011))
                    for lag in range(3):
                        if d_idx + lag < n_days:
                            news_lead_alpha[d_idx + lag] += drift_val / (lag + 1.5)

        sec_ret = sector_factors[item.sector]
        raw_ret = beta * mkt_ret + sec_weight * sec_ret + trend_regime + idio_noise + news_lead_alpha

        # Pullback mean-reversion + earnings gaps
        for t in range(1, n_days):
            raw_ret[t] -= (0.04 if is_etf else 0.07) * raw_ret[t - 1]
            if t in earnings_day_indices:
                raw_ret[t] += earnings_day_indices[t]

        raw_ret = np.clip(raw_ret, -0.18, 0.18)

        # Construct strictly valid OHLCV bars
        start_price = float(rng.uniform(35.0, 420.0)) if not is_etf else float(rng.uniform(75.0, 480.0))
        opens = np.empty(n_days, dtype=float)
        highs = np.empty(n_days, dtype=float)
        lows = np.empty(n_days, dtype=float)
        closes = np.empty(n_days, dtype=float)
        vwaps = np.empty(n_days, dtype=float)
        volumes = np.empty(n_days, dtype=float)

        prev_close = start_price
        base_volume = float(rng.uniform(2_500_000.0, 28_000_000.0))

        for t in range(n_days):
            gap_std = 0.0025 if is_etf else 0.0045
            gap = float(rng.normal(0.0, gap_std)) * float(vol_mult[t])
            if t in earnings_day_indices:
                gap += 0.65 * earnings_day_indices[t]
            open_p = max(1.0, prev_close * (1.0 + gap))
            close_p = max(1.0, prev_close * (1.0 + float(raw_ret[t])))

            intra_range = abs(float(rng.normal(0.0, 0.006 if is_etf else 0.011))) * float(vol_mult[t]) + 0.002
            upper_wick = abs(float(rng.normal(0.0, 0.004))) * float(vol_mult[t])
            lower_wick = abs(float(rng.normal(0.0, 0.004))) * float(vol_mult[t])

            high_p = max(open_p, close_p) * (1.0 + upper_wick + 0.35 * intra_range)
            low_p = min(open_p, close_p) / (1.0 + lower_wick + 0.35 * intra_range)
            low_p = max(0.5, min(low_p, open_p, close_p))
            high_p = max(high_p, open_p, close_p)

            vwap_p = (open_p + high_p + low_p + 2.0 * close_p) / 5.0
            vwap_p = min(max(vwap_p, low_p), high_p)

            vol_spike = 1.0 + 3.0 * abs(float(raw_ret[t])) / 0.02 + (1.4 if t in earnings_day_indices else 0.0)
            vol = base_volume * vol_spike * float(rng.uniform(0.75, 1.35))

            opens[t] = round(open_p, 4)
            highs[t] = round(max(high_p, opens[t], round(close_p, 4)), 4)
            closes[t] = round(close_p, 4)
            lows[t] = round(max(0.01, min(low_p, opens[t], closes[t])), 4)
            # Re-verify strict ordering after rounding
            highs[t] = max(highs[t], opens[t], closes[t])
            lows[t] = min(lows[t], opens[t], closes[t])
            vwaps[t] = round(min(max(vwap_p, lows[t]), highs[t]), 4)
            volumes[t] = round(max(1000.0, vol), 0)

            prev_close = closes[t]

        df = pd.DataFrame(
            {
                "timestamp": pd.DatetimeIndex(timestamps_utc, tz="UTC"),
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "vwap": vwaps,
            }
        )
        bars_saved += bar_store.save_bars(sym, df, timeframe="1d")

        # Materialize NewsArticle records for this symbol
        for d_idx, polarity in scheduled_news:
            article_counter += 1
            pub_utc = _news_timestamp_utc(dates[d_idx], rng)
            if polarity > 0:
                tmpl = _POSITIVE_HEADLINES[int(rng.integers(0, len(_POSITIVE_HEADLINES)))]
            elif polarity < 0:
                tmpl = _NEGATIVE_HEADLINES[int(rng.integers(0, len(_NEGATIVE_HEADLINES)))]
            else:
                tmpl = _NEUTRAL_HEADLINES[int(rng.integers(0, len(_NEUTRAL_HEADLINES)))]
            headline = tmpl.format(sym=sym)
            # Ensure unique headline per date across repeated templates
            summary = f"Synthetic market update for {sym} on {dates[d_idx].isoformat()}."
            news_articles.append(
                NewsArticle(
                    article_id=f"syn-{seed}-{article_counter:06d}",
                    published_at=pub_utc,
                    headline=f"{headline} ({dates[d_idx].isoformat()})",
                    summary=summary,
                    source="synthetic",
                    symbols=[sym],
                )
            )

    # Also ensure at least a couple recent macro/market headlines near end_date
    if dates:
        recent_day = dates[-1] - timedelta(days=0)
        article_counter += 1
        news_articles.append(
            NewsArticle(
                article_id=f"syn-{seed}-{article_counter:06d}",
                published_at=_news_timestamp_utc(recent_day, rng),
                headline=f"Market wrap: SPY steady as investors assess outlook ({recent_day.isoformat()})",
                summary="Broad market summary.",
                source="synthetic",
                symbols=["SPY"],
            )
        )

    news_saved = db.upsert_news_articles(news_articles)
    earnings_saved = db.upsert_earnings(earnings_events)
    macro_saved = db.upsert_macro(macro_points)

    return {
        "symbols": len(universe),
        "bars_saved": bars_saved,
        "news_saved": news_saved,
        "earnings_saved": earnings_saved,
        "macro_saved": macro_saved,
    }
