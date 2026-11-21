from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from agy_trader.core.config import StrategyConfig
from agy_trader.core.models import MacroPoint, NewsArticle, SignalRow, ensure_utc
from agy_trader.nlp.sentiment import SentimentAggregator

MIN_HISTORY_BARS = 65
NY_TZ = ZoneInfo("America/New_York")

# Conservative FRED release availability schedules (America/New_York) on next business day(s)
# after observation date T. Guarantees zero intraday or UTC-midnight macro lookahead.
_FRED_RELEASE_SCHEDULE_ET: dict[str, tuple[int, time]] = {
    "VIXCLS": (1, time(11, 30)),
    "BAMLH0A0HYM2": (1, time(11, 30)),
    "T10Y2Y": (1, time(16, 30)),
}
_DEFAULT_FRED_RELEASE_LAG: tuple[int, time] = (2, time(9, 30))
_US_HOLIDAY_CAL = USFederalHolidayCalendar()


@lru_cache(maxsize=32)
def _us_holidays_for_year(year: int) -> frozenset[date]:
    hols = _US_HOLIDAY_CAL.holidays(start=f"{year}-01-01", end=f"{year}-12-31")
    return frozenset(ts.date() for ts in hols)


def _add_business_days(start_date: date, n_days: int) -> date:
    current = start_date
    added = 0
    while added < n_days:
        current += timedelta(days=1)
        if current.weekday() < 5 and current not in _us_holidays_for_year(current.year):
            added += 1
    return current


def macro_point_available_at_utc(point: MacroPoint) -> datetime:
    """Return the conservative UTC timestamp when a FRED MacroPoint becomes publicly available."""
    biz_days, release_time_et = _FRED_RELEASE_SCHEDULE_ET.get(
        point.series_id, _DEFAULT_FRED_RELEASE_LAG
    )
    release_date = _add_business_days(point.as_of_date, biz_days)
    release_dt_et = datetime.combine(release_date, release_time_et, tzinfo=NY_TZ)
    return ensure_utc(release_dt_et)


@dataclass(frozen=True, slots=True)
class _RawSymbolFeatures:
    symbol: str
    trend_raw: float
    pullback_raw: float
    news_raw: float
    news_velocity_z: float
    atr_pct: float
    ann_vol: float
    ret_5d: float


def _is_valid_price_series(series: pd.Series) -> bool:
    arr = series.to_numpy(dtype=float)
    return bool(arr.size > 0 and np.isfinite(arr).all() and (arr > 0.0).all())


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_series = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    val = float(atr_series.iloc[-1])
    if math.isfinite(val) and val > 0.0:
        return val
    fallback = float((high - low).tail(period).mean())
    return fallback if math.isfinite(fallback) and fallback > 0.0 else 0.0


def _wilder_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().iloc[-1]
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().iloc[-1]
    if not (math.isfinite(float(avg_gain)) and math.isfinite(float(avg_loss))):
        return 50.0
    if float(avg_loss) <= 1e-12:
        return 100.0 if float(avg_gain) > 1e-12 else 50.0
    rs = float(avg_gain) / float(avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def _zscore_cross_section(values: Sequence[float], clip_limit: float = 3.0) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        # Degenerate cross-section has zero dispersion; never leak unstandardized raw values.
        return [0.0 for _ in arr]
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if not math.isfinite(std) or std <= 1e-9:
        return [0.0 for _ in arr]
    z = np.clip((arr - mean) / std, -clip_limit, clip_limit)
    return [float(x) for x in z]


def compute_macro_regime_scale(
    as_of: datetime,
    spy_bars: pd.DataFrame | None,
    macro_points: Sequence[MacroPoint] | None = None,
) -> float:
    """Compute strictly point-in-time macro risk scaling in [0.4, 1.0].

    Uses conservative America/New_York business-day release timestamps for FRED macro
    series (`macro_point_available_at_utc(pt) <= as_of_utc`) and blends with SPY 20d
    realized volatility / 50d EMA trend computed strictly on bars with `timestamp <= as_of_utc`.
    """
    as_of_utc = ensure_utc(as_of)
    scale = 1.0

    # 1. SPY market volatility & trend proxy (strictly <= as_of_utc)
    if spy_bars is not None and not spy_bars.empty:
        hist = spy_bars.loc[spy_bars["timestamp"] <= as_of_utc].sort_values("timestamp")
        if len(hist) >= 51:
            closes = hist["close"].astype(float)
            if _is_valid_price_series(closes.tail(51)):
                log_rets = np.log(closes / closes.shift(1)).dropna()
                spy_ann_vol = float(log_rets.tail(20).std(ddof=1) * math.sqrt(252.0))
                ema_50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
                last_close = float(closes.iloc[-1])

                if math.isfinite(spy_ann_vol):
                    if spy_ann_vol >= 0.35:
                        scale = min(scale, 0.50)
                    elif spy_ann_vol >= 0.25:
                        scale = min(scale, 0.70)
                    elif spy_ann_vol >= 0.20:
                        scale = min(scale, 0.85)
                if math.isfinite(ema_50) and last_close < ema_50 * 0.97:
                    scale = min(scale, scale * 0.85)

    # 2. FRED macro series gated by exact conservative release availability UTC timestamp
    if macro_points:
        latest_by_series: dict[str, MacroPoint] = {}
        for pt in macro_points:
            if not math.isfinite(pt.value):
                continue
            if macro_point_available_at_utc(pt) > as_of_utc:
                continue
            prev = latest_by_series.get(pt.series_id)
            if prev is None or pt.as_of_date > prev.as_of_date:
                latest_by_series[pt.series_id] = pt

        vix = latest_by_series.get("VIXCLS")
        if vix is not None:
            if vix.value >= 35.0:
                scale = min(scale, 0.45)
            elif vix.value >= 25.0:
                scale = min(scale, 0.70)

        hy_spread = latest_by_series.get("BAMLH0A0HYM2")
        if hy_spread is not None and hy_spread.value >= 6.0:
            scale = min(scale, 0.55)

        yield_curve = latest_by_series.get("T10Y2Y")
        if yield_curve is not None and yield_curve.value < -0.5 and scale < 0.85:
            scale = min(scale, 0.50)

    return round(max(0.4, min(1.0, scale)), 4)


class FeatureEngine:
    """Strictly point-in-time quantitative feature and signal builder."""

    def __init__(self, strategy_config: StrategyConfig) -> None:
        self.config = strategy_config
        self.sentiment_agg = SentimentAggregator(
            half_life_hours=strategy_config.news_half_life_hours,
            lookback_hours=strategy_config.news_lookback_hours,
        )

    def build_signals(
        self,
        as_of: datetime,
        bars_by_symbol: Mapping[str, pd.DataFrame],
        news_by_symbol: Mapping[str, Sequence[NewsArticle]] | None = None,
        macro_points: Sequence[MacroPoint] | None = None,
        benchmark_symbol: str = "SPY",
    ) -> list[SignalRow]:
        """Compute cross-sectional SignalRows strictly using data <= as_of (UTC)."""
        as_of_utc = ensure_utc(as_of)
        news_map = news_by_symbol or {}

        # Benchmark 20d return strictly <= as_of_utc (safe fallback on corrupt benchmark bars)
        spy_df = bars_by_symbol.get(benchmark_symbol)
        spy_ret_20d = 0.0
        if spy_df is not None and not spy_df.empty:
            spy_hist = spy_df.loc[spy_df["timestamp"] <= as_of_utc].sort_values("timestamp")
            if len(spy_hist) >= 21:
                spy_tail = spy_hist["close"].astype(float).tail(21)
                if _is_valid_price_series(spy_tail):
                    candidate_spy_ret = float(spy_tail.iloc[-1] / spy_tail.iloc[0] - 1.0)
                    if math.isfinite(candidate_spy_ret):
                        spy_ret_20d = candidate_spy_ret

        macro_scale = compute_macro_regime_scale(
            as_of=as_of_utc,
            spy_bars=spy_df,
            macro_points=macro_points,
        )

        raw_rows: list[_RawSymbolFeatures] = []
        for symbol in sorted(bars_by_symbol.keys()):
            df = bars_by_symbol[symbol]
            if df.empty:
                continue
            hist = df.loc[df["timestamp"] <= as_of_utc].sort_values("timestamp")
            if len(hist) < MIN_HISTORY_BARS:
                continue

            closes = hist["close"].astype(float).reset_index(drop=True)
            highs = hist["high"].astype(float).reset_index(drop=True)
            lows = hist["low"].astype(float).reset_index(drop=True)

            # Reject symbol if any required lookback price is non-finite or non-positive
            if not (
                _is_valid_price_series(closes.tail(MIN_HISTORY_BARS))
                and _is_valid_price_series(highs.tail(MIN_HISTORY_BARS))
                and _is_valid_price_series(lows.tail(MIN_HISTORY_BARS))
            ):
                continue

            last_close = float(closes.iloc[-1])
            ret_5d = float(last_close / float(closes.iloc[-6]) - 1.0)
            ret_20d = float(last_close / float(closes.iloc[-21]) - 1.0)
            ret_60d = float(last_close / float(closes.iloc[-61]) - 1.0)

            log_rets = np.log(closes / closes.shift(1)).dropna()
            daily_vol_20 = float(log_rets.tail(20).std(ddof=1))
            if not math.isfinite(daily_vol_20) or daily_vol_20 < 0.0:
                daily_vol_20 = 0.01
            ann_vol = daily_vol_20 * math.sqrt(252.0)

            vol_20_period = max(daily_vol_20 * math.sqrt(20.0), 0.015)
            vol_60_period = max(daily_vol_20 * math.sqrt(60.0), 0.025)

            atr_14 = _wilder_atr(highs, lows, closes, period=14)
            atr_pct = max(atr_14 / last_close, 0.002)

            ema_20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
            ema_50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
            ema_spread = (ema_20 / max(ema_50, 1e-8) - 1.0) / max(atr_pct, 0.005)

            rel_strength_20 = (ret_20d - spy_ret_20d) / vol_20_period
            trend_raw = (
                0.40 * (ret_20d / vol_20_period)
                + 0.30 * (ret_60d / vol_60_period)
                + 0.15 * ema_spread
                + 0.15 * rel_strength_20
            )

            sma_20 = float(closes.tail(20).mean())
            std_20 = float(closes.tail(20).std(ddof=1))
            z_20 = (last_close - sma_20) / max(std_20, last_close * 0.005)
            rsi_14 = _wilder_rsi(closes, period=14)
            # Mild anti-chase / pullback preference: positive when oversold, negative when overbought
            pullback_raw = -0.5 * max(-3.0, min(3.0, z_20)) - 0.5 * ((rsi_14 - 50.0) / 25.0)

            sym_news = news_map.get(symbol, ())
            agg_sent = self.sentiment_agg.aggregate_for_symbol(sym_news, as_of=as_of_utc)

            if not all(
                math.isfinite(x)
                for x in (
                    trend_raw,
                    pullback_raw,
                    agg_sent.news_score,
                    agg_sent.velocity_z,
                    atr_pct,
                    ann_vol,
                    ret_5d,
                )
            ):
                continue

            raw_rows.append(
                _RawSymbolFeatures(
                    symbol=symbol,
                    trend_raw=trend_raw,
                    pullback_raw=pullback_raw,
                    news_raw=agg_sent.news_score,
                    news_velocity_z=agg_sent.velocity_z,
                    atr_pct=atr_pct,
                    ann_vol=ann_vol,
                    ret_5d=ret_5d,
                )
            )

        if not raw_rows:
            return []

        trend_zs = _zscore_cross_section([r.trend_raw for r in raw_rows], clip_limit=3.0)
        pullback_zs = _zscore_cross_section([r.pullback_raw for r in raw_rows], clip_limit=3.0)
        # Keep news centered at 0.0 (no news == neutral 0.0) scaled into comparable z-score units
        news_zs = [max(-3.0, min(3.0, r.news_raw * 1.8)) for r in raw_rows]

        w_t = self.config.w_trend
        w_p = self.config.w_pullback
        w_n = self.config.w_news_sentiment
        w_sum = w_t + w_p + w_n

        signals: list[SignalRow] = []
        for idx, r in enumerate(raw_rows):
            t_z = trend_zs[idx]
            p_z = pullback_zs[idx]
            n_z = news_zs[idx]
            composite = (w_t * t_z + w_p * p_z + w_n * n_z) / w_sum
            signals.append(
                SignalRow(
                    as_of=as_of_utc,
                    symbol=r.symbol,
                    trend_score=round(t_z, 6),
                    pullback_score=round(p_z, 6),
                    news_score=round(n_z, 6),
                    news_velocity_z=round(r.news_velocity_z, 6),
                    composite_score=round(composite, 6),
                    macro_regime_scale=macro_scale,
                    atr_pct=round(r.atr_pct, 6),
                    ann_vol=round(r.ann_vol, 6),
                    ret_5d=round(r.ret_5d, 6),
                )
            )
        return signals
