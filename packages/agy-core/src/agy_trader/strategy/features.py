from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from agy_trader.core.calendar import nyse_trading_days_apart
from agy_trader.core.config import StrategyConfig
from agy_trader.core.models import (
    MacroPoint,
    NewsArticle,
    SecFilingEvent,
    SignalRow,
    ensure_utc,
)
from agy_trader.data.news_filters import is_forward_only_source
from agy_trader.nlp.sentiment import SentimentAggregator

MIN_HISTORY_BARS = 65
NY_TZ = ZoneInfo("America/New_York")
SAFE_HAVEN_SECTORS = frozenset({"Commodities", "Fixed Income"})
SAFE_HAVEN_SYMBOLS = frozenset({"GLD", "TLT"})

# Conservative FRED release availability schedules (America/New_York) on next business day(s)
# after observation date T. Guarantees zero intraday or UTC-midnight macro lookahead.
_FRED_RELEASE_SCHEDULE_ET: dict[str, tuple[int, time]] = {
    "VIXCLS": (1, time(11, 30)),
    "BAMLH0A0HYM2": (1, time(11, 30)),
    "T10Y2Y": (1, time(16, 30)),
    "DGS3MO": (1, time(16, 30)),
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


def latest_pit_rf_apr(
    as_of_utc: datetime,
    macro_points: Sequence[MacroPoint] | None,
    fallback_apr: float = 0.045,
) -> float:
    """Return strictly point-in-time annualized 3-month T-bill yield (DGS3MO) in [0.0, 0.25]."""
    as_of = ensure_utc(as_of_utc)
    clamped_fallback = (
        max(0.0, min(0.25, float(fallback_apr))) if math.isfinite(fallback_apr) else 0.0
    )
    if not macro_points:
        return clamped_fallback

    latest: MacroPoint | None = None
    for pt in macro_points:
        if pt.series_id != "DGS3MO" or not math.isfinite(pt.value):
            continue
        if macro_point_available_at_utc(pt) > as_of:
            continue
        if latest is None or pt.as_of_date > latest.as_of_date:
            latest = pt

    if latest is None:
        return clamped_fallback
    return max(0.0, min(0.25, float(latest.value) / 100.0))


def _filter_news_by_sources(
    articles: Sequence[NewsArticle],
    allowed_sources: frozenset[str] | None,
) -> Sequence[NewsArticle]:
    if allowed_sources is None:
        return [art for art in articles if not is_forward_only_source(art.source)]
    return [
        art
        for art in articles
        if not is_forward_only_source(art.source)
        and art.source.strip().lower() in allowed_sources
    ]


def _compute_sec_features_for_symbol(
    events: Sequence[SecFilingEvent],
    as_of_utc: datetime,
    insider_lookback_days: int,
    restatement_blackout_days: int,
) -> tuple[float, bool, bool]:
    """Compute strictly PIT Form 4 insider score and 8-K Item 4.02 restatement flag."""
    if not events:
        return 0.0, False, False

    lookback_start = as_of_utc - timedelta(days=insider_lookback_days)
    as_of_et_date = as_of_utc.astimezone(NY_TZ).date()

    net_usd = 0.0
    buyer_count = 0
    seller_count = 0
    has_recent_restatement = False

    for ev in events:
        accepted_utc = ensure_utc(ev.accepted_at)
        if accepted_utc > as_of_utc:
            continue
        if ev.form_type == "4":
            if accepted_utc >= lookback_start:
                net_usd += float(ev.net_insider_buy_usd)
                buyer_count += int(ev.insider_buyer_count)
                seller_count += int(ev.insider_seller_count)
        elif ev.form_type == "8-K" and restatement_blackout_days > 0 and "4.02" in ev.items:
            ev_et_date = accepted_utc.astimezone(NY_TZ).date()
            if nyse_trading_days_apart(as_of_et_date, ev_et_date) <= restatement_blackout_days:
                has_recent_restatement = True

    has_insider_activity = (buyer_count > 0) or (seller_count > 0) or (abs(net_usd) > 0.0)
    if has_insider_activity:
        raw_insider = math.tanh(net_usd / 500_000.0) * (1.0 + 0.25 * min(buyer_count, 4))
        clamped_raw = max(-1.5, min(1.5, raw_insider))
        insider_score = max(-3.0, min(3.0, clamped_raw * 1.8))
    else:
        insider_score = 0.0

    return insider_score, has_insider_activity, has_recent_restatement


@dataclass(frozen=True, slots=True)
class _RawSymbolFeatures:
    symbol: str
    trend_raw: float
    pullback_raw: float
    news_raw: float
    news_velocity_z: float
    news_article_count: int
    insider_score: float
    has_insider_activity: bool
    has_recent_restatement: bool
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
    spy_sma200_defensive_scale: float | None = None,
    *,
    spy_ema50_dual_scale: float | None = None,
    spy_vol_surge_scale: float | None = None,
    spy_vol_surge_thresh: float = 1.65,
    spy_ret10_crash_thresh: float = -0.055,
    spy_ret10_crash_scale: float | None = None,
    regime_mode: Literal[
        "legacy_stacked", "daniel_moskowitz", "none", "sma200_only"
    ] = "legacy_stacked",
) -> float:
    """Compute strictly point-in-time macro risk scaling in [floor, 1.0].

    Uses conservative America/New_York business-day release timestamps for FRED macro
    series (`macro_point_available_at_utc(pt) <= as_of_utc`) and blends with SPY 20d
    realized volatility / 50d EMA / optional 200d SMA trend and optional fast equity
    crash brakes computed strictly on bars with `timestamp <= as_of_utc`.
    """
    as_of_utc = ensure_utc(as_of)

    if regime_mode == "none":
        return 1.0

    if regime_mode == "sma200_only":
        def_scale = (
            spy_sma200_defensive_scale
            if spy_sma200_defensive_scale is not None
            else 0.50
        )
        if spy_bars is not None and not spy_bars.empty:
            spy_hist_200 = spy_bars.loc[spy_bars["timestamp"] <= as_of_utc].sort_values(
                "timestamp"
            )
            if len(spy_hist_200) >= 200:
                closes_200 = spy_hist_200["close"].astype(float).tail(200)
                if _is_valid_price_series(closes_200):
                    sma_200 = float(closes_200.mean())
                    last_c_200 = float(closes_200.iloc[-1])
                    if math.isfinite(sma_200) and last_c_200 < sma_200:
                        return round(max(0.0, min(1.0, float(def_scale))), 4)
        return 1.0

    if regime_mode == "daniel_moskowitz":
        spy_ann_vol_20d = 0.15
        bear_state = False
        last_c = 0.0
        ema_50 = 0.0
        if spy_bars is not None and not spy_bars.empty:
            spy_hist_dm = spy_bars.loc[spy_bars["timestamp"] <= as_of_utc].sort_values("timestamp")
            if len(spy_hist_dm) >= 51:
                closes_dm = spy_hist_dm["close"].astype(float)
                if _is_valid_price_series(closes_dm.tail(51)):
                    log_rets_dm = np.log(closes_dm / closes_dm.shift(1)).dropna()
                    cand_vol = float(log_rets_dm.tail(20).std(ddof=1) * math.sqrt(252.0))
                    if math.isfinite(cand_vol) and cand_vol > 0.0:
                        spy_ann_vol_20d = cand_vol
                    ema_50 = float(closes_dm.ewm(span=50, adjust=False).mean().iloc[-1])
                    last_c = float(closes_dm.iloc[-1])
                    if len(spy_hist_dm) >= 200:
                        closes_200 = closes_dm.tail(200)
                        if _is_valid_price_series(closes_200):
                            sma_200 = float(closes_200.mean())
                            if math.isfinite(sma_200) and last_c < sma_200:
                                bear_state = True
                    if len(spy_hist_dm) >= 253:
                        closes_253 = closes_dm.tail(253)
                        if _is_valid_price_series(closes_253):
                            c_252 = float(closes_253.iloc[0])
                            if math.isfinite(c_252) and last_c < c_252:
                                bear_state = True

        vix_dm: MacroPoint | None = None
        hy_dm: MacroPoint | None = None
        if macro_points:
            latest_dm: dict[str, MacroPoint] = {}
            for pt in macro_points:
                if not math.isfinite(pt.value):
                    continue
                if macro_point_available_at_utc(pt) > as_of_utc:
                    continue
                prev = latest_dm.get(pt.series_id)
                if prev is None or pt.as_of_date > prev.as_of_date:
                    latest_dm[pt.series_id] = pt
            vix_dm = latest_dm.get("VIXCLS")
            hy_dm = latest_dm.get("BAMLH0A0HYM2")

        if vix_dm is not None and vix_dm.value >= 28.0:
            bear_state = True

        if bear_state and (
            spy_ann_vol_20d >= 0.22 or (vix_dm is not None and vix_dm.value >= 25.0)
        ):
            dm_scale = min(1.0, 0.20 / max(spy_ann_vol_20d, 0.10))
            if hy_dm is not None and hy_dm.value >= 6.0:
                dm_scale = min(dm_scale, 0.55)
            if vix_dm is not None and vix_dm.value >= 35.0:
                dm_scale = min(dm_scale, 0.45)
            dm_scale = max(0.40, dm_scale)
        else:
            if (
                spy_ann_vol_20d >= 0.30
                and last_c > 0.0
                and math.isfinite(ema_50)
                and last_c < ema_50
            ):
                dm_scale = 0.85
            else:
                dm_scale = 1.0

        return round(max(0.40, min(1.0, dm_scale)), 4)

    scale = 1.0

    # 1. SPY market volatility & trend proxy (strictly <= as_of_utc)
    spy_hist: pd.DataFrame | None = None
    if spy_bars is not None and not spy_bars.empty:
        spy_hist = spy_bars.loc[spy_bars["timestamp"] <= as_of_utc].sort_values("timestamp")
        if len(spy_hist) >= 51:
            closes = spy_hist["close"].astype(float)
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
                if spy_sma200_defensive_scale is not None and len(spy_hist) >= 200:
                    closes_200 = spy_hist["close"].astype(float).tail(200)
                    if _is_valid_price_series(closes_200):
                        sma_200 = float(closes_200.mean())
                        if math.isfinite(sma_200) and last_close < sma_200:
                            scale = min(scale, spy_sma200_defensive_scale)

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

    floor = 0.4 if spy_sma200_defensive_scale is None else min(0.4, spy_sma200_defensive_scale)
    base_scale = round(max(floor, min(1.0, scale)), 4)

    has_fast_checks = (
        spy_ema50_dual_scale is not None
        or spy_vol_surge_scale is not None
        or spy_ret10_crash_scale is not None
    )
    if not has_fast_checks:
        return base_scale

    fast_scale = base_scale
    if spy_hist is not None and len(spy_hist) >= MIN_HISTORY_BARS:
        closes = spy_hist["close"].astype(float)
        if _is_valid_price_series(closes.tail(MIN_HISTORY_BARS)):
            last_c = float(closes.iloc[-1])
            ema20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
            lr = np.log(closes / closes.shift(1)).dropna()
            vol_10 = float(lr.tail(10).std(ddof=1) * math.sqrt(252.0)) if len(lr) >= 10 else 0.15
            vol_20 = float(lr.tail(20).std(ddof=1) * math.sqrt(252.0)) if len(lr) >= 20 else 0.15
            vol_60 = float(lr.tail(60).std(ddof=1) * math.sqrt(252.0)) if len(lr) >= 60 else vol_20

            # Fast dual-EMA breakdown (before SMA200 reacts)
            if spy_ema50_dual_scale is not None and last_c < ema50 and ema20 < ema50:
                fast_scale = min(fast_scale, spy_ema50_dual_scale)

            # Fast 10d realized volatility surge vs 60d baseline when SPY < EMA20
            surge_ratio = vol_10 / max(vol_60, 0.08)
            if spy_vol_surge_scale is not None and last_c < ema20:
                if vol_10 >= 0.30 or surge_ratio >= spy_vol_surge_thresh:
                    fast_scale = min(fast_scale, spy_vol_surge_scale)
                elif vol_10 >= 0.22 or surge_ratio >= (spy_vol_surge_thresh * 0.85):
                    fast_scale = min(fast_scale, max(spy_vol_surge_scale, 0.70))

            # Fast 10d price drop crash brake
            if spy_ret10_crash_scale is not None and len(closes) >= 11:
                ret_10 = last_c / float(closes.iloc[-11]) - 1.0
                if ret_10 <= spy_ret10_crash_thresh:
                    fast_scale = min(fast_scale, spy_ret10_crash_scale)

    fast_floor = min(floor, 0.20)
    return round(max(fast_floor, min(1.0, fast_scale)), 4)


def _fast_wilder_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> float:
    prev_close = np.empty_like(closes)
    prev_close[0] = closes[0]
    prev_close[1:] = closes[:-1]
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)),
    )
    tr[0] = highs[0] - lows[0]
    alpha = 1.0 / period
    val = float(tr[0])
    one_m = 1.0 - alpha
    for x in tr[1:]:
        val = alpha * float(x) + one_m * val
    if math.isfinite(val) and val > 0.0:
        return val
    fallback = float(np.mean(highs[-period:] - lows[-period:]))
    return fallback if math.isfinite(fallback) and fallback > 0.0 else 0.0


def _fast_ema_last(vals: np.ndarray, span: int) -> float:
    alpha = 2.0 / (span + 1.0)
    one_m = 1.0 - alpha
    val = float(vals[0])
    for x in vals[1:]:
        val = alpha * float(x) + one_m * val
    return val


def _fast_rsi_last(closes: np.ndarray, period: int = 5) -> float:
    diffs = np.diff(closes)
    alpha = 1.0 / period
    one_m = 1.0 - alpha
    ag = max(float(diffs[0]), 0.0)
    al = max(float(-diffs[0]), 0.0)
    for d in diffs[1:]:
        g = float(d) if d > 0.0 else 0.0
        loss = float(-d) if d < 0.0 else 0.0
        ag = alpha * g + one_m * ag
        al = alpha * loss + one_m * al
    if not (math.isfinite(ag) and math.isfinite(al)):
        return 50.0
    if al <= 1e-12:
        return 100.0 if ag > 1e-12 else 50.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def _seg_sharpe(lr_vals: np.ndarray, tot: int, sk: int, fallback: float) -> float:
    n_lr = len(lr_vals)
    eff_sk = sk if n_lr >= (60 + sk) else 0
    seg = lr_vals[max(0, n_lr - tot) : n_lr - eff_sk]
    if len(seg) >= 20:
        return (
            float(np.mean(seg)) / max(float(np.std(seg, ddof=1)), 0.004)
        ) * math.sqrt(252.0)
    return fallback


def _sorted_bars_arrays(
    df: pd.DataFrame,
    as_of_ns: np.int64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if df.empty:
        return None
    parsed = pd.to_datetime(df["timestamp"], utc=True)
    if parsed.isna().any():
        valid_mask = parsed.notna().to_numpy()
        if not np.any(valid_mask):
            return None
        df = df.iloc[valid_mask]
        parsed = parsed.iloc[valid_mask]
    ts_all = parsed.dt.as_unit("ns").astype("int64").to_numpy()
    if ts_all.size == 0:
        return None
    has_vol = "volume" in df.columns
    if not bool(np.all(ts_all[:-1] <= ts_all[1:])):
        order = np.argsort(ts_all, kind="mergesort")
        ts_all = ts_all[order]
        closes_all = df["close"].to_numpy(dtype=np.float64)[order]
        highs_all = df["high"].to_numpy(dtype=np.float64)[order]
        lows_all = df["low"].to_numpy(dtype=np.float64)[order]
        volumes_all = (
            df["volume"].to_numpy(dtype=np.float64)[order]
            if has_vol
            else np.zeros_like(closes_all)
        )
    else:
        closes_all = df["close"].to_numpy(dtype=np.float64)
        highs_all = df["high"].to_numpy(dtype=np.float64)
        lows_all = df["low"].to_numpy(dtype=np.float64)
        volumes_all = (
            df["volume"].to_numpy(dtype=np.float64)
            if has_vol
            else np.zeros_like(closes_all)
        )
    if ts_all[-1] <= as_of_ns:
        return ts_all, closes_all, highs_all, lows_all, volumes_all
    mask = ts_all <= as_of_ns
    if not np.any(mask):
        return None
    return (
        ts_all[mask],
        closes_all[mask],
        highs_all[mask],
        lows_all[mask],
        volumes_all[mask],
    )


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
        fallback_rf_apr: float = 0.045,
        sec_filings_by_symbol: Mapping[str, Sequence[SecFilingEvent]] | None = None,
        restatement_blackout_days: int = 10,
        *,
        universe_symbols: set[str] | frozenset[str] | None = None,
        symbol_sectors: Mapping[str, str] | None = None,
        sector_etf_map: Mapping[str, str] | None = None,
    ) -> list[SignalRow]:
        """Compute cross-sectional SignalRows strictly using data <= as_of (UTC)."""
        as_of_utc = ensure_utc(as_of)
        norm_universe: frozenset[str] | None = (
            frozenset(s.strip().upper() for s in universe_symbols if s.strip())
            if universe_symbols is not None
            else None
        )
        if self.config.momentum_mode == "m2_residual":
            return self._build_signals_m2_residual(
                as_of_utc=as_of_utc,
                bars_by_symbol=bars_by_symbol,
                news_by_symbol=news_by_symbol,
                macro_points=macro_points,
                benchmark_symbol=benchmark_symbol,
                fallback_rf_apr=fallback_rf_apr,
                sec_filings_by_symbol=sec_filings_by_symbol,
                restatement_blackout_days=restatement_blackout_days,
                universe_symbols=norm_universe,
                symbol_sectors=symbol_sectors,
                sector_etf_map=sector_etf_map,
            )

        news_map = news_by_symbol or {}
        sec_map = sec_filings_by_symbol or {}
        allowed_sources: frozenset[str] | None = (
            frozenset(s.strip().lower() for s in self.config.model_news_sources if s.strip())
            if self.config.model_news_sources is not None
            else None
        )

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
            spy_sma200_defensive_scale=self.config.spy_sma200_defensive_scale,
            spy_ema50_dual_scale=self.config.spy_ema50_dual_scale,
            spy_vol_surge_scale=self.config.spy_vol_surge_scale,
            spy_vol_surge_thresh=self.config.spy_vol_surge_thresh,
            spy_ret10_crash_thresh=self.config.spy_ret10_crash_thresh,
            spy_ret10_crash_scale=self.config.spy_ret10_crash_scale,
            regime_mode=self.config.regime_mode,
        )

        raw_rows: list[_RawSymbolFeatures] = []
        for symbol in sorted(bars_by_symbol.keys()):
            if norm_universe is not None and symbol.strip().upper() not in norm_universe:
                continue
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

            sym_news = _filter_news_by_sources(news_map.get(symbol, ()), allowed_sources)
            agg_sent = self.sentiment_agg.aggregate_for_symbol(sym_news, as_of=as_of_utc)
            insider_score, has_insider_act, has_restatement = _compute_sec_features_for_symbol(
                sec_map.get(symbol, ()),
                as_of_utc=as_of_utc,
                insider_lookback_days=self.config.insider_lookback_days,
                restatement_blackout_days=restatement_blackout_days,
            )

            if not all(
                math.isfinite(x)
                for x in (
                    trend_raw,
                    pullback_raw,
                    agg_sent.news_score,
                    agg_sent.velocity_z,
                    insider_score,
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
                    news_article_count=agg_sent.article_count_window,
                    insider_score=insider_score,
                    has_insider_activity=has_insider_act,
                    has_recent_restatement=has_restatement,
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
        w_i = self.config.w_insider

        signals: list[SignalRow] = []
        for idx, r in enumerate(raw_rows):
            t_z = trend_zs[idx]
            p_z = pullback_zs[idx]
            n_z = news_zs[idx]
            i_z = r.insider_score
            effective_w_n = w_n if r.news_article_count > 0 else 0.0
            effective_w_i = w_i if (w_i > 0.0 and r.has_insider_activity) else 0.0
            row_w_sum = w_t + w_p + effective_w_n + effective_w_i
            composite = (
                (w_t * t_z + w_p * p_z + effective_w_n * n_z + effective_w_i * i_z) / row_w_sum
                if row_w_sum > 0.0
                else 0.0
            )
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
                    abs_trend_raw=round(r.trend_raw, 6),
                    insider_score=round(i_z, 6),
                    has_recent_restatement=r.has_recent_restatement,
                )
            )
        return signals

    def _build_signals_m2_residual(
        self,
        *,
        as_of_utc: datetime,
        bars_by_symbol: Mapping[str, pd.DataFrame],
        news_by_symbol: Mapping[str, Sequence[NewsArticle]] | None,
        macro_points: Sequence[MacroPoint] | None,
        benchmark_symbol: str,
        fallback_rf_apr: float = 0.045,
        sec_filings_by_symbol: Mapping[str, Sequence[SecFilingEvent]] | None = None,
        restatement_blackout_days: int = 10,
        universe_symbols: set[str] | frozenset[str] | None = None,
        symbol_sectors: Mapping[str, str] | None = None,
        sector_etf_map: Mapping[str, str] | None = None,
    ) -> list[SignalRow]:
        """Compute multi-horizon OPUS-M2 + CAPM residual Sharpe + 60d peer correlations."""
        as_of_ns = np.int64(pd.Timestamp(as_of_utc).value)
        norm_universe: frozenset[str] | None = (
            frozenset(s.strip().upper() for s in universe_symbols if s.strip())
            if universe_symbols is not None
            else None
        )
        news_map = news_by_symbol or {}
        sec_map = sec_filings_by_symbol or {}
        allowed_sources: frozenset[str] | None = (
            frozenset(s.strip().lower() for s in self.config.model_news_sources if s.strip())
            if self.config.model_news_sources is not None
            else None
        )
        rf_apr = latest_pit_rf_apr(as_of_utc, macro_points, fallback_apr=fallback_rf_apr)
        rf_daily = rf_apr / 252.0

        spy_df = bars_by_symbol.get(benchmark_symbol)
        spy_lr: np.ndarray | None = None
        spy_lr_ts: np.ndarray | None = None
        if spy_df is not None and not spy_df.empty:
            spy_arrs = _sorted_bars_arrays(spy_df, as_of_ns)
            if spy_arrs is not None:
                sts, sc, _, _, _ = spy_arrs
                if len(sc) >= MIN_HISTORY_BARS and np.all(
                    np.isfinite(sc[-MIN_HISTORY_BARS:]) & (sc[-MIN_HISTORY_BARS:] > 0.0)
                ):
                    raw_spy_lr = np.log(sc[1:] / sc[:-1])
                    raw_spy_ts = sts[1:]
                    valid_spy = np.isfinite(raw_spy_lr)
                    if not np.all(valid_spy):
                        raw_spy_lr = raw_spy_lr[valid_spy]
                        raw_spy_ts = raw_spy_ts[valid_spy]
                    spy_lr = raw_spy_lr
                    spy_lr_ts = raw_spy_ts

        # Precompute sector ETF (ts_vals, lr_vals) arrays once per rebalance call
        sector_etf_lr_by_sym: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if self.config.use_sector_residual and sector_etf_map:
            for raw_etf_sym in set(sector_etf_map.values()):
                etf_sym = raw_etf_sym.strip().upper()
                if not etf_sym or etf_sym in sector_etf_lr_by_sym:
                    continue
                etf_df = bars_by_symbol.get(etf_sym)
                if etf_df is None:
                    etf_df = bars_by_symbol.get(raw_etf_sym)
                if etf_df is None or etf_df.empty:
                    continue
                etf_arrs = _sorted_bars_arrays(etf_df, as_of_ns)
                if etf_arrs is None:
                    continue
                ets, ec, _, _, _ = etf_arrs
                if len(ec) >= MIN_HISTORY_BARS and np.all(
                    np.isfinite(ec[-MIN_HISTORY_BARS:]) & (ec[-MIN_HISTORY_BARS:] > 0.0)
                ):
                    raw_etf_lr = np.log(ec[1:] / ec[:-1])
                    raw_etf_ts = ets[1:]
                    valid_etf = np.isfinite(raw_etf_lr)
                    if not np.all(valid_etf):
                        raw_etf_lr = raw_etf_lr[valid_etf]
                        raw_etf_ts = raw_etf_ts[valid_etf]
                    sector_etf_lr_by_sym[etf_sym] = (raw_etf_ts, raw_etf_lr)

        macro_scale = compute_macro_regime_scale(
            as_of=as_of_utc,
            spy_bars=spy_df,
            macro_points=macro_points,
            spy_sma200_defensive_scale=self.config.spy_sma200_defensive_scale,
            spy_ema50_dual_scale=self.config.spy_ema50_dual_scale,
            spy_vol_surge_scale=self.config.spy_vol_surge_scale,
            spy_vol_surge_thresh=self.config.spy_vol_surge_thresh,
            spy_ret10_crash_thresh=self.config.spy_ret10_crash_thresh,
            spy_ret10_crash_scale=self.config.spy_ret10_crash_scale,
            regime_mode=self.config.regime_mode,
        )

        raw_items: list[dict[str, float | int | bool | str]] = []
        lr_ts_list: list[tuple[str, np.ndarray, np.ndarray]] = []
        long_lb = self.config.m2_long_lookback
        long_skip = self.config.m2_long_skip
        med_lb = self.config.m2_med_lookback
        med_skip = self.config.m2_med_skip

        for symbol in sorted(bars_by_symbol.keys()):
            norm_sym = symbol.strip().upper()
            if norm_universe is not None and norm_sym not in norm_universe:
                continue
            arrs = _sorted_bars_arrays(bars_by_symbol[symbol], as_of_ns)
            if arrs is None:
                continue
            ts_vals, closes, highs, lows, volumes = arrs
            if len(closes) < MIN_HISTORY_BARS:
                continue
            if not (
                np.all(np.isfinite(closes[-MIN_HISTORY_BARS:]) & (closes[-MIN_HISTORY_BARS:] > 0.0))
                and np.all(np.isfinite(highs[-MIN_HISTORY_BARS:]) & (highs[-MIN_HISTORY_BARS:] > 0.0))
                and np.all(np.isfinite(lows[-MIN_HISTORY_BARS:]) & (lows[-MIN_HISTORY_BARS:] > 0.0))
            ):
                continue

            bad_close_idx = np.flatnonzero(~(np.isfinite(closes) & (closes > 0.0)))
            clean_closes = (
                closes[int(bad_close_idx[-1]) + 1 :] if bad_close_idx.size > 0 else closes
            )

            last_close = float(closes[-1])
            ret_5d = float(last_close / float(closes[-6]) - 1.0)

            lr_vals = np.log(closes[1:] / closes[:-1])
            lr_ts = ts_vals[1:]
            valid_lr = np.isfinite(lr_vals)
            if not np.all(valid_lr):
                lr_vals = lr_vals[valid_lr]
                lr_ts = lr_ts[valid_lr]
            n_lr = len(lr_vals)
            if n_lr < 55:
                continue

            daily_vol_20 = max(float(np.std(lr_vals[-20:], ddof=1)), 0.004)
            ann_vol = daily_vol_20 * math.sqrt(252.0)
            atr_pct = max(_fast_wilder_atr(highs, lows, closes, 14) / last_close, 0.002)

            w120 = min(120, n_lr)
            lr120 = lr_vals[-w120:]
            m120 = float(np.mean(lr120))
            s120 = max(float(np.std(lr120, ddof=1)), 0.004)
            shp_120 = (m120 / s120) * math.sqrt(252.0)
            abs_shp_120 = (
                ((m120 - rf_daily) / s120) * math.sqrt(252.0)
                if self.config.use_rf_hurdle_for_abs_momentum
                else shp_120
            )

            shp_252_s21 = _seg_sharpe(lr_vals, long_lb, long_skip, shp_120)
            shp_126_s10 = _seg_sharpe(lr_vals, med_lb, med_skip, shp_120)

            eff_sk_long = long_skip if n_lr >= (60 + long_skip) else 0
            seg_long = lr_vals[max(0, n_lr - long_lb) : n_lr - eff_sk_long]
            fip_seg = seg_long if len(seg_long) >= 25 else lr120
            fip_raw = float(np.mean(fip_seg > 0.0) - np.mean(fip_seg < 0.0))
            max_jump_raw = float(np.max(np.abs(fip_seg)))

            raw_ret_long = (
                float(np.sum(seg_long)) if len(seg_long) >= 20 else float(np.sum(lr120))
            )
            raw_ret_252_s126 = (
                float(np.sum(lr_vals[max(0, n_lr - 252) : n_lr - 126]))
                if n_lr >= 180
                else raw_ret_long
            )
            h_win = highs[-min(252, len(highs)) :]
            valid_h = h_win[np.isfinite(h_win) & (h_win > 0.0)]
            max_h = float(np.max(valid_h)) if valid_h.size > 0 else last_close
            prox_52w_high = float(last_close / max_h) if max_h > 0.0 else 0.0

            reg_lb = min(int(self.config.reg_slope_lookback), len(closes))
            closes_reg = closes[-reg_lb:]
            if reg_lb >= 20 and bool(np.all(np.isfinite(closes_reg) & (closes_reg > 0.0))):
                y_reg = np.log(closes_reg)
                x_reg = np.arange(reg_lb, dtype=np.float64)
                dx = x_reg - float(np.mean(x_reg))
                dy = y_reg - float(np.mean(y_reg))
                ss_xx = float(np.dot(dx, dx))
                ss_yy = float(np.dot(dy, dy))
                ss_xy = float(np.dot(dx, dy))
                if ss_xx <= 1e-12 or ss_yy <= 1e-12:
                    reg_slope_r2 = 0.0
                else:
                    slope_reg = ss_xy / ss_xx
                    r2_reg = min(1.0, max(0.0, (ss_xy * ss_xy) / (ss_xx * ss_yy)))
                    reg_slope_r2 = float(np.clip(slope_reg * 252.0, -5.0, 5.0) * r2_reg)
            else:
                reg_slope_r2 = 0.0

            above_sma200 = False
            if len(closes) >= 200:
                c200 = closes[-200:]
                if bool(np.all(np.isfinite(c200) & (c200 > 0.0))):
                    above_sma200 = bool(last_close >= float(np.mean(c200)))

            rshp_252_s21 = shp_252_s21
            if spy_lr is not None and spy_lr_ts is not None:
                idx_s = np.searchsorted(spy_lr_ts, lr_ts)
                valid_j = idx_s < len(spy_lr_ts)
                if np.any(valid_j):
                    pos_a = np.flatnonzero(valid_j)
                    pos_s = idx_s[pos_a]
                    eq_mask = lr_ts[pos_a] == spy_lr_ts[pos_s]
                    pos_a = pos_a[eq_mask]
                    pos_s = pos_s[eq_mask]
                    if len(pos_a) >= 45:
                        ja = lr_vals[pos_a]
                        jm = spy_lr[pos_s]
                        j_ts = lr_ts[pos_a]
                        nj = len(ja)
                        sk21 = long_skip if nj >= 80 else 0
                        lo_j = max(0, nj - long_lb)
                        hi_j = nj - sk21
                        if hi_j - lo_j >= 25:
                            a_seg = ja[lo_j:hi_j]
                            m_seg = jm[lo_j:hi_j]
                            vm = float(np.var(m_seg, ddof=1))
                            if vm <= 1e-10:
                                rshp_252_s21 = 0.0
                            else:
                                b = float(np.cov(a_seg, m_seg, ddof=1)[0, 1]) / vm
                                al = float(np.mean(a_seg)) - b * float(np.mean(m_seg))
                                rs = max(float(np.std(a_seg - (al + b * m_seg), ddof=1)), 0.004)
                                rshp_252_s21 = (al / rs) * math.sqrt(252.0)

                            if self.config.use_sector_residual and sector_etf_lr_by_sym:
                                sec_name = (
                                    (symbol_sectors or {}).get(symbol, "")
                                    or (symbol_sectors or {}).get(norm_sym, "")
                                )
                                raw_sec_etf = (sector_etf_map or {}).get(sec_name, "")
                                sec_etf = raw_sec_etf.strip().upper()
                                if (
                                    sec_etf
                                    and sec_etf != norm_sym
                                    and sec_etf != benchmark_symbol.strip().upper()
                                    and sec_etf in sector_etf_lr_by_sym
                                ):
                                    sec_ts_arr, sec_lr_arr = sector_etf_lr_by_sym[sec_etf]
                                    seg_ts = j_ts[lo_j:hi_j]
                                    idx_sec = np.searchsorted(sec_ts_arr, seg_ts)
                                    valid_sec = idx_sec < len(sec_ts_arr)
                                    if np.any(valid_sec):
                                        pos_seg = np.flatnonzero(valid_sec)
                                        pos_sec = idx_sec[pos_seg]
                                        eq_sec = seg_ts[pos_seg] == sec_ts_arr[pos_sec]
                                        pos_seg = pos_seg[eq_sec]
                                        pos_sec = pos_sec[eq_sec]
                                        if len(pos_seg) >= 45:
                                            stock_seg = a_seg[pos_seg]
                                            spy_seg = m_seg[pos_seg]
                                            sec_seg = sec_lr_arr[pos_sec]
                                            n_obs = len(stock_seg)
                                            x_mat = np.column_stack(
                                                [
                                                    np.ones(n_obs, dtype=np.float64),
                                                    spy_seg,
                                                    sec_seg,
                                                ]
                                            )
                                            try:
                                                cond = float(np.linalg.cond(x_mat))
                                            except np.linalg.LinAlgError:
                                                cond = float("inf")
                                            if math.isfinite(cond) and cond < 1e6:
                                                coeffs, _, _, _ = np.linalg.lstsq(
                                                    x_mat, stock_seg, rcond=None
                                                )
                                                if len(coeffs) == 3 and bool(
                                                    np.all(np.isfinite(coeffs))
                                                ):
                                                    alpha_2f = float(coeffs[0])
                                                    resid_2f = stock_seg - (x_mat @ coeffs)
                                                    rs_2f = float(np.std(resid_2f, ddof=1))
                                                    if math.isfinite(rs_2f) and rs_2f > 1e-10:
                                                        rshp_252_s21 = (
                                                            alpha_2f / max(rs_2f, 0.004)
                                                        ) * math.sqrt(252.0)

            sh_ok = False
            sh_strength = 0.0
            if symbol in SAFE_HAVEN_SYMBOLS:
                ema20 = _fast_ema_last(clean_closes, 20)
                sma50 = float(np.mean(clean_closes[-50:]))
                lr60 = lr_vals[-min(60, n_lr) :]
                shp_60 = (
                    float(np.mean(lr60)) / max(float(np.std(lr60, ddof=1)), 0.004)
                ) * math.sqrt(252.0)
                sh_ok = (last_close > sma50) or (shp_60 > 0.25 and last_close > ema20)
                sh_strength = max(
                    0.0,
                    0.5 * max(0.0, shp_60)
                    + 0.5 * max(0.0, shp_120)
                    + (0.3 if last_close > sma50 else 0.0),
                )

            rsi5 = _fast_rsi_last(clean_closes, 5)
            pullback_5 = (50.0 - rsi5) / 25.0

            sym_news = _filter_news_by_sources(news_map.get(symbol, ()), allowed_sources)
            agg_sent = self.sentiment_agg.aggregate_for_symbol(sym_news, as_of=as_of_utc)
            insider_score, has_insider_act, has_restatement = _compute_sec_features_for_symbol(
                sec_map.get(symbol, ()),
                as_of_utc=as_of_utc,
                insider_lookback_days=self.config.insider_lookback_days,
                restatement_blackout_days=restatement_blackout_days,
            )

            c_win_60 = closes[-min(60, len(closes)) :]
            v_win_60 = volumes[-min(60, len(volumes)) :]
            valid_dv = (
                np.isfinite(c_win_60)
                & (c_win_60 > 0.0)
                & np.isfinite(v_win_60)
                & (v_win_60 >= 0.0)
            )
            mddv_60d = (
                float(np.median(c_win_60[valid_dv] * v_win_60[valid_dv]))
                if bool(np.any(valid_dv))
                else 0.0
            )
            log_mddv = math.log(max(mddv_60d, 1.0))

            if not all(
                math.isfinite(x)
                for x in (
                    shp_120,
                    abs_shp_120,
                    shp_126_s10,
                    shp_252_s21,
                    rshp_252_s21,
                    raw_ret_long,
                    raw_ret_252_s126,
                    prox_52w_high,
                    reg_slope_r2,
                    mddv_60d,
                    log_mddv,
                    pullback_5,
                    fip_raw,
                    max_jump_raw,
                    agg_sent.news_score,
                    agg_sent.velocity_z,
                    insider_score,
                    atr_pct,
                    ann_vol,
                    ret_5d,
                    sh_strength,
                )
            ):
                continue

            lr_ts_list.append((symbol, lr_ts, lr_vals))
            raw_items.append(
                {
                    "symbol": symbol,
                    "shp_120": shp_120,
                    "abs_shp_120": abs_shp_120,
                    "shp_126_s10": shp_126_s10,
                    "shp_252_s21": shp_252_s21,
                    "rshp_252_s21": rshp_252_s21,
                    "raw_ret_long": raw_ret_long,
                    "raw_ret_252_s126": raw_ret_252_s126,
                    "prox_52w_high": prox_52w_high,
                    "reg_slope_r2": reg_slope_r2,
                    "mddv_60d": mddv_60d,
                    "log_mddv": log_mddv,
                    "above_sma200": above_sma200,
                    "pullback_5": pullback_5,
                    "fip_raw": fip_raw,
                    "max_jump_raw": max_jump_raw,
                    "news_raw": agg_sent.news_score,
                    "news_velocity_z": agg_sent.velocity_z,
                    "news_count": agg_sent.article_count_window,
                    "insider_score": insider_score,
                    "has_insider_activity": has_insider_act,
                    "has_recent_restatement": has_restatement,
                    "atr_pct": atr_pct,
                    "ann_vol": ann_vol,
                    "ret_5d": ret_5d,
                    "sh_ok": sh_ok,
                    "sh_strength": sh_strength,
                }
            )

        if not raw_items:
            return []

        peer_corrs: dict[str, dict[str, float]] = {str(r["symbol"]): {} for r in raw_items}
        if len(lr_ts_list) >= 2:
            all_ts = np.unique(np.concatenate([t[-70:] for _, t, _ in lr_ts_list]))
            tail_ts = all_ts[-min(60, len(all_ts)) :]
            if len(tail_ts) >= 20:
                m_syms = len(lr_ts_list)
                mat = np.zeros((len(tail_ts), m_syms), dtype=np.float64)
                syms_order: list[str] = []
                for j, (sym, ts_arr, lr_arr) in enumerate(lr_ts_list):
                    syms_order.append(sym)
                    ts_sub = ts_arr[-70:]
                    lr_sub = lr_arr[-70:]
                    pos = np.searchsorted(tail_ts, ts_sub)
                    ok = pos < len(tail_ts)
                    pos_ok = pos[ok]
                    lr_ok = lr_sub[ok]
                    eq = tail_ts[pos_ok] == ts_sub[ok]
                    mat[pos_ok[eq], j] = lr_ok[eq]
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr_m = np.atleast_2d(np.asarray(np.corrcoef(mat, rowvar=False), dtype=np.float64))
                for i in range(m_syms):
                    si = syms_order[i]
                    row_corr: dict[str, float] = {}
                    for j in range(m_syms):
                        if i == j:
                            continue
                        v = float(corr_m[i, j])
                        row_corr[syms_order[j]] = v if math.isfinite(v) else 0.0
                    peer_corrs[si] = row_corr

        zs = {
            k: _zscore_cross_section([float(r[k]) for r in raw_items])
            for k in (
                "shp_126_s10",
                "shp_252_s21",
                "rshp_252_s21",
                "raw_ret_long",
                "raw_ret_252_s126",
                "prox_52w_high",
                "reg_slope_r2",
                "log_mddv",
                "pullback_5",
                "fip_raw",
                "max_jump_raw",
            )
        }
        news_zs = [max(-3.0, min(3.0, float(r["news_raw"]) * 1.8)) for r in raw_items]

        m2_sum = (
            self.config.m2_w_long
            + self.config.m2_w_med
            + self.config.m2_w_residual
            + self.config.m2_w_raw_long
            + self.config.m2_w_intermed
            + self.config.m2_w_52w_high
            + self.config.m2_w_reg_slope
            + self.config.m2_w_liquidity
        )
        w_l = self.config.m2_w_long / m2_sum
        w_m = self.config.m2_w_med / m2_sum
        w_r = self.config.m2_w_residual / m2_sum
        w_raw = self.config.m2_w_raw_long / m2_sum
        w_intermed = self.config.m2_w_intermed / m2_sum
        w_52w = self.config.m2_w_52w_high / m2_sum
        w_reg = self.config.m2_w_reg_slope / m2_sum
        w_liq = self.config.m2_w_liquidity / m2_sum

        w_t = self.config.w_trend
        w_p = self.config.w_pullback
        w_n = self.config.w_news_sentiment
        w_i = self.config.w_insider
        w_fip = self.config.w_fip
        w_jump = self.config.w_jump_penalty

        t_scores: list[float] = []
        eligible_flags: list[bool] = []
        for idx, r in enumerate(raw_items):
            t_val = (
                w_l * zs["shp_252_s21"][idx]
                + w_m * zs["shp_126_s10"][idx]
                + w_r * zs["rshp_252_s21"][idx]
                + w_raw * zs["raw_ret_long"][idx]
                + w_intermed * zs["raw_ret_252_s126"][idx]
                + w_52w * zs["prox_52w_high"][idx]
                + w_reg * zs["reg_slope_r2"][idx]
                + w_liq * zs["log_mddv"][idx]
            )
            t_scores.append(t_val)
            is_elig = True
            if (
                self.config.max_stock_ann_vol is not None
                and float(r["ann_vol"]) > self.config.max_stock_ann_vol
            ):
                is_elig = False
            if self.config.trend_above_sma200_filter and not bool(r["above_sma200"]):
                is_elig = False
            eligible_flags.append(is_elig)

        if self.config.fip_gate_quantile is not None:
            elig_indices = [idx for idx, flag in enumerate(eligible_flags) if flag]
            if elig_indices:
                elig_indices.sort(
                    key=lambda idx: (-float(raw_items[idx]["fip_raw"]), str(raw_items[idx]["symbol"]))
                )
                k_keep = max(
                    1, math.ceil(len(elig_indices) * self.config.fip_gate_quantile)
                )
                keep_set = set(elig_indices[:k_keep])
                for idx in elig_indices:
                    if idx not in keep_set:
                        eligible_flags[idx] = False

        signals: list[SignalRow] = []
        for idx, r in enumerate(raw_items):
            sym = str(r["symbol"])
            t_score = t_scores[idx]
            p_score = zs["pullback_5"][idx]
            fip_z = zs["fip_raw"][idx]
            jump_z = zs["max_jump_raw"][idx]
            n_score = news_zs[idx]
            i_score = float(r["insider_score"])
            eff_wn = w_n if int(r["news_count"]) > 0 else 0.0
            eff_wi = w_i if (w_i > 0.0 and bool(r["has_insider_activity"])) else 0.0
            if w_fip > 0.0 or w_jump > 0.0:
                denom = w_t + w_p + eff_wn + eff_wi + w_fip
                num = (
                    w_t * t_score
                    + w_p * p_score
                    + eff_wn * n_score
                    + eff_wi * i_score
                    + w_fip * fip_z
                    - w_jump * max(0.0, jump_z)
                )
                composite = num / denom if denom > 0.0 else 0.0
            else:
                denom = w_t + w_p + eff_wn + eff_wi
                composite = (
                    (w_t * t_score + w_p * p_score + eff_wn * n_score + eff_wi * i_score) / denom
                    if denom > 0.0
                    else 0.0
                )
            signals.append(
                SignalRow(
                    as_of=as_of_utc,
                    symbol=sym,
                    trend_score=round(t_score, 6),
                    pullback_score=round(p_score, 6),
                    news_score=round(n_score, 6),
                    news_velocity_z=round(float(r["news_velocity_z"]), 6),
                    composite_score=round(composite, 6),
                    macro_regime_scale=macro_scale,
                    atr_pct=round(float(r["atr_pct"]), 6),
                    ann_vol=round(float(r["ann_vol"]), 6),
                    ret_5d=round(float(r["ret_5d"]), 6),
                    abs_trend_raw=round(float(r["abs_shp_120"]), 6),
                    peer_corr_60d=peer_corrs.get(sym, {}),
                    safe_haven_trend_ok=bool(r["sh_ok"]),
                    safe_haven_strength=float(r["sh_strength"]),
                    insider_score=round(i_score, 6),
                    has_recent_restatement=bool(r["has_recent_restatement"]),
                    fip_score=round(fip_z, 6),
                    mddv_60d=round(float(r["mddv_60d"]), 2),
                    stage1_finalist=bool(eligible_flags[idx]),
                )
            )
        return signals
