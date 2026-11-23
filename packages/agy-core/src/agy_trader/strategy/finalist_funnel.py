from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime

import numpy as np
import pandas as pd

from agy_trader.core.models import SignalRow, ensure_utc
from agy_trader.strategy.features import _zscore_cross_section


def compute_pit_pead_proxy(
    df_bars: pd.DataFrame,
    as_of: datetime,
    earnings_dates: Sequence[date] = (),
    *,
    lookback_days: int = 45,
    min_age_days: int = 1,
) -> float:
    """Compute a strictly point-in-time Post-Earnings Announcement Drift (PEAD) proxy in [-1.5, 1.5].

    Strictly filters ``df_bars`` to ``timestamp <= as_of`` (UTC) before inspecting prices or volumes.
    Uses confirmed ``earnings_dates`` when available in the trailing ``[min_age_days, lookback_days]``
    window (scoring the 2-bar ``[e_idx - 1 -> e_idx + 1]`` reaction window once ``e_idx + 1 < n_bars``
    so both before-market-open and after-market-close announcements are signed accurately), and falls
    back to detecting the strongest abnormal volume-spike + price-gap catalyst bar
    (``volume >= 2.2 * trailing_20d_median_volume`` and ``abs(ret_1d) >= 0.03``) when no confirmed
    earnings date is present.
    """
    if df_bars is None or df_bars.empty:
        return 0.0

    as_of_utc = ensure_utc(as_of)
    parsed_ts = pd.to_datetime(df_bars["timestamp"], utc=True, errors="coerce")
    mask = parsed_ts.notna() & (parsed_ts <= as_of_utc)
    if not bool(mask.any()):
        return 0.0

    hist = df_bars.loc[mask].assign(_ts=parsed_ts.loc[mask]).sort_values("_ts")
    n_bars = len(hist)
    if n_bars < 30:
        return 0.0

    closes = hist["close"].to_numpy(dtype=np.float64)
    if not bool(np.isfinite(closes[-1]) and closes[-1] > 0.0):
        return 0.0

    vols = (
        hist["volume"].to_numpy(dtype=np.float64)
        if "volume" in hist.columns
        else np.zeros(n_bars, dtype=np.float64)
    )
    last_close = float(closes[-1])
    if not math.isfinite(last_close) or last_close <= 0.0:
        return 0.0
    as_of_date = as_of_utc.date()

    # 1. Confirmed earnings dates strictly in [min_age_days, lookback_days] calendar days <= as_of_date
    eff_min_cal = max(1, int(min_age_days))
    eff_max_cal = max(eff_min_cal, int(lookback_days))
    if earnings_dates:
        valid_ed = [
            ed
            for ed in earnings_dates
            if eff_min_cal <= (as_of_date - ed).days <= eff_max_cal
        ]
        if valid_ed:
            latest_ed = max(valid_ed)
            bar_dates = [ts.date() for ts in hist["_ts"]]
            e_idx: int | None = None
            for idx in range(1, n_bars):
                if bar_dates[idx] >= latest_ed:
                    e_idx = idx
                    break
            # Require e_idx + 1 < n_bars so full BMO/AMC reaction is realized on bar e_idx+1
            if e_idx is not None and 1 <= e_idx and (e_idx + 1) < n_bars:
                prev_c = float(closes[e_idx - 1])
                day0_c = float(closes[e_idx])
                react_idx = e_idx + 1
                event_close = float(closes[react_idx])
                if (
                    math.isfinite(prev_c)
                    and math.isfinite(day0_c)
                    and math.isfinite(event_close)
                    and prev_c > 0.0
                    and day0_c > 0.0
                    and event_close > 0.0
                ):
                    event_ret = event_close / prev_c - 1.0
                    post_ret = last_close / event_close - 1.0
                    total_ret = last_close / prev_c - 1.0
                    if all(math.isfinite(x) for x in (event_ret, post_ret, total_ret)):
                        ref_ret = event_ret if abs(event_ret) >= 0.005 else total_ret
                        if abs(ref_ret) < 1e-6:
                            return 0.0
                        hold_mult = 1.0 if (post_ret * ref_ret >= -0.02) else 0.3
                        mag = min(
                            1.5,
                            (
                                abs(event_ret)
                                + max(0.0, math.copysign(1.0, ref_ret) * post_ret)
                            )
                            / 0.05,
                        )
                        raw_score = math.copysign(1.0, ref_ret) * mag * hold_mult
                        if math.isfinite(raw_score):
                            return float(max(-1.5, min(1.5, raw_score)))
            elif e_idx is not None and e_idx == n_bars - 1:
                # Earnings day is the final bar (t+1 AMC reaction not yet realized): wait for t+1
                return 0.0

    # 2. Fallback: strongest volume-spike + price-jump earnings-proxy catalyst in trailing window
    eff_min_bar_age = max(1, int(min_age_days))
    max_bar_age = min(int(lookback_days), n_bars - 11)
    if max_bar_age < eff_min_bar_age:
        return 0.0

    best_strength = 0.0
    best_event_ret = 0.0
    best_event_close = 0.0

    for age in range(eff_min_bar_age, max_bar_age + 1):
        idx = n_bars - 1 - age
        if idx < 10:
            break
        prev_c = float(closes[idx - 1])
        ev_c = float(closes[idx])
        if prev_c <= 0.0 or ev_c <= 0.0 or not (math.isfinite(prev_c) and math.isfinite(ev_c)):
            continue
        bar_ret = ev_c / prev_c - 1.0
        if not math.isfinite(bar_ret) or abs(bar_ret) < 0.03:
            continue
        vol_win = vols[max(0, idx - 20) : idx]
        valid_vols = vol_win[np.isfinite(vol_win) & (vol_win > 0.0)]
        if valid_vols.size < 10:
            continue
        med_vol = float(np.median(valid_vols))
        bar_vol = float(vols[idx])
        if med_vol <= 0.0 or not math.isfinite(bar_vol) or bar_vol < 2.2 * med_vol:
            continue
        strength = abs(bar_ret) * min(5.0, bar_vol / med_vol)
        if math.isfinite(strength) and strength > best_strength:
            best_strength = strength
            best_event_ret = bar_ret
            best_event_close = ev_c

    if best_strength <= 0.0 or best_event_close <= 0.0:
        return 0.0

    post_event_ret = last_close / best_event_close - 1.0
    if not math.isfinite(post_event_ret):
        return 0.0
    hold_mult = 1.0 if (post_event_ret * best_event_ret >= -0.02) else 0.3
    raw = (
        math.copysign(1.0, best_event_ret)
        * min(1.5, abs(best_event_ret) / 0.05)
        * hold_mult
    )
    if not math.isfinite(raw):
        return 0.0
    return float(max(-1.5, min(1.5, raw)))


def apply_stage2_finalist_overlay(
    signals: Sequence[SignalRow],
    *,
    bars_by_symbol: Mapping[str, pd.DataFrame],
    held_symbols: set[str] | frozenset[str],
    as_of: datetime,
    finalist_pool_size: int = 40,
    w_pead: float = 0.0,
    w_trend: float = 0.80,
    w_pullback: float = 0.0,
    w_news_sentiment: float = 0.0,
    w_insider: float = 0.0,
    w_fip: float = 0.0,
    earnings_by_symbol: Mapping[str, Sequence[date]] | None = None,
) -> list[SignalRow]:
    """Two-Stage Finalist Funnel: evaluate Stage-2 PEAD overlay strictly on Top-N finalists + held.

    Enforces the Two-Stage Funnel invariant cleanly via ``stage1_finalist``:
      - Stage-1 finalists (Top-``finalist_pool_size`` + currently held positions) are marked
        ``stage1_finalist=True`` and receive the cross-sectionally z-scored Stage-2 PEAD overlay.
      - Non-finalists are marked ``stage1_finalist=False`` and keep their original ``composite_score``
        un-mutated so positive/neutral non-finalists are never turned into fake short signals, while
        ``PortfolioConstructor`` restricts long selection strictly to ``s.stage1_finalist``.
    """
    if not signals:
        return []

    as_of_utc = ensure_utc(as_of)
    norm_held = frozenset(s.strip().upper() for s in held_symbols if s.strip())
    pool_k = max(1, int(finalist_pool_size))

    sorted_stage1 = sorted(signals, key=lambda s: (-s.composite_score, s.symbol))
    top_pool_syms = {s.symbol for s in sorted_stage1[:pool_k]}
    finalist_syms = top_pool_syms | (norm_held & {s.symbol for s in signals})

    finalist_order = [s for s in sorted_stage1 if s.symbol in finalist_syms]
    earnings_map = earnings_by_symbol or {}

    raw_pead_by_sym: dict[str, float] = {}
    for s in finalist_order:
        df = bars_by_symbol.get(s.symbol)
        e_dates = earnings_map.get(s.symbol, ())
        raw_pead_by_sym[s.symbol] = (
            compute_pit_pead_proxy(df, as_of_utc, e_dates)
            if df is not None and not df.empty
            else 0.0
        )

    pead_zs = _zscore_cross_section(
        [raw_pead_by_sym[s.symbol] for s in finalist_order],
        clip_limit=3.0,
    )
    pead_z_by_sym = {
        s.symbol: round(pead_zs[idx], 6) for idx, s in enumerate(finalist_order)
    }

    updated_by_sym: dict[str, SignalRow] = {}
    for s in finalist_order:
        pz = pead_z_by_sym[s.symbol]
        if w_pead > 0.0:
            eff_wn = w_news_sentiment if abs(s.news_score) > 1e-12 else 0.0
            eff_wi = w_insider if abs(s.insider_score) > 1e-12 else 0.0
            base_weight_sum = max(
                w_trend + w_pullback + eff_wn + eff_wi + w_fip,
                1e-6,
            )
            new_comp = round(
                (base_weight_sum * s.composite_score + w_pead * pz)
                / (base_weight_sum + w_pead),
                6,
            )
        else:
            new_comp = s.composite_score
        updated_by_sym[s.symbol] = s.model_copy(
            update={
                "pead_score": pz,
                "composite_score": new_comp,
                "stage1_finalist": True,
            }
        )

    result: list[SignalRow] = []
    for s in signals:
        if s.symbol in updated_by_sym:
            result.append(updated_by_sym[s.symbol])
        else:
            result.append(s.model_copy(update={"stage1_finalist": False}))

    return result
