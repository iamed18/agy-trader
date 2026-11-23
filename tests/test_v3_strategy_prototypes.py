from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from agy_trader.backtest.engine import BacktestEngine
from agy_trader.core.config import AppConfig, BacktestConfig, RiskConfig, StrategyConfig
from agy_trader.core.models import (
    AssetRiskMetadata,
    AssetType,
    ControlDials,
    MacroPoint,
    SignalRow,
    UniverseSymbol,
)
from agy_trader.strategy.features import FeatureEngine, compute_macro_regime_scale
from agy_trader.strategy.finalist_funnel import (
    apply_stage2_finalist_overlay,
    compute_pit_pead_proxy,
)
from agy_trader.strategy.portfolio import PortfolioConstructor


def _make_bars_df(
    closes: list[float] | np.ndarray,
    *,
    start_dt: datetime = datetime(2025, 1, 2, 21, 0, tzinfo=UTC),
    volumes: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    arr = np.asarray(closes, dtype=float)
    n = len(arr)
    vols = (
        np.asarray(volumes, dtype=float)
        if volumes is not None
        else np.full(n, 1_000_000.0, dtype=float)
    )
    rows = []
    cur = start_dt
    for i in range(n):
        while cur.weekday() >= 5:
            cur += timedelta(days=1)
        c = float(arr[i])
        rows.append(
            {
                "timestamp": cur,
                "open": c * 0.998 if np.isfinite(c) else c,
                "high": c * 1.006 if np.isfinite(c) else c,
                "low": c * 0.994 if np.isfinite(c) else c,
                "close": c,
                "volume": float(vols[i]),
            }
        )
        cur += timedelta(days=1)
    return pd.DataFrame(rows)


def test_config_v02_defaults_and_v03_validators() -> None:
    cfg = StrategyConfig()
    assert cfg.use_sector_residual is False
    assert cfg.w_fip == 0.0
    assert cfg.w_jump_penalty == 0.0
    assert cfg.vol_sizing_mode == "inv_atr"
    assert cfg.vol_sizing_floor == 0.18
    assert cfg.vol_sizing_cap == 0.45
    assert cfg.portfolio_vol_target_ann is None
    assert cfg.rebalance_tranches == 1
    assert cfg.regime_mode == "legacy_stacked"
    assert cfg.w_pead == 0.0
    assert cfg.finalist_pool_size == 40

    rcfg = RiskConfig()
    assert rcfg.max_drawdown_cooldown_days is None
    assert rcfg.delisting_haircut_pct == 0.10

    with pytest.raises(ValueError, match="vol_sizing_cap must be >= vol_sizing_floor"):
        StrategyConfig(vol_sizing_floor=0.40, vol_sizing_cap=0.25)

    # Reject w_fip > 0, w_jump_penalty > 0, or use_sector_residual=True when momentum_mode != "m2_residual"
    with pytest.raises(ValueError, match="require momentum_mode == 'm2_residual'"):
        StrategyConfig(momentum_mode="legacy", w_fip=0.25)
    with pytest.raises(ValueError, match="require momentum_mode == 'm2_residual'"):
        StrategyConfig(momentum_mode="legacy", w_jump_penalty=0.10)
    with pytest.raises(ValueError, match="require momentum_mode == 'm2_residual'"):
        StrategyConfig(momentum_mode="legacy", use_sector_residual=True)

    # Pure FIP weight is valid under momentum_mode="m2_residual"
    fip_only = StrategyConfig(
        momentum_mode="m2_residual",
        w_trend=0.0,
        w_pullback=0.0,
        w_news_sentiment=0.0,
        w_fip=0.5,
    )
    assert fip_only.w_fip == 0.5


def test_track2_sector_residual_and_fallback() -> None:
    rng = np.random.default_rng(42)
    n = 160
    spy_rets = rng.normal(0.0004, 0.008, size=n)
    xlk_rets = 1.1 * spy_rets + rng.normal(0.0012, 0.006, size=n)
    # AAPL rides XLK beta with negative idiosyncratic alpha; MSFT has strong positive alpha over XLK
    aapl_rets = 0.2 * spy_rets + 1.25 * (xlk_rets - spy_rets) - 0.0005 + rng.normal(0.0, 0.004, size=n)
    msft_rets = 0.8 * spy_rets + 0.5 * (xlk_rets - spy_rets) + 0.0018 + rng.normal(0.0, 0.004, size=n)
    xle_rets = 0.7 * spy_rets + rng.normal(-0.0005, 0.009, size=n)
    xom_rets = 0.9 * xle_rets + rng.normal(0.0002, 0.005, size=n)

    def _to_prices(r: np.ndarray) -> np.ndarray:
        return 100.0 * np.exp(np.cumsum(r))

    bars = {
        "SPY": _make_bars_df(_to_prices(spy_rets)),
        "XLK": _make_bars_df(_to_prices(xlk_rets)),
        "AAPL": _make_bars_df(_to_prices(aapl_rets)),
        "MSFT": _make_bars_df(_to_prices(msft_rets)),
        "XOM": _make_bars_df(_to_prices(xom_rets)),
        # Short history ETF (<45 bars) triggers seamless 1-factor SPY fallback
        "XLC": _make_bars_df(_to_prices(spy_rets[-30:]), start_dt=datetime(2025, 6, 1, 21, 0, tzinfo=UTC)),
        "META": _make_bars_df(_to_prices(msft_rets)),
    }
    as_of = pd.Timestamp(bars["SPY"]["timestamp"].iloc[-1]).to_pydatetime()
    sectors = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "XLK": "Technology",  # XLK must never regress on itself!
        "XOM": "Energy",
        "META": "Communication Services",
    }
    etf_map = {
        "Technology": "XLK",
        "Energy": "XLE",  # Missing from bars -> falls back to 1-factor SPY
        "Communication Services": "XLC",  # <45 aligned bars -> falls back to 1-factor SPY
    }

    cfg_1f = StrategyConfig(
        momentum_mode="m2_residual",
        m2_w_long=0.0,
        m2_w_med=0.0,
        m2_w_residual=1.0,
        w_trend=1.0,
        w_pullback=0.0,
        w_news_sentiment=0.0,
        use_sector_residual=False,
    )
    cfg_2f = cfg_1f.model_copy(update={"use_sector_residual": True})

    sigs_1f = {
        s.symbol: s
        for s in FeatureEngine(cfg_1f).build_signals(
            as_of,
            bars,
            universe_symbols={"AAPL", "MSFT", "XOM", "META", "XLK"},
            symbol_sectors=sectors,
            sector_etf_map=etf_map,
        )
    }
    sigs_2f = {
        s.symbol: s
        for s in FeatureEngine(cfg_2f).build_signals(
            as_of,
            bars,
            universe_symbols={"AAPL", "MSFT", "XOM", "META", "XLK"},
            symbol_sectors=sectors,
            sector_etf_map=etf_map,
        )
    }

    assert set(sigs_2f.keys()) == {"AAPL", "MSFT", "XOM", "META", "XLK"}
    # Under 2-factor sector residual, MSFT (positive alpha over XLK) beats AAPL
    assert sigs_2f["MSFT"].trend_score > sigs_2f["AAPL"].trend_score
    # And the 2-factor score differs from 1-factor where XLK is present
    assert sigs_2f["AAPL"].trend_score != sigs_1f["AAPL"].trend_score


def test_track2_fip_and_jump_penalty_and_v02_parity() -> None:
    n = 150
    # Smooth steady trend vs jumpy single-day spike with identical endpoint return
    smooth_lr = np.full(n, 0.0015, dtype=float)
    smooth_lr[::7] = -0.0003
    total_log_ret = float(np.sum(smooth_lr))

    jumpy_lr = np.full(n, -0.0004, dtype=float)
    jumpy_lr[60] = total_log_ret - float(np.sum(jumpy_lr)) + (-0.0004)

    flat_lr = np.linspace(-0.001, 0.001, n)
    spy_lr = np.full(n, 0.0005, dtype=float)

    bars = {
        "SPY": _make_bars_df(100.0 * np.exp(np.cumsum(spy_lr))),
        "SMOOTH": _make_bars_df(100.0 * np.exp(np.cumsum(smooth_lr))),
        "JUMPY": _make_bars_df(100.0 * np.exp(np.cumsum(jumpy_lr))),
        "MID": _make_bars_df(100.0 * np.exp(np.cumsum(flat_lr))),
    }
    as_of = pd.Timestamp(bars["SPY"]["timestamp"].iloc[-1]).to_pydatetime()

    cfg_base = StrategyConfig(
        momentum_mode="m2_residual",
        w_trend=0.80,
        w_pullback=0.20,
        w_news_sentiment=0.0,
        w_fip=0.0,
        w_jump_penalty=0.0,
    )
    sigs_base = {
        s.symbol: s
        for s in FeatureEngine(cfg_base).build_signals(
            as_of, bars, universe_symbols={"SMOOTH", "JUMPY", "MID"}
        )
    }
    assert sigs_base["SMOOTH"].fip_score > sigs_base["JUMPY"].fip_score

    cfg_fip_jump = cfg_base.model_copy(update={"w_fip": 0.25, "w_jump_penalty": 0.20})
    sigs_fj = {
        s.symbol: s
        for s in FeatureEngine(cfg_fip_jump).build_signals(
            as_of, bars, universe_symbols={"SMOOTH", "JUMPY", "MID"}
        )
    }
    # Smooth trend gets boosted and jumpy trend gets penalized
    assert (sigs_fj["SMOOTH"].composite_score - sigs_fj["JUMPY"].composite_score) > (
        sigs_base["SMOOTH"].composite_score - sigs_base["JUMPY"].composite_score
    )


def test_track6_daniel_moskowitz_regime_scale() -> None:
    # 1. Bull / V-recovery regime (SPY > SMA200 and SPY > close[-253]) with elevated vol (~25%)
    n = 260
    bull_rets = np.where(np.arange(n) % 2 == 0, 0.018, -0.014)
    spy_bull = _make_bars_df(100.0 * np.exp(np.cumsum(bull_rets)))
    as_of_bull = pd.Timestamp(spy_bull["timestamp"].iloc[-1]).to_pydatetime()

    legacy_bull = compute_macro_regime_scale(
        as_of_bull,
        spy_bull,
        spy_sma200_defensive_scale=0.50,
        regime_mode="legacy_stacked",
    )
    dm_bull = compute_macro_regime_scale(
        as_of_bull,
        spy_bull,
        spy_sma200_defensive_scale=0.50,
        regime_mode="daniel_moskowitz",
    )
    assert dm_bull == 1.0
    assert dm_bull > legacy_bull

    # 2. Bear + high vol panic state (SPY < SMA200 and vol > 35% + VIX=36)
    bear_rets = np.where(np.arange(n) % 2 == 0, -0.028, 0.022)
    spy_bear = _make_bars_df(100.0 * np.exp(np.cumsum(bear_rets)))
    as_of_bear = pd.Timestamp(spy_bear["timestamp"].iloc[-1]).to_pydatetime()
    macro_pts = [
        MacroPoint(as_of_date=(as_of_bear - timedelta(days=5)).date(), series_id="VIXCLS", value=36.0),
        MacroPoint(
            as_of_date=(as_of_bear - timedelta(days=5)).date(),
            series_id="BAMLH0A0HYM2",
            value=6.5,
        ),
    ]
    dm_bear = compute_macro_regime_scale(
        as_of_bear,
        spy_bear,
        macro_points=macro_pts,
        regime_mode="daniel_moskowitz",
    )
    assert 0.40 <= dm_bear <= 0.45


def test_track4_vol_sizing_modes_and_portfolio_vol_target_and_non_psd_matrix() -> None:
    as_of = datetime(2025, 6, 15, 20, 0, tzinfo=UTC)
    signals = [
        SignalRow(
            as_of=as_of,
            symbol="LOWVOL",
            trend_score=1.2,
            pullback_score=0.5,
            news_score=0.0,
            news_velocity_z=0.0,
            composite_score=1.1,
            macro_regime_scale=1.0,
            atr_pct=0.006,  # Very low ATR
            ann_vol=0.10,   # Below 0.18 floor
            ret_5d=0.01,
            abs_trend_raw=0.8,
            peer_corr_60d={"HIGHVOL": 0.65},
        ),
        SignalRow(
            as_of=as_of,
            symbol="HIGHVOL",
            trend_score=1.1,
            pullback_score=0.4,
            news_score=0.0,
            news_velocity_z=0.0,
            composite_score=1.0,
            macro_regime_scale=1.0,
            atr_pct=0.036,
            ann_vol=0.60,   # Above 0.45 cap
            ret_5d=0.02,
            abs_trend_raw=0.7,
            peer_corr_60d={"LOWVOL": 0.65},
        ),
    ]
    meta = {
        "LOWVOL": AssetRiskMetadata(symbol="LOWVOL", asset_type=AssetType.equity, sector="Utilities"),
        "HIGHVOL": AssetRiskMetadata(symbol="HIGHVOL", asset_type=AssetType.equity, sector="Tech"),
    }
    dials = ControlDials(allow_shorting=False, updated_at=as_of)
    rcfg = RiskConfig(max_long_position_pct=0.80, max_sector_gross_pct=0.80, max_net_exposure=1.0)

    # 1. Equal weight mode
    ctor_eq = PortfolioConstructor(
        StrategyConfig(vol_sizing_mode="equal", top_k_long=2, min_signal_threshold=0.1),
        rcfg,
    )
    w_eq = ctor_eq.compute_target_weights(signals, dials, as_of, meta)
    assert pytest.approx(w_eq["LOWVOL"], rel=1e-6) == pytest.approx(w_eq["HIGHVOL"], rel=1e-6)

    # 2. Capped inv-vol mode (ratio bounded by cap/floor = 0.45/0.18 = 2.5x instead of 6.0x ATR ratio)
    ctor_cap = PortfolioConstructor(
        StrategyConfig(
            vol_sizing_mode="capped_inv_vol",
            vol_sizing_floor=0.18,
            vol_sizing_cap=0.45,
            top_k_long=2,
            min_signal_threshold=0.1,
        ),
        rcfg,
    )
    w_cap = ctor_cap.compute_target_weights(signals, dials, as_of, meta)
    assert pytest.approx(w_cap["LOWVOL"] / w_cap["HIGHVOL"], rel=1e-4) == 0.45 / 0.18

    # 3. Portfolio-level ex-ante vol target scales down when portfolio vol > target
    ctor_vt = PortfolioConstructor(
        StrategyConfig(
            vol_sizing_mode="equal",
            portfolio_vol_target_ann=0.15,
            top_k_long=2,
            min_signal_threshold=0.1,
        ),
        rcfg,
    )
    w_vt = ctor_vt.compute_target_weights(signals, dials, as_of, meta)
    assert sum(w_vt.values()) < sum(w_eq.values())
    assert pytest.approx(w_vt["LOWVOL"] / w_vt["HIGHVOL"], rel=1e-6) == 1.0

    # 4. Regression (HIGH #6): non-PSD peer_corr_60d + ann_vol == 0.0 must still scale down under vol target!
    non_psd_signals = [
        SignalRow(
            as_of=as_of,
            symbol="A",
            trend_score=1.0,
            pullback_score=0.0,
            news_score=0.0,
            news_velocity_z=0.0,
            composite_score=1.2,
            macro_regime_scale=1.0,
            atr_pct=0.02,
            ann_vol=0.0,  # Zero vol must floor to 0.05
            ret_5d=0.01,
            peer_corr_60d={"B": -0.99, "C": -0.99},
        ),
        SignalRow(
            as_of=as_of,
            symbol="B",
            trend_score=1.0,
            pullback_score=0.0,
            news_score=0.0,
            news_velocity_z=0.0,
            composite_score=1.1,
            macro_regime_scale=1.0,
            atr_pct=0.02,
            ann_vol=0.50,
            ret_5d=0.01,
            peer_corr_60d={"A": -0.99, "C": -0.99},
        ),
        SignalRow(
            as_of=as_of,
            symbol="C",
            trend_score=1.0,
            pullback_score=0.0,
            news_score=0.0,
            news_velocity_z=0.0,
            composite_score=1.0,
            macro_regime_scale=1.0,
            atr_pct=0.02,
            ann_vol=0.50,
            ret_5d=0.01,
            peer_corr_60d={"A": -0.99, "B": -0.99},
        ),
    ]
    meta_3 = {
        "A": AssetRiskMetadata(symbol="A", asset_type=AssetType.equity, sector="S1"),
        "B": AssetRiskMetadata(symbol="B", asset_type=AssetType.equity, sector="S2"),
        "C": AssetRiskMetadata(symbol="C", asset_type=AssetType.equity, sector="S3"),
    }
    ctor_non_psd = PortfolioConstructor(
        StrategyConfig(
            vol_sizing_mode="equal",
            portfolio_vol_target_ann=0.08,
            top_k_long=3,
            min_signal_threshold=0.1,
        ),
        rcfg,
    )
    w_non_psd = ctor_non_psd.compute_target_weights(non_psd_signals, dials, as_of, meta_3)
    # Without PSD projection & diag_var floor, [-0.99, -0.99, -0.99] has negative eigenvalue and would fail to scale
    assert sum(w_non_psd.values()) < 0.75


def test_target_weights_override_validation_and_pause_new_entries() -> None:
    as_of = datetime(2025, 6, 15, 20, 0, tzinfo=UTC)
    meta = {
        "AAPL": AssetRiskMetadata(symbol="AAPL", asset_type=AssetType.equity, sector="Tech"),
        "MSFT": AssetRiskMetadata(symbol="MSFT", asset_type=AssetType.equity, sector="Tech"),
    }
    ctor = PortfolioConstructor(StrategyConfig(), RiskConfig(max_long_position_pct=0.50))
    dials = ControlDials(pause_new_entries=True, allow_shorting=False, updated_at=as_of)

    # 1. NaN / inf in target_weights_override raises ValueError
    with pytest.raises(ValueError, match="non-finite weight"):
        ctor.build_proposed_orders(
            signals=[],
            current_positions={"AAPL": 100.0},
            prices={"AAPL": 100.0},
            equity=100_000.0,
            dials=dials,
            as_of=as_of,
            universe_meta=meta,
            target_weights_override={"AAPL": float("nan")},
        )
    with pytest.raises(ValueError, match="non-finite weight"):
        ctor.build_proposed_orders(
            signals=[],
            current_positions={"AAPL": 100.0},
            prices={"AAPL": 100.0},
            equity=100_000.0,
            dials=dials,
            as_of=as_of,
            universe_meta=meta,
            target_weights_override={"AAPL": float("inf")},
        )

    # 2. Lowercase keys are normalized and pause_new_entries blocks fresh opens / increases while allowing reductions
    orders_paused = ctor.build_proposed_orders(
        signals=[],
        current_positions={"AAPL": 200.0},  # current weight = 0.20
        prices={"AAPL": 100.0, "MSFT": 100.0},
        equity=100_000.0,
        dials=dials,
        as_of=as_of,
        universe_meta=meta,
        target_weights_override={"aapl": 0.10, "msft": 0.20},
    )
    # MSFT new entry is suppressed by pause_new_entries; AAPL reduction from 0.20 -> 0.10 is allowed
    assert [o.symbol for o in orders_paused] == ["AAPL"]
    assert orders_paused[0].is_risk_reducing is True


def test_track3_pit_pead_proxy_amc_inf_and_two_stage_funnel_no_fake_shorts() -> None:
    n = 50
    closes = np.linspace(100.0, 105.0, n)
    vols = np.full(n, 1_000_000.0)
    # Create a volume-spike + +6% earnings gap at index 35 (age 14 bars) that holds its gain
    closes[35:] = np.linspace(112.0, 116.0, n - 35)
    vols[35] = 3_500_000.0
    df = _make_bars_df(closes, volumes=vols)
    as_of = pd.Timestamp(df["timestamp"].iloc[-1]).to_pydatetime()

    score_fallback = compute_pit_pead_proxy(df, as_of)
    assert score_fallback > 0.5

    # Confirmed earnings_dates branch
    ed_confirmed = date.fromisoformat(pd.Timestamp(df["timestamp"].iloc[35]).date().isoformat())
    score_confirmed = compute_pit_pead_proxy(df, as_of, earnings_dates=[ed_confirmed])
    assert score_confirmed > 0.5

    # Regression (HIGH #5a): AMC earnings where stock drifts +0.6% on ed (idx 35) and crashes -10% on ed+1 (idx 36)
    amc_closes = np.full(n, 100.0)
    amc_closes[35] = 100.60  # +0.6% pre-earnings drift on ed
    amc_closes[36:] = 90.54  # -10% crash on ed+1 after AMC earnings release
    df_amc = _make_bars_df(amc_closes)
    ed_amc = date.fromisoformat(pd.Timestamp(df_amc["timestamp"].iloc[35]).date().isoformat())
    score_amc = compute_pit_pead_proxy(df_amc, as_of, earnings_dates=[ed_amc])
    assert score_amc < -0.5

    # Regression (HIGH #5b): +inf price on confirmed earnings bar must NOT produce NaN or crash
    df_inf = df_amc.copy()
    df_inf.loc[35, "close"] = float("inf")
    score_inf = compute_pit_pead_proxy(df_inf, as_of, earnings_dates=[ed_amc])
    assert np.isfinite(score_inf)

    # Strict PIT causality: mutating a bar AFTER as_of_mid must have zero effect on as_of_mid
    as_of_mid = pd.Timestamp(df["timestamp"].iloc[32]).to_pydatetime()
    score_mid_clean = compute_pit_pead_proxy(df, as_of_mid)
    df_mutated = df.copy()
    df_mutated.loc[35:, "close"] *= 3.0
    df_mutated.loc[35:, "volume"] *= 10.0
    score_mid_mutated = compute_pit_pead_proxy(df_mutated, as_of_mid)
    assert score_mid_clean == score_mid_mutated == 0.0

    # Regression (HIGH #4): Two-Stage Funnel with allow_shorting=True must NEVER turn positive-score
    # non-finalists into fake short candidates, while still preventing non-finalists from leapfrogging
    # Stage-1 finalists into Top-K longs!
    neg_closes = np.linspace(100.0, 102.0, n)
    neg_closes[35:] = np.linspace(88.0, 84.0, n - 35)  # Strong negative PEAD
    neg_vols = np.full(n, 1_000_000.0)
    neg_vols[35] = 4_000_000.0
    df_neg = _make_bars_df(neg_closes, volumes=neg_vols)
    df_flat = _make_bars_df(np.linspace(100.0, 103.0, n))

    base_signals = [
        SignalRow(
            as_of=as_of,
            symbol=f"SYM{i}",
            trend_score=0.5 - i * 0.04,
            pullback_score=0.0,
            news_score=0.0,
            news_velocity_z=0.0,
            composite_score=round(0.45 - i * 0.05, 6),  # SYM0=0.45 .. SYM4=0.25, SYM5=0.20, SYM6=0.15, SYM7=0.10
            macro_regime_scale=1.0,
            atr_pct=0.015,
            ann_vol=0.22,
            ret_5d=0.01,
        )
        for i in range(8)
    ]
    bars_map = {s.symbol: df_flat for s in base_signals}
    bars_map["SYM0"] = df
    bars_map["SYM4"] = df_neg  # Finalist #5 gets -3z PEAD -> composite drops negative

    overlaid = apply_stage2_finalist_overlay(
        base_signals,
        bars_by_symbol=bars_map,
        held_symbols=set(),
        as_of=as_of,
        finalist_pool_size=5,  # SYM0..SYM4 are finalists; SYM5..SYM7 are positive-score non-finalists
        w_pead=0.50,
        w_trend=0.50,
    )
    by_sym = {s.symbol: s for s in overlaid}
    for fin_idx in range(5):
        assert by_sym[f"SYM{fin_idx}"].stage1_finalist is True
    for non_fin_idx in range(5, 8):
        # Non-finalists keep their positive composite_score un-mutated and stage1_finalist=False
        assert by_sym[f"SYM{non_fin_idx}"].stage1_finalist is False
        assert by_sym[f"SYM{non_fin_idx}"].composite_score > 0.0
        assert by_sym[f"SYM{non_fin_idx}"].pead_score == 0.0

    # Verify PortfolioConstructor with allow_shorting=True neither shorts positive non-finalists
    # nor selects non-finalists as longs even when SYM4's composite dropped below SYM5's
    meta_all = {
        s.symbol: AssetRiskMetadata(
            symbol=s.symbol, asset_type=AssetType.equity, sector=f"Sec{idx}"
        )
        for idx, s in enumerate(base_signals)
    }
    ctor = PortfolioConstructor(
        StrategyConfig(top_k_long=5, top_k_short=3, min_signal_threshold=0.05),
        RiskConfig(max_long_position_pct=0.25, max_short_position_pct=0.20),
    )
    dials_ls = ControlDials(allow_shorting=True, updated_at=as_of)
    tw = ctor.compute_target_weights(overlaid, dials_ls, as_of, meta_all)
    for non_fin_idx in range(5, 8):
        assert tw[f"SYM{non_fin_idx}"] == 0.0


def test_backtest_delisting_forced_exit_and_dynamic_pit_universe() -> None:
    n = 90
    spy_df = _make_bars_df(np.linspace(100.0, 115.0, n))
    a_df = _make_bars_df(np.linspace(100.0, 140.0, n))
    # DELIST trades strongly through bar 72, then disappears (delisted!)
    delist_df = _make_bars_df(np.linspace(100.0, 160.0, 73))

    universe = [
        UniverseSymbol(symbol="SPY", asset_type=AssetType.etf, sector="Broad"),
        UniverseSymbol(symbol="AAA", asset_type=AssetType.equity, sector="Tech"),
        UniverseSymbol(symbol="DELIST", asset_type=AssetType.equity, sector="Finance"),
    ]
    app_cfg = AppConfig(
        strategy=StrategyConfig(
            w_trend=1.0,
            w_pullback=0.0,
            w_news_sentiment=0.0,
            top_k_long=2,
            top_k_short=0,
            min_signal_threshold=0.0,
            rebalance_every_n_days=1,
            rebalance_tranches=4,  # Regression (CRITICAL #2): 4 tranches must NOT re-buy DELIST!
        ),
        risk=RiskConfig(
            max_long_position_pct=0.45,
            max_sector_gross_pct=0.50,
            delisting_haircut_pct=0.10,
        ),
        backtest=BacktestConfig(initial_capital=100_000.0),
    )
    engine = BacktestEngine(app_cfg, universe)
    d0 = pd.Timestamp(spy_df["timestamp"].iloc[0]).date()
    d_switch = pd.Timestamp(spy_df["timestamp"].iloc[85]).date()
    eligible = {
        d0: frozenset({"AAA", "DELIST"}),
        d_switch: frozenset({"AAA"}),
    }

    series = engine.run_with_series(
        {"SPY": spy_df, "AAA": a_df, "DELIST": delist_df},
        allow_shorting=False,
        use_news=False,
        eligible_symbols_by_date=eligible,
    )
    delist_trades = [
        t for t in series.result.trades if t.symbol == "DELIST" and t.exit_reason == "delisted_forced_exit"
    ]
    assert len(delist_trades) == 1
    delist_exit_ts = delist_trades[0].exit_time
    # Verify DELIST is never re-entered or traded after delisted_forced_exit even with rebalance_tranches=4
    trades_after_delist = [
        t for t in series.result.trades if t.symbol == "DELIST" and t.entry_time >= delist_exit_ts
    ]
    assert trades_after_delist == []
    last_px = float(delist_df["close"].iloc[-1])
    assert delist_trades[0].exit_price == pytest.approx(round(last_px * 0.90, 6), rel=1e-6)
    assert len(series.ew_universe_equities) == len(series.result.equity_curve)


def test_backtest_multi_tranche_delayed_reconstitution_and_vol_target_overlay() -> None:
    n = 90
    spy_df = _make_bars_df(np.linspace(100.0, 115.0, n))
    a_df = _make_bars_df(np.linspace(100.0, 145.0, n))
    b_df = _make_bars_df(np.linspace(100.0, 138.0, n))

    universe = [
        UniverseSymbol(symbol="SPY", asset_type=AssetType.etf, sector="Broad"),
        UniverseSymbol(symbol="AAA", asset_type=AssetType.equity, sector="Tech"),
        UniverseSymbol(symbol="BBB", asset_type=AssetType.equity, sector="Health"),
    ]
    app_cfg = AppConfig(
        strategy=StrategyConfig(
            w_trend=1.0,
            w_pullback=0.0,
            w_news_sentiment=0.0,
            top_k_long=2,
            top_k_short=0,
            min_signal_threshold=0.0,
            rebalance_every_n_days=2,
            rebalance_tranches=4,
            portfolio_vol_target_ann=0.06,  # Low vol target must govern blended 4-tranche portfolio!
        ),
        risk=RiskConfig(
            max_long_position_pct=0.50,
            max_sector_gross_pct=0.60,
            max_net_exposure=0.90,
            max_gross_exposure=0.90,
        ),
        backtest=BacktestConfig(initial_capital=100_000.0),
    )
    # First reconstitution date is bar 72 (after several empty-signal rebalance days at bars 65..71)
    d_late = pd.Timestamp(spy_df["timestamp"].iloc[72]).date()
    res = BacktestEngine(app_cfg, universe).run(
        {"SPY": spy_df, "AAA": a_df, "BBB": b_df},
        allow_shorting=False,
        use_news=False,
        eligible_symbols_by_date={d_late: frozenset({"AAA", "BBB"})},
    )
    # Before d_late, gross exposure is 0.0; on the first active rebalance at/after d_late,
    # all 4 tranches seed cleanly (not locked 3/4 in empty cash!) and respect portfolio_vol_target_ann
    active_pts = [pt for pt in res.equity_curve if pt.timestamp.date() >= d_late and pt.gross_exposure > 0.0]
    assert active_pts
    # First active day deploys full vol-targeted budget (well above 1/4 of target, and capped by vol target < 0.90)
    assert 0.20 < active_pts[0].gross_exposure < 0.85


def test_backtest_tranche_rebalancing_and_drawdown_cooldown_preserves_reporting_drawdown() -> None:
    n = 95
    spy_df = _make_bars_df(np.linspace(100.0, 112.0, n))
    s1_df = _make_bars_df(np.linspace(100.0, 135.0, n))
    # S2 rises until bar 68, then crashes -40% to trigger portfolio drawdown > 5%, then flattens
    s2_prices = np.concatenate([np.linspace(100.0, 145.0, 68), np.full(n - 68, 70.0)])
    s2_df = _make_bars_df(s2_prices)

    universe = [
        UniverseSymbol(symbol="SPY", asset_type=AssetType.etf, sector="Broad"),
        UniverseSymbol(symbol="S1", asset_type=AssetType.equity, sector="Tech"),
        UniverseSymbol(symbol="S2", asset_type=AssetType.equity, sector="Health"),
    ]
    base_strat = StrategyConfig(
        w_trend=1.0,
        w_pullback=0.0,
        w_news_sentiment=0.0,
        top_k_long=2,
        top_k_short=0,
        min_signal_threshold=0.0,
        rebalance_every_n_days=2,
        rebalance_tranches=2,
    )
    cfg_no_cooldown = AppConfig(
        strategy=base_strat,
        risk=RiskConfig(
            max_long_position_pct=0.50,
            max_sector_gross_pct=0.60,
            max_drawdown_halt_pct=0.05,
            daily_loss_halt_pct=0.02,
            max_drawdown_cooldown_days=None,
            stop_loss_atr_mult_long=0.5,
        ),
    )
    cfg_cooldown = cfg_no_cooldown.model_copy(
        update={
            "risk": cfg_no_cooldown.risk.model_copy(update={"max_drawdown_cooldown_days": 3})
        }
    )

    res_no_cd = BacktestEngine(cfg_no_cooldown, universe).run(
        {"SPY": spy_df, "S1": s1_df, "S2": s2_df},
        allow_shorting=False,
        use_news=False,
    )
    res_cd = BacktestEngine(cfg_cooldown, universe).run(
        {"SPY": spy_df, "S1": s1_df, "S2": s2_df},
        allow_shorting=False,
        use_news=False,
    )
    assert res_cd.metrics.total_orders_filled >= res_no_cd.metrics.total_orders_filled

    # Regression (HIGH #7): reporting EquityPoint.drawdown_pct must remain strictly negative whenever
    # equity < running peak equity, even after max_drawdown_cooldown_days resets risk_hwm!
    peak_eq = 0.0
    underwater_after_cooldown = 0
    for pt in res_cd.equity_curve:
        peak_eq = max(peak_eq, pt.equity)
        if pt.equity < peak_eq - 1.0:
            assert pt.drawdown_pct < 0.0
            underwater_after_cooldown += 1
    assert underwater_after_cooldown >= 5
