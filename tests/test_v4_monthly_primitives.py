from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from agy_ops.daemons.trader import TraderDaemon
from agy_trader.backtest.engine import BacktestEngine
from agy_trader.core.calendar import is_nyse_trading_day
from agy_trader.core.config import (
    AppConfig,
    BacktestConfig,
    EnvSettings,
    RiskConfig,
    StrategyConfig,
    load_app_config,
)
from agy_trader.core.models import (
    AssetRiskMetadata,
    AssetType,
    ControlDials,
    MacroPoint,
    SignalRow,
    UniverseSymbol,
)
from agy_trader.storage.db import StateDB
from agy_trader.strategy.features import FeatureEngine, compute_macro_regime_scale
from agy_trader.strategy.portfolio import PortfolioConstructor


def _make_business_bars(
    closes: np.ndarray,
    *,
    start_d: date = date(2024, 1, 2),
    high_mult: float = 1.005,
    low_mult: float = 0.995,
    volumes: np.ndarray | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cur = datetime(start_d.year, start_d.month, start_d.day, 21, 0, tzinfo=UTC)
    for idx, px in enumerate(closes):
        while cur.weekday() >= 5:
            cur += timedelta(days=1)
        c = float(px)
        v = float(volumes[idx]) if volumes is not None else 1_000_000.0
        rows.append(
            {
                "timestamp": cur,
                "open": c * 0.999,
                "high": c * high_mult,
                "low": c * low_mult,
                "close": c,
                "volume": v,
            }
        )
        cur += timedelta(days=1)
    return pd.DataFrame(rows)


def test_v4_config_defaults_preserve_v02_and_validate() -> None:
    cfg = StrategyConfig()
    assert cfg.rebalance_schedule == "every_n_days"
    assert cfg.regime_mode == "legacy_stacked"
    assert cfg.m2_w_raw_long == 0.0
    assert cfg.m2_w_intermed == 0.0
    assert cfg.m2_w_52w_high == 0.0
    assert cfg.m2_w_reg_slope == 0.0
    assert cfg.reg_slope_lookback == 126
    assert cfg.fip_gate_quantile is None
    assert cfg.max_stock_ann_vol is None
    assert cfg.trend_above_sma200_filter is False
    assert cfg.exclude_safe_havens_from_equity_slots is False

    # Legacy strategy.yaml still loads cleanly without error
    repo_config_dir = Path(__file__).resolve().parents[1] / "config"
    live_cfg = load_app_config(repo_config_dir)
    assert live_cfg.strategy.rebalance_schedule == "every_n_days"

    # V4 fields must require momentum_mode == 'm2_residual'
    for bad_kwargs in (
        {"m2_w_raw_long": 0.5},
        {"m2_w_intermed": 0.5},
        {"m2_w_52w_high": 0.5},
        {"m2_w_reg_slope": 0.5},
        {"m2_w_liquidity": 0.5},
        {"fip_gate_quantile": 0.25},
        {"max_stock_ann_vol": 0.35},
        {"trend_above_sma200_filter": True},
        {"vol_sizing_mode": "liquidity_pow"},
    ):
        with pytest.raises(ValueError, match="require momentum_mode == 'm2_residual'"):
            StrategyConfig(momentum_mode="legacy", **bad_kwargs)

    # Pure raw_long (with legacy M2 sub-weights zeroed) must validate cleanly
    raw_only = StrategyConfig(
        momentum_mode="m2_residual",
        m2_w_long=0.0,
        m2_w_med=0.0,
        m2_w_residual=0.0,
        m2_w_raw_long=1.0,
        regime_mode="none",
        rebalance_schedule="monthly_first",
    )
    assert raw_only.m2_w_raw_long == 1.0


def test_regime_modes_none_and_sma200_only_strict_pit() -> None:
    # Build 260 bars of SPY: first 210 uptrending around 100->140, then crash below SMA200
    up = np.linspace(100.0, 140.0, 210)
    down = np.linspace(139.0, 95.0, 50)
    spy_df = _make_business_bars(np.concatenate([up, down]))
    crash_ts = spy_df["timestamp"].iloc[-1].to_pydatetime()
    bull_ts = spy_df["timestamp"].iloc[205].to_pydatetime()

    # Future VIX/HY shocks on macro_points should be completely ignored by 'none' and 'sma200_only'
    macro = [
        MacroPoint(as_of_date=date(2020, 1, 1), series_id="VIXCLS", value=45.0),
    ]

    # 'none' always returns 1.0 even during crash & high VIX
    assert compute_macro_regime_scale(crash_ts, spy_df, macro, regime_mode="none") == 1.0

    # 'sma200_only' returns 1.0 during bull window even with high VIX
    assert (
        compute_macro_regime_scale(
            bull_ts,
            spy_df,
            macro,
            spy_sma200_defensive_scale=0.45,
            regime_mode="sma200_only",
        )
        == 1.0
    )

    # 'sma200_only' returns defensive scale strictly when last_close < SMA200
    assert (
        compute_macro_regime_scale(
            crash_ts,
            spy_df,
            macro,
            spy_sma200_defensive_scale=0.45,
            regime_mode="sma200_only",
        )
        == 0.45
    )

    # Zero lookahead: truncating spy_df to bull_ts gives identical scale as passing full df with as_of=bull_ts
    assert compute_macro_regime_scale(
        bull_ts,
        spy_df,
        macro,
        spy_sma200_defensive_scale=0.45,
        regime_mode="sma200_only",
    ) == compute_macro_regime_scale(
        bull_ts,
        spy_df.loc[spy_df["timestamp"] <= bull_ts],
        macro,
        spy_sma200_defensive_scale=0.45,
        regime_mode="sma200_only",
    )


def test_new_m2_signals_and_stage1_gates_ranking_and_causality() -> None:
    n = 280
    t = np.arange(n, dtype=float)

    # FAST_WINNER: Strong steady uptrend near 52w high, low vol
    c_winner = 100.0 * np.exp(0.0020 * t)
    # VOL_LOTTERY: Higher cumulative return via huge erratic spikes -> higher ann_vol (> 0.60)
    rng = np.random.default_rng(7)
    r_lottery = rng.normal(0.0025, 0.045, size=n)
    c_lottery = 100.0 * np.exp(np.cumsum(r_lottery))
    # DOWNTREND: monotonically drifting down (< SMA200)
    c_down = 200.0 * np.exp(-0.0018 * t)
    # LATE_FADE: rose strongly early (t-252..t-126) then flat/down recently -> below SMA200 or lower 52w high
    c_mid = np.concatenate([100.0 * np.exp(0.004 * np.arange(150)), np.linspace(180.0, 125.0, n - 150)])

    bars = {
        "SPY": _make_business_bars(100.0 * np.exp(0.0005 * t)),
        "WINN": _make_business_bars(c_winner),
        "LOTT": _make_business_bars(c_lottery),
        "DOWN": _make_business_bars(c_down),
        "LATE": _make_business_bars(c_mid),
    }
    as_of = bars["SPY"]["timestamp"].iloc[-1].to_pydatetime()

    # 1. Regression Slope * R2 puts smooth WINN above DOWN
    cfg_reg = StrategyConfig(
        momentum_mode="m2_residual",
        m2_w_long=0.0,
        m2_w_med=0.0,
        m2_w_residual=0.0,
        m2_w_reg_slope=1.0,
        reg_slope_lookback=126,
        w_trend=1.0,
        w_pullback=0.0,
        w_news_sentiment=0.0,
    )
    sigs_reg = {s.symbol: s for s in FeatureEngine(cfg_reg).build_signals(as_of, bars, universe_symbols={"WINN", "DOWN"})}
    assert sigs_reg["WINN"].trend_score > sigs_reg["DOWN"].trend_score
    assert all(np.isfinite(s.composite_score) for s in sigs_reg.values())

    # 2. 52w high proximity ranks WINN (~1.0) far above LATE (~0.69) and DOWN
    cfg_52w = StrategyConfig(
        momentum_mode="m2_residual",
        m2_w_long=0.0,
        m2_w_med=0.0,
        m2_w_residual=0.0,
        m2_w_52w_high=1.0,
        w_trend=1.0,
        w_pullback=0.0,
        w_news_sentiment=0.0,
    )
    sigs_52w = {s.symbol: s for s in FeatureEngine(cfg_52w).build_signals(as_of, bars, universe_symbols={"WINN", "LATE", "DOWN"})}
    assert sigs_52w["WINN"].trend_score > sigs_52w["LATE"].trend_score > sigs_52w["DOWN"].trend_score

    # 3. Intermediate momentum (t-252..t-126) scores LATE high (it rallied hard in first 150 bars) vs DOWN
    cfg_intermed = StrategyConfig(
        momentum_mode="m2_residual",
        m2_w_long=0.0,
        m2_w_med=0.0,
        m2_w_residual=0.0,
        m2_w_intermed=1.0,
        w_trend=1.0,
        w_pullback=0.0,
        w_news_sentiment=0.0,
    )
    sigs_int = {s.symbol: s for s in FeatureEngine(cfg_intermed).build_signals(as_of, bars, universe_symbols={"LATE", "DOWN"})}
    assert sigs_int["LATE"].trend_score > sigs_int["DOWN"].trend_score

    # 4. Stage-1 gates: max_stock_ann_vol excludes LOTT, trend_above_sma200_filter excludes DOWN,
    #    and fip_gate_quantile=0.50 strictly gates on fip_raw (smoother SMTH beats choppy CHOP even when CHOP has higher t_score)
    c_chop = np.empty(n, dtype=float)
    c_chop[0] = 100.0
    for i in range(1, n):
        c_chop[i] = c_chop[i - 1] * (1.032 if i % 4 == 0 else 0.992)
    c_smth = 100.0 * np.exp(0.0012 * t)
    bars_fip = {
        "SPY": bars["SPY"],
        "CHOP": _make_business_bars(c_chop),
        "SMTH": _make_business_bars(c_smth),
        "LOTT": bars["LOTT"],
        "DOWN": bars["DOWN"],
    }
    cfg_gated = StrategyConfig(
        momentum_mode="m2_residual",
        m2_w_long=0.0,
        m2_w_med=0.0,
        m2_w_residual=0.0,
        m2_w_raw_long=1.0,
        w_trend=1.0,
        w_pullback=0.0,
        w_news_sentiment=0.0,
        w_fip=0.0,
        max_stock_ann_vol=0.55,
        trend_above_sma200_filter=True,
        fip_gate_quantile=0.50,
    )
    sigs_gated = {
        s.symbol: s
        for s in FeatureEngine(cfg_gated).build_signals(
            as_of, bars_fip, universe_symbols={"CHOP", "SMTH", "LOTT", "DOWN"}
        )
    }
    assert sigs_gated["LOTT"].stage1_finalist is False
    assert sigs_gated["DOWN"].stage1_finalist is False
    # Prove t_scores favors CHOP (higher raw return), yet fip_gate_quantile keeps smoother SMTH
    assert sigs_gated["CHOP"].trend_score > sigs_gated["SMTH"].trend_score
    assert sigs_gated["SMTH"].stage1_finalist is True
    assert sigs_gated["CHOP"].stage1_finalist is False

    # 5. above_sma200 fails closed when < 200 bars exist and trend_above_sma200_filter=True
    bars_short = {sym: df.iloc[-120:].copy() for sym, df in bars.items()}
    sigs_short = {
        s.symbol: s
        for s in FeatureEngine(
            StrategyConfig(
                momentum_mode="m2_residual",
                m2_w_raw_long=1.0,
                m2_w_long=0.0,
                m2_w_med=0.0,
                m2_w_residual=0.0,
                w_trend=1.0,
                w_pullback=0.0,
                w_news_sentiment=0.0,
                trend_above_sma200_filter=True,
            )
        ).build_signals(as_of, bars_short, universe_symbols={"WINN", "DOWN"})
    }
    assert all(not s.stage1_finalist for s in sigs_short.values())

    # 6. Causal PIT truncation test across all new signals + fip_gate_quantile + above_sma200
    as_of_mid = bars_fip["SPY"]["timestamp"].iloc[260].to_pydatetime()
    mutated_bars: dict[str, pd.DataFrame] = {}
    trunc_bars: dict[str, pd.DataFrame] = {}
    for sym, df in bars_fip.items():
        m_df = df.copy()
        m_df.loc[m_df["timestamp"] > as_of_mid, ["open", "high", "low", "close"]] *= 4.0
        m_df.loc[m_df["timestamp"] > as_of_mid, "volume"] *= 50.0
        mutated_bars[sym] = m_df
        trunc_bars[sym] = df.loc[df["timestamp"] <= as_of_mid].copy()

    for test_cfg in (
        cfg_reg,
        cfg_52w,
        cfg_intermed,
        cfg_gated,
    ):
        s_mut = {
            s.symbol: s
            for s in FeatureEngine(test_cfg).build_signals(
                as_of_mid, mutated_bars, universe_symbols={"CHOP", "SMTH", "LOTT", "DOWN"}
            )
        }
        s_tr = {
            s.symbol: s
            for s in FeatureEngine(test_cfg).build_signals(
                as_of_mid, trunc_bars, universe_symbols={"CHOP", "SMTH", "LOTT", "DOWN"}
            )
        }
        assert set(s_mut.keys()) == set(s_tr.keys())
        for sym, tr_sig in s_tr.items():
            assert s_mut[sym].trend_score == pytest.approx(tr_sig.trend_score)
            assert s_mut[sym].composite_score == pytest.approx(tr_sig.composite_score)
            assert s_mut[sym].mddv_60d == pytest.approx(tr_sig.mddv_60d)
            assert s_mut[sym].stage1_finalist == tr_sig.stage1_finalist


def test_exclude_safe_havens_from_equity_slots() -> None:
    now = datetime(2025, 6, 2, 20, 0, tzinfo=UTC)
    meta = {
        "AAPL": AssetRiskMetadata(symbol="AAPL", asset_type=AssetType.equity, sector="Technology"),
        "GLD": AssetRiskMetadata(symbol="GLD", asset_type=AssetType.etf, sector="Commodities"),
        "TLT": AssetRiskMetadata(symbol="TLT", asset_type=AssetType.etf, sector="Fixed Income"),
    }
    signals = [
        SignalRow(
            as_of=now,
            symbol="GLD",
            trend_score=2.5,
            pullback_score=0.0,
            news_score=0.0,
            news_velocity_z=0.0,
            composite_score=2.5,
            macro_regime_scale=0.50,
            atr_pct=0.01,
            ann_vol=0.14,
            ret_5d=0.01,
            safe_haven_trend_ok=True,
            safe_haven_strength=1.5,
        ),
        SignalRow(
            as_of=now,
            symbol="AAPL",
            trend_score=1.2,
            pullback_score=0.0,
            news_score=0.0,
            news_velocity_z=0.0,
            composite_score=1.2,
            macro_regime_scale=0.50,
            atr_pct=0.015,
            ann_vol=0.22,
            ret_5d=0.02,
        ),
    ]
    dials = ControlDials(updated_at=now)
    cfg = StrategyConfig(
        momentum_mode="m2_residual",
        top_k_long=1,
        top_k_short=0,
        min_signal_threshold=0.0,
        exclude_safe_havens_from_equity_slots=True,
        safe_haven_rotation_fraction=0.0,
    )
    risk = RiskConfig(max_long_position_pct=0.50, max_net_exposure=1.0, max_sector_gross_pct=1.0)
    w = PortfolioConstructor(cfg, risk).compute_target_weights(signals, dials, now, meta)
    assert w.get("GLD", 0.0) == 0.0
    assert w.get("AAPL", 0.0) > 0.0

    # With safe_haven_rotation_fraction=1.0 in bear macro_scale=0.50, GLD enters via rotation sleeve (not equity slot)
    cfg_rot = cfg.model_copy(update={"safe_haven_rotation_fraction": 1.0, "waterfill_weights": True})
    w_rot = PortfolioConstructor(cfg_rot, risk).compute_target_weights(signals, dials, now, meta)
    assert w_rot["AAPL"] > 0.0
    assert w_rot["GLD"] > 0.0

    # Short slots must also exclude safe-haven ETFs when exclude_safe_havens_from_equity_slots=True
    meta_sh = {
        **meta,
        "MSFT": AssetRiskMetadata(
            symbol="MSFT",
            asset_type=AssetType.equity,
            sector="Technology",
            shortable=True,
            easy_to_borrow=True,
        ),
    }
    short_sigs = [
        SignalRow(
            as_of=now,
            symbol="GLD",
            trend_score=-2.5,
            pullback_score=0.0,
            news_score=0.0,
            news_velocity_z=0.0,
            composite_score=-2.5,
            macro_regime_scale=0.50,
            atr_pct=0.01,
            ann_vol=0.14,
            ret_5d=-0.01,
        ),
        SignalRow(
            as_of=now,
            symbol="MSFT",
            trend_score=-1.2,
            pullback_score=0.0,
            news_score=0.0,
            news_velocity_z=0.0,
            composite_score=-1.2,
            macro_regime_scale=0.50,
            atr_pct=0.015,
            ann_vol=0.22,
            ret_5d=-0.02,
        ),
        signals[1],
    ]
    cfg_sh = cfg.model_copy(
        update={
            "top_k_long": 1,
            "top_k_short": 1,
            "short_only_in_bear_regime": True,
        }
    )
    risk_sh = RiskConfig(
        max_long_position_pct=0.50,
        max_short_position_pct=0.20,
        min_net_exposure=-0.50,
        max_net_exposure=1.0,
        max_gross_exposure=1.0,
        max_sector_gross_pct=1.0,
    )
    w_sh = PortfolioConstructor(cfg_sh, risk_sh).compute_target_weights(short_sigs, dials, now, meta_sh)
    assert w_sh.get("GLD", 0.0) == 0.0
    assert w_sh.get("MSFT", 0.0) < 0.0

    # End-to-end BacktestEngine verification: safe havens never enter equity long slots
    n = 120
    t = np.arange(n, dtype=float)
    bt_bars = {
        "SPY": _make_business_bars(100.0 * np.exp(0.0003 * t)),
        "GLD": _make_business_bars(100.0 * np.exp(0.0040 * t)),
        "TLT": _make_business_bars(100.0 * np.exp(0.0035 * t)),
        "EQ1": _make_business_bars(100.0 * np.exp(0.0010 * t)),
    }
    bt_univ = [
        UniverseSymbol(symbol="GLD", asset_type=AssetType.etf, sector="Commodities"),
        UniverseSymbol(symbol="TLT", asset_type=AssetType.etf, sector="Fixed Income"),
        UniverseSymbol(symbol="EQ1", asset_type=AssetType.equity, sector="Technology"),
    ]
    bt_cfg = AppConfig(
        strategy=StrategyConfig(
            momentum_mode="m2_residual",
            m2_w_long=0.0,
            m2_w_med=0.0,
            m2_w_residual=0.0,
            m2_w_raw_long=1.0,
            w_trend=1.0,
            w_pullback=0.0,
            w_news_sentiment=0.0,
            top_k_long=2,
            top_k_short=0,
            regime_mode="none",
            exclude_safe_havens_from_equity_slots=True,
            rebalance_schedule="monthly_first",
        ),
        risk=RiskConfig(max_long_position_pct=0.90, max_sector_gross_pct=0.95),
        backtest=BacktestConfig(initial_capital=100_000.0),
    )
    bt_res = BacktestEngine(bt_cfg, bt_univ).run(bt_bars, allow_shorting=False, use_news=False)
    traded_syms = {tr.symbol for tr in bt_res.trades}
    assert "GLD" not in traded_syms
    assert "TLT" not in traded_syms


def test_monthly_first_and_monthly_last_rebalance_schedules() -> None:
    # Build 260 trading days starting 2024-01-02 (approx 12 months -> ~8-9 active months after 65-bar warmup)
    n = 260
    t = np.arange(n, dtype=float)
    spy_df = _make_business_bars(100.0 * np.exp(0.0004 * t))
    # Zig-zag leadership mid-year so rebalance dates fire orders cleanly
    c_aaa = np.concatenate([100.0 * np.exp(0.0020 * np.arange(140)), 100.0 * np.exp(0.0020 * 140) * np.exp(-0.0025 * np.arange(n - 140))])
    c_bbb = np.concatenate([100.0 * np.exp(0.0001 * np.arange(140)), 100.0 * np.exp(0.0001 * 140) * np.exp(0.0035 * np.arange(n - 140))])
    bars = {
        "SPY": spy_df,
        "AAA": _make_business_bars(c_aaa),
        "BBB": _make_business_bars(c_bbb),
        "CCC": _make_business_bars(100.0 * np.exp(0.0006 * t)),
        "DDD": _make_business_bars(100.0 * np.exp(-0.0010 * t)),
    }
    univ = [
        UniverseSymbol(symbol="AAA", asset_type=AssetType.equity, sector="Technology"),
        UniverseSymbol(symbol="BBB", asset_type=AssetType.equity, sector="Healthcare"),
        UniverseSymbol(symbol="CCC", asset_type=AssetType.equity, sector="Financials"),
        UniverseSymbol(symbol="DDD", asset_type=AssetType.equity, sector="Industrials"),
    ]
    base_risk = RiskConfig(
        max_long_position_pct=0.60,
        max_net_exposure=1.0,
        max_gross_exposure=1.0,
        max_sector_gross_pct=0.80,
        max_drawdown_halt_pct=0.35,
        daily_loss_halt_pct=0.08,
        stop_loss_atr_mult_long=10.0,
    )
    common_strat: dict[str, Any] = {
        "momentum_mode": "m2_residual",
        "m2_w_long": 0.0,
        "m2_w_med": 0.0,
        "m2_w_residual": 0.0,
        "m2_w_raw_long": 1.0,
        "w_trend": 1.0,
        "w_pullback": 0.0,
        "w_news_sentiment": 0.0,
        "top_k_long": 2,
        "top_k_short": 0,
        "min_signal_threshold": 0.0,
        "exit_signal_threshold": -0.5,
        "regime_mode": "none",
        "vol_sizing_mode": "equal",
        "rebalance_deadband_pct": 0.0,
    }

    cfg_first = AppConfig(
        strategy=StrategyConfig(rebalance_schedule="monthly_first", **common_strat),
        risk=base_risk,
        backtest=BacktestConfig(initial_capital=100_000.0),
    )
    cfg_last = AppConfig(
        strategy=StrategyConfig(rebalance_schedule="monthly_last", **common_strat),
        risk=base_risk,
        backtest=BacktestConfig(initial_capital=100_000.0),
    )
    cfg_daily = AppConfig(
        strategy=StrategyConfig(rebalance_schedule="every_n_days", rebalance_every_n_days=1, **common_strat),
        risk=base_risk,
        backtest=BacktestConfig(initial_capital=100_000.0),
    )

    res_first = BacktestEngine(cfg_first, univ).run(bars, allow_shorting=False, use_news=False)
    res_last = BacktestEngine(cfg_last, univ).run(bars, allow_shorting=False, use_news=False)
    res_daily = BacktestEngine(cfg_daily, univ).run(bars, allow_shorting=False, use_news=False)

    assert res_first.metrics.total_orders_filled >= 2
    assert res_last.metrics.total_orders_filled >= 2
    assert res_first.metrics.annual_turnover < res_daily.metrics.annual_turnover or (
        res_first.metrics.total_orders_filled <= res_daily.metrics.total_orders_filled
    )
    # Monthly schedules deploy capital cleanly on day 0
    assert res_first.equity_curve[0].gross_exposure > 0.90
    assert res_last.equity_curve[0].gross_exposure > 0.90
    # Monthly-first and monthly-last must rebalance on distinct non-day-0 dates -> strict difference
    ny_tz = ZoneInfo("America/New_York")
    first_days = {tr.entry_time.astimezone(ny_tz).date() for tr in res_first.trades}
    last_days = {tr.entry_time.astimezone(ny_tz).date() for tr in res_last.trades}
    assert first_days != last_days

    # Verify next_open schedule dates differ cleanly and match month boundaries on subsequent months
    cfg_first_open = cfg_first.model_copy(
        update={"backtest": BacktestConfig(initial_capital=100_000.0, execution_timing="next_open")}
    )
    cfg_last_open = cfg_last.model_copy(
        update={"backtest": BacktestConfig(initial_capital=100_000.0, execution_timing="next_open")}
    )
    res_first_open = BacktestEngine(cfg_first_open, univ).run(bars, allow_shorting=False, use_news=False)
    res_last_open = BacktestEngine(cfg_last_open, univ).run(bars, allow_shorting=False, use_news=False)
    open_first_days = {tr.entry_time.astimezone(ny_tz).date() for tr in res_first_open.trades}
    open_last_days = {tr.entry_time.astimezone(ny_tz).date() for tr in res_last_open.trades}
    assert open_first_days != open_last_days


def test_liquidity_mddv_factor_and_sizing_and_adaptive_63_126() -> None:
    cfg = StrategyConfig()
    assert cfg.m2_w_liquidity == 0.0
    assert cfg.liquidity_weight_power == 0.5

    with pytest.raises(ValueError, match="require momentum_mode == 'm2_residual'"):
        StrategyConfig(momentum_mode="legacy", m2_w_liquidity=0.3)

    n = 150
    t = np.arange(n, dtype=float)
    # MEGA & SMAL have identical price paths, except MEGA has 100x higher trailing dollar volume
    c_same = 100.0 * np.exp(0.0010 * t)
    v_mega = np.full(n, 10_000_000.0)
    # Inject future volume spike after day 140 to verify strict PIT slicing (as_of=day 139)
    v_smal = np.full(n, 100_000.0)
    v_smal[140:] = 999_999_999.0

    # FAST_TURN dropped early then surged strongly over last 63 days
    c_turn = np.concatenate([100.0 * np.exp(-0.001 * np.arange(70)), 93.0 * np.exp(0.003 * np.arange(n - 70))])
    v_turn = np.full(n, 1_000_000.0)

    bars = {
        "SPY": _make_business_bars(100.0 * np.exp(0.0005 * t)),
        "MEGA": _make_business_bars(c_same, volumes=v_mega),
        "SMAL": _make_business_bars(c_same, volumes=v_smal),
        "TURN": _make_business_bars(c_turn, volumes=v_turn),
    }
    as_of = bars["SPY"]["timestamp"].iloc[139].to_pydatetime()

    cfg_liq = StrategyConfig(
        momentum_mode="m2_residual",
        m2_long_lookback=126,
        m2_long_skip=0,
        m2_med_lookback=63,
        m2_med_skip=0,
        m2_w_long=0.25,
        m2_w_med=0.45,
        m2_w_residual=0.0,
        m2_w_liquidity=0.30,
        w_trend=0.80,
        w_fip=0.20,
        w_pullback=0.0,
        w_news_sentiment=0.0,
        top_k_long=3,
        top_k_short=0,
        min_signal_threshold=0.0,
        exit_signal_threshold=-0.5,
        vol_sizing_mode="liquidity_pow",
        liquidity_weight_power=0.5,
    )
    sigs = {
        s.symbol: s
        for s in FeatureEngine(cfg_liq).build_signals(
            as_of, bars, universe_symbols={"MEGA", "SMAL", "TURN"}
        )
    }
    # Strictly PIT: SMAL must NOT see day >= 140 volume spike
    assert sigs["MEGA"].mddv_60d > 50.0 * sigs["SMAL"].mddv_60d
    assert sigs["SMAL"].mddv_60d < 15_000_000.0
    assert sigs["MEGA"].composite_score > sigs["SMAL"].composite_score

    # Verify adaptive 63+126 ranking without liquidity works with skip=0
    cfg_adapt = cfg_liq.model_copy(update={"m2_w_liquidity": 0.0})
    sigs_adapt = {
        s.symbol: s
        for s in FeatureEngine(cfg_adapt).build_signals(
            as_of, bars, universe_symbols={"MEGA", "TURN"}
        )
    }
    assert sigs_adapt["TURN"].trend_score > sigs_adapt["MEGA"].trend_score

    # Portfolio weight test for vol_sizing_mode="liquidity_pow"
    meta = {
        sym: AssetRiskMetadata(symbol=sym, asset_type=AssetType.equity, sector=f"Sec{i}")
        for i, sym in enumerate(("BIG", "MID"))
    }
    dials = ControlDials(updated_at=as_of)
    risk = RiskConfig(
        max_long_position_pct=0.95,
        max_net_exposure=1.0,
        max_gross_exposure=1.0,
        max_sector_gross_pct=0.98,
    )
    pair_sigs = [
        SignalRow(
            as_of=as_of,
            symbol="BIG",
            trend_score=1.0,
            pullback_score=0.0,
            news_score=0.0,
            news_velocity_z=0.0,
            composite_score=1.0,
            macro_regime_scale=1.0,
            atr_pct=0.015,
            ann_vol=0.22,
            ret_5d=0.01,
            mddv_60d=100_000_000.0,
        ),
        SignalRow(
            as_of=as_of,
            symbol="MID",
            trend_score=1.0,
            pullback_score=0.0,
            news_score=0.0,
            news_velocity_z=0.0,
            composite_score=1.0,
            macro_regime_scale=1.0,
            atr_pct=0.015,
            ann_vol=0.22,
            ret_5d=0.01,
            mddv_60d=1_000_000.0,
        ),
    ]
    weights = PortfolioConstructor(cfg_liq, risk).compute_target_weights(
        pair_sigs, dials, as_of, meta
    )
    assert weights["BIG"] > weights["MID"] > 0.0
    assert pytest.approx(weights["BIG"] / weights["MID"], rel=1e-4) == 10.0

    # Zero-MDDV outlier must get the 1e-3 floor weight, strictly below a real $500k MDDV symbol
    meta_z = {
        sym: AssetRiskMetadata(symbol=sym, asset_type=AssetType.equity, sector=f"Sec{i}")
        for i, sym in enumerate(("SMALL_LIQ", "ZERO"))
    }
    pair_zero = [
        pair_sigs[0].model_copy(update={"symbol": "SMALL_LIQ", "mddv_60d": 500_000.0}),
        pair_sigs[0].model_copy(update={"symbol": "ZERO", "mddv_60d": 0.0}),
    ]
    risk_uncapped = RiskConfig(
        max_long_position_pct=0.99,
        max_net_exposure=1.0,
        max_gross_exposure=1.0,
        max_sector_gross_pct=1.0,
    )
    w_zero = PortfolioConstructor(cfg_liq, risk_uncapped).compute_target_weights(
        pair_zero, dials, as_of, meta_z
    )
    assert w_zero["SMALL_LIQ"] > w_zero["ZERO"] > 0.0
    assert pytest.approx(w_zero["SMALL_LIQ"] / w_zero["ZERO"], rel=1e-4) == (0.5 / 1e-3) ** 0.5


def test_run_cpcv_respects_start_end_and_eligible_schedule() -> None:
    n = 420
    t = np.arange(n, dtype=float)
    spy_df = _make_business_bars(100.0 * np.exp(0.0004 * t), start_d=date(2020, 1, 2))
    all_dates = [ts.date() for ts in spy_df["timestamp"]]
    bars = {
        "SPY": spy_df,
        "AAA": _make_business_bars(100.0 * np.exp(0.0012 * t), start_d=date(2020, 1, 2)),
        "BBB": _make_business_bars(100.0 * np.exp(0.0008 * t), start_d=date(2020, 1, 2)),
    }
    univ = [
        UniverseSymbol(symbol="AAA", asset_type=AssetType.equity, sector="Technology"),
        UniverseSymbol(symbol="BBB", asset_type=AssetType.equity, sector="Healthcare"),
    ]
    sched_start_idx = 220
    sched = {d: frozenset({"AAA", "BBB"}) for d in all_dates[sched_start_idx:]}
    cfg = AppConfig(
        strategy=StrategyConfig(
            momentum_mode="m2_residual",
            m2_w_long=0.0,
            m2_w_med=0.0,
            m2_w_residual=0.0,
            m2_w_raw_long=1.0,
            w_trend=1.0,
            w_pullback=0.0,
            w_news_sentiment=0.0,
            top_k_long=2,
            top_k_short=0,
            regime_mode="none",
            rebalance_schedule="monthly_first",
        ),
        risk=RiskConfig(max_long_position_pct=0.60, max_sector_gross_pct=0.80),
        backtest=BacktestConfig(initial_capital=100_000.0),
    )
    eng = BacktestEngine(cfg, univ)
    cpcv = eng.run_cpcv(
        bars,
        eligible_symbols_by_date=sched,
        start_date=all_dates[150],
        end_date=all_dates[400],
        n_groups=4,
        k_test_groups=2,
        purge_days=5,
        embargo_days=2,
        allow_shorting=False,
        use_news=False,
    )
    assert cpcv.n_paths == 6
    sched_first = all_dates[sched_start_idx]
    end_bound = all_dates[400]
    for fold in cpcv.fold_metrics:
        assert date.fromisoformat(fold.start_date) >= sched_first
        assert date.fromisoformat(fold.end_date) <= end_bound
        assert fold.total_orders_filled > 0


def test_trader_daemon_active_rebalance_checkpoint_monthly_schedules(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    config_dir.mkdir()
    db = StateDB(data_dir / "state.db")
    settings = EnvSettings(
        _env_file=None,
        alpaca_paper=True,
        alpaca_api_key="test_key",
        alpaca_secret_key="test_secret",
    )
    daemon = TraderDaemon(
        settings,
        data_dir,
        config_dir,
        db=db,
        network_check_fn=lambda: True,
    )
    ny_tz = ZoneInfo("America/New_York")

    def _check(sched: str, d: date) -> str | None:
        dt_et = datetime(d.year, d.month, d.day, 15, 45, tzinfo=ny_tz)
        dt_utc = dt_et.astimezone(UTC)
        cfg = AppConfig(
            strategy=StrategyConfig(
                rebalance_checkpoints_et=["15:40"],
                rebalance_schedule=sched,  # type: ignore[arg-type]
            )
        )
        return daemon._active_rebalance_checkpoint(
            now_utc=dt_utc,
            now_et=dt_et,
            next_close_utc=None,
            app_cfg=cfg,
        )

    # 2026-01-01 is NYSE holiday, 2026-01-02 is first trading day of Jan 2026
    assert not is_nyse_trading_day(date(2026, 1, 1))
    assert _check("monthly_first", date(2026, 1, 1)) is None
    assert _check("monthly_first", date(2026, 1, 2)) == "15:40"
    assert _check("monthly_first", date(2026, 1, 5)) is None

    # 2026-01-30 is Friday (last NYSE trading day of Jan 2026; Jan 31 is Saturday)
    assert _check("monthly_last", date(2026, 1, 29)) is None
    assert _check("monthly_last", date(2026, 1, 30)) == "15:40"

    # Legacy every_n_days=1 still triggers on any trading day
    assert _check("every_n_days", date(2026, 1, 5)) == "15:40"


