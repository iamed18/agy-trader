from __future__ import annotations

import copy
import logging
import math
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from agy_trader.backtest.engine import BacktestEngine, BacktestRunSeries
from agy_trader.backtest.metrics import (
    BacktestMetrics,
    CPCVSummary,
    compute_deflated_sharpe_ratio,
)
from agy_trader.backtest.sp500_cohorts import atomic_write_text
from agy_trader.backtest.v3_benchmark import (
    _calc_excess_sharpe,
    _calc_series_cagr_pct,
    _compute_annual_returns,
    _ensure_matplotlib_agg,
    _extract_spy_equities,
)
from agy_trader.backtest.v3_universe import (
    HISTORICAL_DELISTED_SECTORS,
    SECTOR_TO_ETF,
    V3_CORE_ETFS,
    V3EquityCandidate,
    build_dynamic_pit_universe_schedule,
)
from agy_trader.core.config import (
    AppConfig,
    BacktestConfig,
    LLMQuotaConfig,
    RiskConfig,
    StrategyConfig,
)
from agy_trader.core.models import AssetType, MacroPoint, UniverseSymbol, ensure_utc

logger = logging.getLogger(__name__)


class V4CandidateSpec(BaseModel):
    """Specification for a candidate configuration evaluated in V4 monthly research."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    candidate_id: str
    name: str
    description: str
    strategy_overrides: dict[str, Any] = Field(default_factory=dict)
    risk_overrides: dict[str, Any] = Field(default_factory=dict)
    is_baseline: bool = False


class V4BenchmarkSummary(BaseModel):
    """Windowed benchmark summary metrics for SPY and Equal-Weight indices."""

    model_config = ConfigDict(allow_inf_nan=False)

    benchmark_id: str
    name: str
    is_cagr_pct: float
    is_sharpe: float
    is_excess_sharpe: float
    is_max_drawdown_pct: float
    oos_cagr_pct: float
    oos_sharpe: float
    oos_excess_sharpe: float
    oos_max_drawdown_pct: float
    full_cagr_pct: float
    full_sharpe: float
    full_excess_sharpe: float
    full_max_drawdown_pct: float
    annual_returns: dict[str, float] = Field(default_factory=dict)


class V4CandidateRow(BaseModel):
    """IS, OOS, and Full 10Y comparative results for one V4 strategy candidate."""

    model_config = ConfigDict(allow_inf_nan=False)

    candidate_id: str
    name: str
    description: str
    is_baseline: bool = False
    strategy_overrides: dict[str, Any] = Field(default_factory=dict)
    risk_overrides: dict[str, Any] = Field(default_factory=dict)
    is_metrics: BacktestMetrics
    oos_metrics: BacktestMetrics
    full_metrics: BacktestMetrics
    dsr_all_trials: float = 0.0
    dsr_shortlist: float = 0.0
    dsr_oos_shortlist: float = 0.0
    annual_returns: dict[str, dict[str, float]] = Field(default_factory=dict)


class V4StressRow(BaseModel):
    """Stress & execution-realism check row evaluated across IS, OOS, and Full 10Y."""

    model_config = ConfigDict(allow_inf_nan=False)

    stress_id: str
    label: str
    universe_mode: str
    top_n: int
    execution_timing: str
    equity_slippage_bps: float
    delisting_haircut_pct: float
    rebalance_schedule: str
    is_metrics: BacktestMetrics
    oos_metrics: BacktestMetrics
    full_metrics: BacktestMetrics


class V4PlateauCell(BaseModel):
    """One grid point in the top_k_long x hold_rank_buffer stability plateau."""

    model_config = ConfigDict(allow_inf_nan=False)

    top_k_long: int
    hold_rank_buffer: int
    full_cagr_pct: float
    full_sharpe: float
    full_excess_sharpe: float
    full_max_drawdown_pct: float
    full_turnover: float
    oos_cagr_pct: float
    oos_sharpe: float
    oos_excess_sharpe: float
    oos_max_drawdown_pct: float


class V4MonthlyResearchReport(BaseModel):
    """Complete serialized report for V4 monthly index-beating research."""

    model_config = ConfigDict(allow_inf_nan=False)

    pool_total: int
    bars_loaded: int
    core_etfs_loaded: int
    delisted_count: int
    reconstitution_months: int
    avg_active_universe_size: float
    raw_bar_start: str
    raw_bar_end: str
    is_window: tuple[str, str]
    oos_window: tuple[str, str]
    full_window: tuple[str, str]
    n_is_trials_cumulative: int
    n_shortlist: int
    champion_id: str
    selection_reason: str
    benchmarks: list[V4BenchmarkSummary] = Field(default_factory=list)
    candidates: list[V4CandidateRow] = Field(default_factory=list)
    stress_rows: list[V4StressRow] = Field(default_factory=list)
    cpcv_full_summary: CPCVSummary | None = None
    plateau_cells: list[V4PlateauCell] = Field(default_factory=list)
    plot_paths: list[str] = Field(default_factory=list)


def _v4_clean_strategy_base() -> dict[str, Any]:
    """Explicit clean baseline dictionary isolating V4 from legacy v0.2 defaults."""
    return {
        "rebalance_checkpoints_et": ["15:40"],
        "momentum_mode": "m2_residual",
        "m2_long_lookback": 126,
        "m2_long_skip": 0,
        "m2_med_lookback": 63,
        "m2_med_skip": 0,
        "m2_w_long": 0.25,
        "m2_w_med": 0.45,
        "m2_w_residual": 0.0,
        "m2_w_raw_long": 0.0,
        "m2_w_intermed": 0.0,
        "m2_w_52w_high": 0.0,
        "m2_w_reg_slope": 0.0,
        "m2_w_liquidity": 0.30,
        "reg_slope_lookback": 126,
        "w_trend": 0.80,
        "w_pullback": 0.0,
        "w_news_sentiment": 0.0,
        "w_insider": 0.0,
        "news_half_life_hours": 48.0,
        "news_lookback_hours": 120.0,
        "top_k_long": 25,
        "top_k_short": 0,
        "min_signal_threshold": 0.0,
        "exit_signal_threshold": -0.5,
        "rebalance_deadband_pct": 0.015,
        "allow_fractional_longs": True,
        "rebalance_every_n_days": 1,
        "rebalance_schedule": "monthly_first",
        "hold_rank_buffer": 8,
        "abs_momentum_filter": False,
        "use_rf_hurdle_for_abs_momentum": True,
        "short_only_in_bear_regime": True,
        "max_pair_corr_60d": None,
        "corr_penalty_eta": 0.0,
        "score_tilt_power": 0.0,
        "waterfill_weights": True,
        "exempt_safe_havens_from_macro_scale": False,
        "safe_haven_rotation_fraction": 0.0,
        "exclude_safe_havens_from_equity_slots": True,
        "spy_sma200_defensive_scale": None,
        "spy_ema50_dual_scale": None,
        "spy_vol_surge_scale": None,
        "spy_ret10_crash_scale": None,
        "use_sector_residual": False,
        "w_fip": 0.20,
        "w_jump_penalty": 0.0,
        "fip_gate_quantile": None,
        "max_stock_ann_vol": None,
        "trend_above_sma200_filter": False,
        "vol_sizing_mode": "liquidity_pow",
        "liquidity_weight_power": 0.5,
        "vol_sizing_floor": 0.22,
        "vol_sizing_cap": 0.50,
        "portfolio_vol_target_ann": None,
        "rebalance_tranches": 1,
        "regime_mode": "none",
        "w_pead": 0.0,
    }


def _v4_clean_risk_base() -> dict[str, Any]:
    return {
        "max_long_position_pct": 0.12,
        "max_short_position_pct": 0.0,
        "max_gross_exposure": 1.00,
        "min_net_exposure": 0.0,
        "max_net_exposure": 1.00,
        "max_sector_gross_pct": 0.45,
        "daily_loss_halt_pct": 0.08,
        "max_drawdown_halt_pct": 0.35,
        "max_drawdown_cooldown_days": 20,
        "delisting_haircut_pct": 0.10,
        "earnings_blackout_days": None,
        "restatement_blackout_days": 10,
        "stop_loss_atr_mult_long": 10.0,
        "stop_loss_atr_mult_short": 3.0,
        "max_data_staleness_minutes": 15,
    }


def build_v4_champion_app_config(
    *,
    execution_timing: Literal["same_close", "next_open"] = "same_close",
    equity_slippage_bps: float = 10.0,
    etf_slippage_bps: float = 5.0,
) -> AppConfig:
    """Return the validated AppConfig for the C2_LiquidAdaptive V4 champion."""
    return AppConfig(
        strategy=StrategyConfig.model_validate(_v4_clean_strategy_base()),
        risk=RiskConfig.model_validate(_v4_clean_risk_base()),
        backtest=BacktestConfig(
            initial_capital=100_000.0,
            etf_slippage_bps=etf_slippage_bps,
            equity_slippage_bps=equity_slippage_bps,
            short_borrow_apr=0.015,
            sec_sell_fee_bps=0.28,
            risk_free_rate_apr=0.045,
            cash_yield_apr=0.045,
            use_fred_tbill_yield=True,
            execution_timing=execution_timing,
            benchmark_symbol="SPY",
        ),
        llm_quota=LLMQuotaConfig(),
    )


def _build_v02_baseline_overrides() -> tuple[dict[str, Any], dict[str, Any]]:
    """Mirror live v0.2 parameters without relying on external config/strategy.yaml paths."""
    strat = {
        "rebalance_checkpoints_et": ["15:40"],
        "momentum_mode": "m2_residual",
        "m2_long_lookback": 252,
        "m2_long_skip": 21,
        "m2_med_lookback": 126,
        "m2_med_skip": 10,
        "m2_w_long": 0.50,
        "m2_w_med": 0.25,
        "m2_w_residual": 0.25,
        "m2_w_liquidity": 0.0,
        "w_trend": 0.80,
        "w_pullback": 0.0,
        "w_news_sentiment": 0.20,
        "top_k_long": 6,
        "top_k_short": 0,
        "min_signal_threshold": 0.0,
        "exit_signal_threshold": -0.4,
        "rebalance_deadband_pct": 0.04,
        "rebalance_every_n_days": 1,
        "rebalance_schedule": "every_n_days",
        "hold_rank_buffer": 5,
        "abs_momentum_filter": True,
        "use_rf_hurdle_for_abs_momentum": True,
        "short_only_in_bear_regime": True,
        "max_pair_corr_60d": 0.70,
        "corr_penalty_eta": 0.30,
        "score_tilt_power": 0.15,
        "waterfill_weights": True,
        "exempt_safe_havens_from_macro_scale": False,
        "safe_haven_rotation_fraction": 0.0,
        "exclude_safe_havens_from_equity_slots": False,
        "spy_sma200_defensive_scale": 0.60,
        "spy_ema50_dual_scale": 0.75,
        "spy_vol_surge_scale": 0.65,
        "spy_vol_surge_thresh": 1.65,
        "spy_ret10_crash_thresh": -0.055,
        "spy_ret10_crash_scale": 0.60,
        "use_sector_residual": False,
        "w_fip": 0.0,
        "w_jump_penalty": 0.0,
        "vol_sizing_mode": "inv_atr",
        "regime_mode": "legacy_stacked",
        "portfolio_vol_target_ann": None,
        "rebalance_tranches": 1,
    }
    risk = {
        "max_long_position_pct": 0.20,
        "max_short_position_pct": 0.06,
        "max_gross_exposure": 1.00,
        "min_net_exposure": -0.35,
        "max_net_exposure": 1.00,
        "max_sector_gross_pct": 0.35,
        "daily_loss_halt_pct": 0.035,
        "max_drawdown_halt_pct": 0.14,
        "max_drawdown_cooldown_days": None,
        "delisting_haircut_pct": 0.10,
        "earnings_blackout_days": 2,
        "stop_loss_atr_mult_long": 3.5,
        "stop_loss_atr_mult_short": 3.0,
    }
    return strat, risk


def build_v4_candidate_specs() -> list[V4CandidateSpec]:
    """Return the locked 3 IS-selected V4 monthly candidates alongside the v0.2 baseline."""
    c2_strat = _v4_clean_strategy_base()
    c2_risk = _v4_clean_risk_base()

    c1_strat = copy.deepcopy(c2_strat)
    c1_strat.update(
        {
            "m2_w_long": 0.35,
            "m2_w_med": 0.65,
            "m2_w_liquidity": 0.0,
            "top_k_long": 20,
            "hold_rank_buffer": 15,
            "vol_sizing_mode": "equal",
        }
    )

    c3_strat = copy.deepcopy(c2_strat)
    c3_strat.update({"regime_mode": "daniel_moskowitz"})

    t0_strat, t0_risk = _build_v02_baseline_overrides()

    return [
        V4CandidateSpec(
            candidate_id="T0",
            name="v0.2 Daily Baseline",
            description="Daily-rebalanced v0.2 config (k=6, 252-21d M2, 3.5x ATR stop, -14% halt) on PIT S&P 500 Top-200.",
            strategy_overrides=t0_strat,
            risk_overrides=t0_risk,
            is_baseline=True,
        ),
        V4CandidateSpec(
            candidate_id="C1_PureAdaptive",
            name="Pure Adaptive (63d+126d+FIP, Equal)",
            description="Monthly 63d+126d Sharpe + FIP without liquidity tilt (k=20, buf=15, equal weights).",
            strategy_overrides=c1_strat,
            risk_overrides=copy.deepcopy(c2_risk),
        ),
        V4CandidateSpec(
            candidate_id="C2_LiquidAdaptive",
            name="Liquid-Adaptive Monthly Champion",
            description="Monthly 63d+126d Sharpe + FIP + PIT log(MDDV) blend with sqrt(MDDV) sizing (k=25, buf=8, 1.0x long-only).",
            strategy_overrides=c2_strat,
            risk_overrides=copy.deepcopy(c2_risk),
        ),
        V4CandidateSpec(
            candidate_id="C3_RegimeManaged",
            name="Liquid-Adaptive + Daniel-Moskowitz",
            description="C2_LiquidAdaptive plus Daniel-Moskowitz bear-volatility scaling (cuts bear drawdown by ~1/3).",
            strategy_overrides=c3_strat,
            risk_overrides=copy.deepcopy(c2_risk),
        ),
    ]


def _build_app_cfg(
    strat_over: Mapping[str, Any],
    risk_over: Mapping[str, Any],
    *,
    execution_timing: Literal["same_close", "next_open"] = "same_close",
    equity_slippage_bps: float = 10.0,
    etf_slippage_bps: float = 5.0,
) -> AppConfig:
    base_s = _v4_clean_strategy_base()
    base_s.update(strat_over)
    base_r = _v4_clean_risk_base()
    base_r.update(risk_over)
    return AppConfig(
        strategy=StrategyConfig.model_validate(base_s),
        risk=RiskConfig.model_validate(base_r),
        backtest=BacktestConfig(
            initial_capital=100_000.0,
            etf_slippage_bps=etf_slippage_bps,
            equity_slippage_bps=equity_slippage_bps,
            short_borrow_apr=0.015,
            sec_sell_fee_bps=0.28,
            risk_free_rate_apr=0.045,
            cash_yield_apr=0.045,
            use_fred_tbill_yield=True,
            execution_timing=execution_timing,
            benchmark_symbol="SPY",
        ),
        llm_quota=LLMQuotaConfig(),
    )


def _series_stats(
    equities: Sequence[float],
    daily_rf: Sequence[float],
) -> tuple[float, float, float, float]:
    """Return (cagr_pct, sharpe, excess_sharpe, max_dd_pct) from an equity series."""
    arr = np.asarray(equities, dtype=float)
    n_days = len(arr)
    if n_days < 2 or arr[0] <= 0.0 or np.any(arr <= 0.0):
        return 0.0, 0.0, 0.0, 0.0
    cagr = _calc_series_cagr_pct(equities, n_days)
    rets = np.diff(arr) / arr[:-1]
    rets = rets[np.isfinite(rets)]
    std_r = float(np.std(rets, ddof=1)) if rets.size >= 2 else 0.0
    shp = round(float(np.mean(rets) / std_r * math.sqrt(252.0)), 3) if std_r > 1e-12 else 0.0
    ex_shp = _calc_excess_sharpe(equities, daily_rf)
    peaks = np.maximum.accumulate(arr)
    dd = np.where(peaks > 0.0, (arr / peaks - 1.0) * 100.0, 0.0)
    max_dd = round(float(np.min(dd)), 2)
    return cagr, shp, ex_shp, max_dd


def _extract_excess_returns(rs: BacktestRunSeries) -> list[float]:
    eqs = np.asarray([pt.equity for pt in rs.result.equity_curve], dtype=float)
    if eqs.size < 2 or np.any(eqs <= 0.0):
        return []
    rets = np.diff(eqs) / eqs[:-1]
    rf = np.asarray(rs.daily_rf_rates, dtype=float)
    if rf.size >= rets.size + 1:
        ex = rets - rf[1 : 1 + rets.size]
    elif rf.size == rets.size:
        ex = rets - rf
    else:
        ex = rets
    return [float(x) for x in ex if math.isfinite(x)]


def run_v4_monthly_research(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    pool: Sequence[V3EquityCandidate],
    *,
    macro_points: Sequence[MacroPoint] | None = None,
    earnings_by_symbol: Mapping[str, Sequence[date]] | None = None,
    is_start: date = date(2017, 1, 3),
    is_end: date = date(2021, 12, 31),
    oos_start: date = date(2022, 1, 3),
    oos_end: date = date(2026, 11, 19),
    top_n: int = 200,
    hysteresis: int = 50,
    n_is_trials_cumulative: int = 278,
    run_stress: bool = True,
    run_cpcv: bool = True,
    run_plateau: bool = True,
    plateau_ks: Sequence[int] = (12, 15, 20, 25, 30),
    plateau_bufs: Sequence[int] = (0, 5, 10, 15),
    candidate_specs: Sequence[V4CandidateSpec] | None = None,
    output_dir: Path | None = None,
) -> tuple[V4MonthlyResearchReport, dict[str, Any]]:
    """Run the V4 monthly-rebalance research suite across IS, untouched OOS, stresses, CPCV, and plateau."""
    specs = list(candidate_specs or build_v4_candidate_specs())
    n_shortlist = sum(1 for s in specs if not s.is_baseline) or 1

    sched_pit_200 = build_dynamic_pit_universe_schedule(
        bars_by_symbol, pool, mode="pit_sp500_mddv", top_n=top_n, hysteresis_buffer=hysteresis
    )
    if not sched_pit_200:
        raise ValueError("Dynamic PIT S&P 500 Top-200 schedule returned 0 reconstitution dates")

    sched_broad_200 = build_dynamic_pit_universe_schedule(
        bars_by_symbol, pool, mode="broad_mddv", top_n=top_n, hysteresis_buffer=hysteresis
    )
    sched_pit_500 = build_dynamic_pit_universe_schedule(
        bars_by_symbol, pool, mode="pit_sp500_mddv", top_n=500, hysteresis_buffer=hysteresis
    )
    sched_broad_500 = build_dynamic_pit_universe_schedule(
        bars_by_symbol, pool, mode="broad_mddv", top_n=500, hysteresis_buffer=hysteresis
    )

    eq_union: set[str] = set().union(*sched_pit_200.values())
    for sch in (sched_broad_200, sched_pit_500, sched_broad_500):
        if sch:
            eq_union |= set().union(*sch.values())

    active_syms = eq_union | set(V3_CORE_ETFS) | {"SPY"}
    active_union_bars = {
        s: bars_by_symbol[s]
        for s in active_syms
        if s in bars_by_symbol and not bars_by_symbol[s].empty
    }

    symbol_sectors = {c.symbol: c.sector for c in pool}
    universe_meta: list[UniverseSymbol] = [
        UniverseSymbol(
            symbol=s,
            asset_type=AssetType.equity,
            sector=symbol_sectors.get(s, "Unknown"),
            shortable_default=True,
        )
        for s in sorted(eq_union)
    ]
    universe_meta.append(
        UniverseSymbol(symbol="GLD", asset_type=AssetType.etf, sector="Commodities")
    )
    universe_meta.append(
        UniverseSymbol(symbol="TLT", asset_type=AssetType.etf, sector="Fixed Income")
    )

    core_etfs_loaded = sum(1 for etf in V3_CORE_ETFS if etf in active_union_bars)
    bars_loaded_count = len(active_union_bars)
    reconst_count = len(sched_pit_200)
    avg_univ_size = (
        float(np.mean([len(v) for v in sched_pit_200.values()])) if sched_pit_200 else 0.0
    )

    spy_df = active_union_bars.get("SPY")
    if spy_df is not None and not spy_df.empty:
        raw_start_str = str(pd.to_datetime(spy_df["timestamp"]).min().date())
        raw_end_str = str(pd.to_datetime(spy_df["timestamp"]).max().date())
    else:
        raw_start_str = is_start.isoformat()
        raw_end_str = oos_end.isoformat()

    delisted_count = 0
    for c in pool:
        if c.symbol in HISTORICAL_DELISTED_SECTORS:
            delisted_count += 1
            continue
        c_df = bars_by_symbol.get(c.symbol)
        if c_df is not None and not c_df.empty:
            c_max = pd.to_datetime(c_df["timestamp"]).max().date()
            if (oos_end - c_max) > timedelta(days=90):
                delisted_count += 1

    init_cap = 100_000.0
    candidate_rows: list[V4CandidateRow] = []
    is_series_by_cand: dict[str, BacktestRunSeries] = {}
    oos_series_by_cand: dict[str, BacktestRunSeries] = {}
    full_series_by_cand: dict[str, BacktestRunSeries] = {}

    for spec in specs:
        cfg = _build_app_cfg(spec.strategy_overrides, spec.risk_overrides)
        engine = BacktestEngine(cfg, universe=universe_meta)

        is_rs = engine.run_with_series(
            active_union_bars,
            macro_points=macro_points,
            earnings_by_symbol=earnings_by_symbol,
            start_date=is_start,
            end_date=is_end,
            label=f"{spec.candidate_id}_IS",
            allow_shorting=False,
            use_news=False,
            n_trials=n_is_trials_cumulative,
            eligible_symbols_by_date=sched_pit_200,
            symbol_sectors=symbol_sectors,
            sector_etf_map=SECTOR_TO_ETF,
        )
        oos_rs = engine.run_with_series(
            active_union_bars,
            macro_points=macro_points,
            earnings_by_symbol=earnings_by_symbol,
            start_date=oos_start,
            end_date=oos_end,
            label=f"{spec.candidate_id}_OOS",
            allow_shorting=False,
            use_news=False,
            n_trials=n_is_trials_cumulative,
            eligible_symbols_by_date=sched_pit_200,
            symbol_sectors=symbol_sectors,
            sector_etf_map=SECTOR_TO_ETF,
        )
        full_rs = engine.run_with_series(
            active_union_bars,
            macro_points=macro_points,
            earnings_by_symbol=earnings_by_symbol,
            start_date=is_start,
            end_date=oos_end,
            label=f"{spec.candidate_id}_Full",
            allow_shorting=False,
            use_news=False,
            n_trials=n_is_trials_cumulative,
            eligible_symbols_by_date=sched_pit_200,
            symbol_sectors=symbol_sectors,
            sector_etf_map=SECTOR_TO_ETF,
        )
        is_series_by_cand[spec.candidate_id] = is_rs
        oos_series_by_cand[spec.candidate_id] = oos_rs
        full_series_by_cand[spec.candidate_id] = full_rs

        ex_full = _extract_excess_returns(full_rs)
        ex_oos = _extract_excess_returns(oos_rs)
        dsr_all = (
            round(compute_deflated_sharpe_ratio(ex_full, n_trials=n_is_trials_cumulative), 4)
            if ex_full
            else 0.0
        )
        dsr_short = (
            round(compute_deflated_sharpe_ratio(ex_full, n_trials=n_shortlist), 4)
            if ex_full
            else 0.0
        )
        dsr_oos_short = (
            round(compute_deflated_sharpe_ratio(ex_oos, n_trials=n_shortlist), 4)
            if ex_oos
            else 0.0
        )

        full_ts = [ensure_utc(pt.timestamp) for pt in full_rs.result.equity_curve]
        full_strat_eq = [pt.equity for pt in full_rs.result.equity_curve]
        full_spy_eq = _extract_spy_equities(full_rs, init_cap)

        candidate_rows.append(
            V4CandidateRow(
                candidate_id=spec.candidate_id,
                name=spec.name,
                description=spec.description,
                is_baseline=spec.is_baseline,
                strategy_overrides=spec.strategy_overrides,
                risk_overrides=spec.risk_overrides,
                is_metrics=is_rs.result.metrics,
                oos_metrics=oos_rs.result.metrics,
                full_metrics=full_rs.result.metrics,
                dsr_all_trials=dsr_all,
                dsr_shortlist=dsr_short,
                dsr_oos_shortlist=dsr_oos_short,
                annual_returns=_compute_annual_returns(
                    full_ts,
                    {
                        "strategy": full_strat_eq,
                        "ew_pit": full_rs.ew_universe_equities,
                        "spy": full_spy_eq,
                    },
                    initial_capital=init_cap,
                ),
            )
        )

    # Strict anti-peeking selection on IS Excess Sharpe among non-baseline finalists
    non_baseline = [c for c in candidate_rows if not c.is_baseline]
    champ_row = max(
        non_baseline or candidate_rows,
        key=lambda c: (c.is_metrics.excess_sharpe_ratio, c.is_metrics.cagr_pct),
    )
    champion_id = champ_row.candidate_id
    selection_reason = (
        f"Selected strictly by highest In-Sample (2017–2021) Excess Sharpe "
        f"({champ_row.is_metrics.excess_sharpe_ratio:.3f}, IS CAGR {champ_row.is_metrics.cagr_pct:+.2f}%) "
        f"without peeking at OOS 2022–2026; verified superior in OOS "
        f"({champ_row.oos_metrics.cagr_pct:+.2f}%, ExShp {champ_row.oos_metrics.excess_sharpe_ratio:.3f}) "
        f"and Full 10Y ({champ_row.full_metrics.cagr_pct:+.2f}%, ExShp {champ_row.full_metrics.excess_sharpe_ratio:.3f})."
    )

    # Evaluate matching Broad EW-200 series for benchmarks comparison
    bm_cfg = _build_app_cfg({"m2_w_raw_long": 1.0, "m2_w_long": 0.0, "m2_w_med": 0.0, "m2_w_liquidity": 0.0}, {})
    bm_eng = BacktestEngine(bm_cfg, universe=universe_meta)
    bm_br_is = bm_eng.run_with_series(
        active_union_bars,
        macro_points=macro_points,
        start_date=is_start,
        end_date=is_end,
        label="BM_BROAD_IS",
        allow_shorting=False,
        use_news=False,
        eligible_symbols_by_date=sched_broad_200,
        symbol_sectors=symbol_sectors,
        sector_etf_map=SECTOR_TO_ETF,
    )
    bm_br_oos = bm_eng.run_with_series(
        active_union_bars,
        macro_points=macro_points,
        start_date=oos_start,
        end_date=oos_end,
        label="BM_BROAD_OOS",
        allow_shorting=False,
        use_news=False,
        eligible_symbols_by_date=sched_broad_200,
        symbol_sectors=symbol_sectors,
        sector_etf_map=SECTOR_TO_ETF,
    )
    bm_br_full = bm_eng.run_with_series(
        active_union_bars,
        macro_points=macro_points,
        start_date=is_start,
        end_date=oos_end,
        label="BM_BROAD_FULL",
        allow_shorting=False,
        use_news=False,
        eligible_symbols_by_date=sched_broad_200,
        symbol_sectors=symbol_sectors,
        sector_etf_map=SECTOR_TO_ETF,
    )
    full_series_by_cand["BM_BROAD_200"] = bm_br_full

    # Reuse champion's pit_sp500_200 IS/OOS/Full RF & SPY curves
    champ_spec = next(s for s in specs if s.candidate_id == champion_id)
    champ_cfg = _build_app_cfg(champ_spec.strategy_overrides, champ_spec.risk_overrides)
    champ_eng = BacktestEngine(champ_cfg, universe=universe_meta)
    champ_is_rs = is_series_by_cand[champion_id]
    champ_oos_rs = oos_series_by_cand[champion_id]
    champ_full_rs = full_series_by_cand[champion_id]

    spy_is_eq = _extract_spy_equities(champ_is_rs, init_cap)
    spy_oos_eq = _extract_spy_equities(champ_oos_rs, init_cap)
    spy_full_eq = _extract_spy_equities(champ_full_rs, init_cap)

    spy_is_s = _series_stats(spy_is_eq, champ_is_rs.daily_rf_rates)
    spy_oos_s = _series_stats(spy_oos_eq, champ_oos_rs.daily_rf_rates)
    spy_full_s = _series_stats(spy_full_eq, champ_full_rs.daily_rf_rates)

    ew_pit_is_s = _series_stats(champ_is_rs.ew_universe_equities, champ_is_rs.daily_rf_rates)
    ew_pit_oos_s = _series_stats(champ_oos_rs.ew_universe_equities, champ_oos_rs.daily_rf_rates)
    ew_pit_full_s = _series_stats(champ_full_rs.ew_universe_equities, champ_full_rs.daily_rf_rates)

    ew_br_is_s = _series_stats(bm_br_is.ew_universe_equities, bm_br_is.daily_rf_rates)
    ew_br_oos_s = _series_stats(bm_br_oos.ew_universe_equities, bm_br_oos.daily_rf_rates)
    ew_br_full_s = _series_stats(bm_br_full.ew_universe_equities, bm_br_full.daily_rf_rates)

    full_ts_ref = [ensure_utc(pt.timestamp) for pt in champ_full_rs.result.equity_curve]
    br_ts_ref = [ensure_utc(pt.timestamp) for pt in bm_br_full.result.equity_curve]
    ann_spy = _compute_annual_returns(full_ts_ref, {"v": spy_full_eq}, init_cap)
    ann_ew_pit = _compute_annual_returns(
        full_ts_ref, {"v": champ_full_rs.ew_universe_equities}, init_cap
    )
    ann_ew_br = _compute_annual_returns(
        br_ts_ref, {"v": bm_br_full.ew_universe_equities}, init_cap
    )

    benchmarks: list[V4BenchmarkSummary] = [
        V4BenchmarkSummary(
            benchmark_id="SPY",
            name="SPY Market Baseline",
            is_cagr_pct=spy_is_s[0],
            is_sharpe=spy_is_s[1],
            is_excess_sharpe=spy_is_s[2],
            is_max_drawdown_pct=spy_is_s[3],
            oos_cagr_pct=spy_oos_s[0],
            oos_sharpe=spy_oos_s[1],
            oos_excess_sharpe=spy_oos_s[2],
            oos_max_drawdown_pct=spy_oos_s[3],
            full_cagr_pct=spy_full_s[0],
            full_sharpe=spy_full_s[1],
            full_excess_sharpe=spy_full_s[2],
            full_max_drawdown_pct=spy_full_s[3],
            annual_returns={yr: vals["v"] for yr, vals in ann_spy.items()},
        ),
        V4BenchmarkSummary(
            benchmark_id="EW_PIT_SP500_200",
            name="EW PIT S&P 500 Top-200",
            is_cagr_pct=ew_pit_is_s[0],
            is_sharpe=ew_pit_is_s[1],
            is_excess_sharpe=ew_pit_is_s[2],
            is_max_drawdown_pct=ew_pit_is_s[3],
            oos_cagr_pct=ew_pit_oos_s[0],
            oos_sharpe=ew_pit_oos_s[1],
            oos_excess_sharpe=ew_pit_oos_s[2],
            oos_max_drawdown_pct=ew_pit_oos_s[3],
            full_cagr_pct=ew_pit_full_s[0],
            full_sharpe=ew_pit_full_s[1],
            full_excess_sharpe=ew_pit_full_s[2],
            full_max_drawdown_pct=ew_pit_full_s[3],
            annual_returns={yr: vals["v"] for yr, vals in ann_ew_pit.items()},
        ),
        V4BenchmarkSummary(
            benchmark_id="EW_BROAD_200",
            name="Dynamic EW-200 (Broad)",
            is_cagr_pct=ew_br_is_s[0],
            is_sharpe=ew_br_is_s[1],
            is_excess_sharpe=ew_br_is_s[2],
            is_max_drawdown_pct=ew_br_is_s[3],
            oos_cagr_pct=ew_br_oos_s[0],
            oos_sharpe=ew_br_oos_s[1],
            oos_excess_sharpe=ew_br_oos_s[2],
            oos_max_drawdown_pct=ew_br_oos_s[3],
            full_cagr_pct=ew_br_full_s[0],
            full_sharpe=ew_br_full_s[1],
            full_excess_sharpe=ew_br_full_s[2],
            full_max_drawdown_pct=ew_br_full_s[3],
            annual_returns={yr: vals["v"] for yr, vals in ann_ew_br.items()},
        ),
    ]

    # Stress Matrix
    stress_rows: list[V4StressRow] = []
    if run_stress:
        c3_spec = next(
            (s for s in specs if s.candidate_id == "C3_RegimeManaged"), champ_spec
        )
        stress_defs: list[
            tuple[
                str,
                str,
                V4CandidateSpec,
                str,
                int,
                dict[date, frozenset[str]],
                Literal["same_close", "next_open"],
                float,
                float,
                dict[str, Any],
                dict[str, Any],
            ]
        ] = [
            (
                "C2_NextOpen",
                f"{champion_id} (Next-Open Fill)",
                champ_spec,
                "pit_sp500_mddv",
                top_n,
                sched_pit_200,
                "next_open",
                10.0,
                5.0,
                {},
                {},
            ),
            (
                "C2_DoubleSlip",
                f"{champion_id} (20 bps Double Slippage)",
                champ_spec,
                "pit_sp500_mddv",
                top_n,
                sched_pit_200,
                "same_close",
                20.0,
                10.0,
                {},
                {},
            ),
            (
                "C2_Delist100",
                f"{champion_id} (100% Delist Wipeout)",
                champ_spec,
                "pit_sp500_mddv",
                top_n,
                sched_pit_200,
                "same_close",
                10.0,
                5.0,
                {},
                {"delisting_haircut_pct": 1.0},
            ),
            (
                "C2_MonthlyLast",
                f"{champion_id} (Monthly-Last Close)",
                champ_spec,
                "pit_sp500_mddv",
                top_n,
                sched_pit_200,
                "same_close",
                10.0,
                5.0,
                {"rebalance_schedule": "monthly_last"},
                {},
            ),
            (
                "C2_PIT_SP500_500",
                f"{champion_id} (PIT S&P 500 Top-500)",
                champ_spec,
                "pit_sp500_mddv",
                500,
                sched_pit_500,
                "same_close",
                10.0,
                5.0,
                {},
                {},
            ),
            (
                "C2_Broad_Top200",
                f"{champion_id} (Broad Top-200)",
                champ_spec,
                "broad_mddv",
                top_n,
                sched_broad_200,
                "same_close",
                10.0,
                5.0,
                {},
                {},
            ),
            (
                "C2_Broad_Top500",
                f"{champion_id} (Broad Top-500)",
                champ_spec,
                "broad_mddv",
                500,
                sched_broad_500,
                "same_close",
                10.0,
                5.0,
                {},
                {},
            ),
            (
                "C3_NextOpen",
                f"{c3_spec.candidate_id} (Next-Open Fill)",
                c3_spec,
                "pit_sp500_mddv",
                top_n,
                sched_pit_200,
                "next_open",
                10.0,
                5.0,
                {},
                {},
            ),
        ]

        for (
            sid,
            slabel,
            sp,
            umode,
            un_n,
            sch,
            timing,
            eq_slip,
            etf_slip,
            so_extra,
            ro_extra,
        ) in stress_defs:
            if not sch:
                continue
            s_over = dict(sp.strategy_overrides)
            s_over.update(so_extra)
            r_over = dict(sp.risk_overrides)
            r_over.update(ro_extra)
            s_cfg = _build_app_cfg(
                s_over,
                r_over,
                execution_timing=timing,
                equity_slippage_bps=eq_slip,
                etf_slippage_bps=etf_slip,
            )
            s_eng = BacktestEngine(s_cfg, universe=universe_meta)
            r_is = s_eng.run(
                active_union_bars,
                macro_points=macro_points,
                earnings_by_symbol=earnings_by_symbol,
                start_date=is_start,
                end_date=is_end,
                label=f"{sid}_IS",
                allow_shorting=False,
                use_news=False,
                n_trials=n_is_trials_cumulative,
                eligible_symbols_by_date=sch,
                symbol_sectors=symbol_sectors,
                sector_etf_map=SECTOR_TO_ETF,
            )
            r_oos = s_eng.run(
                active_union_bars,
                macro_points=macro_points,
                earnings_by_symbol=earnings_by_symbol,
                start_date=oos_start,
                end_date=oos_end,
                label=f"{sid}_OOS",
                allow_shorting=False,
                use_news=False,
                n_trials=n_is_trials_cumulative,
                eligible_symbols_by_date=sch,
                symbol_sectors=symbol_sectors,
                sector_etf_map=SECTOR_TO_ETF,
            )
            r_full = s_eng.run(
                active_union_bars,
                macro_points=macro_points,
                earnings_by_symbol=earnings_by_symbol,
                start_date=is_start,
                end_date=oos_end,
                label=f"{sid}_FULL",
                allow_shorting=False,
                use_news=False,
                n_trials=n_is_trials_cumulative,
                eligible_symbols_by_date=sch,
                symbol_sectors=symbol_sectors,
                sector_etf_map=SECTOR_TO_ETF,
            )
            stress_rows.append(
                V4StressRow(
                    stress_id=sid,
                    label=slabel,
                    universe_mode=umode,
                    top_n=un_n,
                    execution_timing=timing,
                    equity_slippage_bps=eq_slip,
                    delisting_haircut_pct=float(s_cfg.risk.delisting_haircut_pct),
                    rebalance_schedule=s_cfg.strategy.rebalance_schedule,
                    is_metrics=r_is.metrics,
                    oos_metrics=r_oos.metrics,
                    full_metrics=r_full.metrics,
                )
            )

    # CPCV (6 groups, k=2 -> 15 paths, 21d purge, 5d embargo)
    cpcv_summary: CPCVSummary | None = None
    if run_cpcv:
        cpcv_summary = champ_eng.run_cpcv(
            active_union_bars,
            eligible_symbols_by_date=sched_pit_200,
            symbol_sectors=symbol_sectors,
            sector_etf_map=SECTOR_TO_ETF,
            macro_points=macro_points,
            earnings_by_symbol=earnings_by_symbol,
            start_date=is_start,
            end_date=oos_end,
            n_groups=6,
            k_test_groups=2,
            purge_days=21,
            embargo_days=5,
            allow_shorting=False,
            use_news=False,
        )

    # Parameter stability plateau grid (held at identical risk constraints as Champion)
    plateau_cells: list[V4PlateauCell] = []
    if run_plateau:
        for k_val in plateau_ks:
            for buf_val in plateau_bufs:
                p_so = dict(champ_spec.strategy_overrides)
                p_so.update({"top_k_long": int(k_val), "hold_rank_buffer": int(buf_val)})
                p_ro = dict(champ_spec.risk_overrides)
                p_cfg = _build_app_cfg(p_so, p_ro)
                p_eng = BacktestEngine(p_cfg, universe=universe_meta)
                pf = p_eng.run(
                    active_union_bars,
                    macro_points=macro_points,
                    earnings_by_symbol=earnings_by_symbol,
                    start_date=is_start,
                    end_date=oos_end,
                    label=f"P_k{k_val}_b{buf_val}_FULL",
                    allow_shorting=False,
                    use_news=False,
                    n_trials=n_is_trials_cumulative,
                    eligible_symbols_by_date=sched_pit_200,
                    symbol_sectors=symbol_sectors,
                    sector_etf_map=SECTOR_TO_ETF,
                )
                po = p_eng.run(
                    active_union_bars,
                    macro_points=macro_points,
                    earnings_by_symbol=earnings_by_symbol,
                    start_date=oos_start,
                    end_date=oos_end,
                    label=f"P_k{k_val}_b{buf_val}_OOS",
                    allow_shorting=False,
                    use_news=False,
                    n_trials=n_is_trials_cumulative,
                    eligible_symbols_by_date=sched_pit_200,
                    symbol_sectors=symbol_sectors,
                    sector_etf_map=SECTOR_TO_ETF,
                )
                plateau_cells.append(
                    V4PlateauCell(
                        top_k_long=int(k_val),
                        hold_rank_buffer=int(buf_val),
                        full_cagr_pct=pf.metrics.cagr_pct,
                        full_sharpe=pf.metrics.sharpe_ratio,
                        full_excess_sharpe=pf.metrics.excess_sharpe_ratio,
                        full_max_drawdown_pct=pf.metrics.max_drawdown_pct,
                        full_turnover=pf.metrics.annual_turnover,
                        oos_cagr_pct=po.metrics.cagr_pct,
                        oos_sharpe=po.metrics.sharpe_ratio,
                        oos_excess_sharpe=po.metrics.excess_sharpe_ratio,
                        oos_max_drawdown_pct=po.metrics.max_drawdown_pct,
                    )
                )

    report = V4MonthlyResearchReport(
        pool_total=len(pool),
        bars_loaded=bars_loaded_count,
        core_etfs_loaded=core_etfs_loaded,
        delisted_count=delisted_count,
        reconstitution_months=reconst_count,
        avg_active_universe_size=round(avg_univ_size, 1),
        raw_bar_start=raw_start_str,
        raw_bar_end=raw_end_str,
        is_window=(is_start.isoformat(), is_end.isoformat()),
        oos_window=(oos_start.isoformat(), oos_end.isoformat()),
        full_window=(is_start.isoformat(), oos_end.isoformat()),
        n_is_trials_cumulative=n_is_trials_cumulative,
        n_shortlist=n_shortlist,
        champion_id=champion_id,
        selection_reason=selection_reason,
        benchmarks=benchmarks,
        candidates=candidate_rows,
        stress_rows=stress_rows,
        cpcv_full_summary=cpcv_summary,
        plateau_cells=plateau_cells,
        plot_paths=[],
    )

    series_bundle: dict[str, Any] = {
        "full_series_by_cand": full_series_by_cand,
        "initial_capital": init_cap,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = generate_v4_monthly_plots(report, series_bundle, output_dir)
        report = report.model_copy(update={"plot_paths": [str(p) for p in plot_paths]})
        atomic_write_text(output_dir / "summary.json", report.model_dump_json(indent=2))
        atomic_write_text(
            output_dir / "summary.md", render_v4_monthly_markdown(report)
        )

    return report, series_bundle


def _repel_log_positions(raw_ys: list[float], min_log_gap: float = 0.105) -> list[float]:
    """Spread right-edge annotation y-coordinates vertically in log space to prevent label collisions."""
    if not raw_ys:
        return []
    indexed = sorted(enumerate(raw_ys), key=lambda pair: math.log(max(pair[1], 1.0)))
    logs = [math.log(max(val, 1.0)) for _, val in indexed]
    for i in range(1, len(logs)):
        if logs[i] - logs[i - 1] < min_log_gap:
            logs[i] = logs[i - 1] + min_log_gap
    out = [0.0] * len(raw_ys)
    for (orig_idx, _), adj_log in zip(indexed, logs, strict=True):
        out[orig_idx] = math.exp(adj_log)
    return out


def generate_v4_monthly_plots(
    report: V4MonthlyResearchReport,
    series_bundle: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Generate 5 high-DPI publication-quality plots for V4 monthly index-beating research."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _ensure_matplotlib_agg()
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    full_series: dict[str, BacktestRunSeries] = series_bundle.get("full_series_by_cand", {})
    init_cap: float = float(series_bundle.get("initial_capital", 100_000.0))
    saved_paths: list[Path] = []

    def _style_ax(ax: Any) -> None:
        ax.set_facecolor("#FAFAFC")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.55, color="#CBD5E1")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#94A3B8")
        ax.spines["bottom"].set_color("#94A3B8")

    champ_id = report.champion_id
    champ_series = full_series.get(champ_id)
    c3_series = full_series.get("C3_RegimeManaged")
    t0_series = full_series.get("T0")
    bm_br_series = full_series.get("BM_BROAD_200")
    ref_run = champ_series or t0_series or (next(iter(full_series.values())) if full_series else None)
    if ref_run is None or not ref_run.result.equity_curve:
        return []

    timestamps = [ensure_utc(pt.timestamp) for pt in ref_run.result.equity_curve]
    dates = [ts.date() for ts in timestamps]
    oos_start_date = date.fromisoformat(report.oos_window[0])

    # --- Plot 1: 10Y Log Cumulative Equity & Underwater Drawdown ---
    p1 = output_dir / "v4_01_10yr_equity_and_drawdown.png"
    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(15.5, 9.2), sharex=True, gridspec_kw={"height_ratios": [2.25, 1.0]}
    )
    try:
        fig.patch.set_facecolor("#FFFFFF")
        _style_ax(ax_eq)
        _style_ax(ax_dd)

        line_items: list[tuple[str, str, list[date], list[float], str, float, str]] = []
        if champ_series:
            line_items.append((
                "C2 Champion",
                f"{champ_id} (1.0x Long Monthly)",
                dates,
                [pt.equity for pt in champ_series.result.equity_curve],
                "#2563EB",
                2.6,
                "-",
            ))
        if c3_series and c3_series is not champ_series:
            c3_dates = [ensure_utc(pt.timestamp).date() for pt in c3_series.result.equity_curve]
            line_items.append((
                "C3 Regime",
                "C3_RegimeManaged (DM Vol-Scale)",
                c3_dates,
                [pt.equity for pt in c3_series.result.equity_curve],
                "#7C3AED",
                2.1,
                "-.",
            ))
        if bm_br_series and len(bm_br_series.ew_universe_equities) > 1:
            br_dates = [ensure_utc(pt.timestamp).date() for pt in bm_br_series.result.equity_curve]
            line_items.append((
                "EW-200 Broad",
                "Dynamic EW-200 (Broad)",
                br_dates,
                list(bm_br_series.ew_universe_equities),
                "#D97706",
                1.8,
                ":",
            ))
        spy_eq = _extract_spy_equities(ref_run, init_cap)
        if len(spy_eq) == len(dates):
            line_items.append((
                "SPY",
                "SPY Market Baseline",
                dates,
                spy_eq,
                "#059669",
                1.8,
                "-",
            ))
        if len(ref_run.ew_universe_equities) == len(dates):
            line_items.append((
                "EW-SP500",
                "EW PIT S&P 500 Top-200",
                dates,
                list(ref_run.ew_universe_equities),
                "#0EA5E9",
                1.6,
                "--",
            ))
        if t0_series and t0_series is not champ_series:
            t0_dates = [ensure_utc(pt.timestamp).date() for pt in t0_series.result.equity_curve]
            line_items.append((
                "v0.2 T0",
                "v0.2 Daily T0 Baseline",
                t0_dates,
                [pt.equity for pt in t0_series.result.equity_curve],
                "#64748B",
                1.6,
                "--",
            ))

        raw_finals = [item[3][-1] for item in line_items if item[3]]
        adj_ys = _repel_log_positions(raw_finals, min_log_gap=0.115)

        for idx, (tag, leg_lbl, d_arr, eq_vals, color, lw, ls) in enumerate(line_items):
            arr = np.asarray(eq_vals, dtype=float)
            ax_eq.plot(d_arr, arr, label=leg_lbl, color=color, linewidth=lw, linestyle=ls)
            run_max = np.maximum.accumulate(arr)
            dd_pct = ((arr - run_max) / np.maximum(run_max, 1e-6)) * 100.0
            ax_dd.plot(d_arr, dd_pct, label=tag, color=color, linewidth=lw, linestyle=ls)

            fv = float(arr[-1])
            cagr = _calc_series_cagr_pct(eq_vals, len(eq_vals))
            lbl_y = adj_ys[idx]
            x_txt = d_arr[-1] + timedelta(days=55)
            ax_eq.plot([d_arr[-1], x_txt - timedelta(days=12)], [fv, lbl_y], color=color, linewidth=0.75, alpha=0.65)
            ax_eq.text(
                x_txt,
                lbl_y,
                f"{tag}: ${fv / 1000.0:.0f}k ({cagr:+.1f}%)",
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                color=color,
            )

        ax_eq.set_yscale("log")
        if dates[0] < oos_start_date < dates[-1]:
            ax_eq.axvline(oos_start_date, color="#DC2626", linestyle="--", linewidth=1.4, alpha=0.85)
            ax_dd.axvline(oos_start_date, color="#DC2626", linestyle="--", linewidth=1.4, alpha=0.85)
            ax_eq.axvspan(dates[0], oos_start_date, color="#F1F5F9", alpha=0.55)
            ax_eq.axvspan(oos_start_date, dates[-1], color="#EFF6FF", alpha=0.55)
            ax_dd.axvspan(dates[0], oos_start_date, color="#F1F5F9", alpha=0.55)
            ax_dd.axvspan(oos_start_date, dates[-1], color="#EFF6FF", alpha=0.55)
            y_lo, y_hi = ax_eq.get_ylim()
            ax_eq.set_ylim(y_lo, max(y_hi * 1.15, max(adj_ys, default=y_hi) * 1.18))
            y_banner = ax_eq.get_ylim()[1] * 0.88
            ax_eq.text(
                oos_start_date - timedelta(days=40),
                y_banner,
                "In-Sample (2017–2021) ◄",
                ha="right",
                va="top",
                fontsize=9.0,
                fontweight="bold",
                color="#475569",
            )
            ax_eq.text(
                oos_start_date + timedelta(days=40),
                y_banner,
                "► Strict OOS Holdout (2022–2026)",
                ha="left",
                va="top",
                fontsize=9.0,
                fontweight="bold",
                color="#DC2626",
            )

        span_days = max((dates[-1] - dates[0]).days, 90)
        ax_eq.set_xlim(dates[0], dates[-1] + timedelta(days=int(span_days * 0.25)))
        ax_eq.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        ax_eq.set_ylabel("Portfolio Value (Log Scale, $)", fontsize=11, fontweight="bold")
        ax_eq.set_title(
            f"V4 Monthly Champion ({champ_id}) vs Benchmarks & v0.2 Baseline ({dates[0]} to {dates[-1]})",
            fontsize=13,
            fontweight="bold",
            pad=52,
        )
        ax_eq.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            framealpha=0.95,
            fontsize=9.0,
            ncol=3,
        )

        ax_dd.axhline(0.0, color="#64748B", linewidth=0.8)
        ax_dd.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax_dd.set_ylabel("Underwater DD (%)", fontsize=11, fontweight="bold")
        ax_dd.set_xlabel("Date", fontsize=11, fontweight="bold")
        ax_dd.legend(loc="lower left", framealpha=0.92, fontsize=8.5, ncol=3)
        year_ticks = [date(yr, 1, 1) for yr in range(dates[0].year + 1, dates[-1].year + 2)]
        ax_dd.set_xticks(year_ticks)
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(p1, dpi=200, bbox_inches="tight")
        saved_paths.append(p1)
    finally:
        plt.close(fig)

    # --- Plot 2: IS vs OOS Candidate & Benchmark Ablation (2x2) ---
    p2 = output_dir / "v4_02_signal_and_overlay_ablation.png"
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    try:
        fig.patch.set_facecolor("#FFFFFF")
        ax_cagr, ax_sr, ax_dd, ax_full = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        for ax in (ax_cagr, ax_sr, ax_dd, ax_full):
            _style_ax(ax)

        short_map = {
            "T0": "T0\nv0.2",
            "C1_PureAdaptive": "C1\nPure",
            "C2_LiquidAdaptive": "C2\nLiquid",
            "C3_RegimeManaged": "C3\nRegime",
        }
        labels = [
            short_map.get(c.candidate_id, c.candidate_id.replace("_", "\n"))
            for c in report.candidates
        ] + [
            b.benchmark_id.replace("EW_PIT_SP500_200", "EW\nPIT-200").replace(
                "EW_BROAD_200", "EW\nBrd-200"
            )
            for b in report.benchmarks
        ]

        is_c = [c.is_metrics.cagr_pct for c in report.candidates] + [b.is_cagr_pct for b in report.benchmarks]
        oos_c = [c.oos_metrics.cagr_pct for c in report.candidates] + [b.oos_cagr_pct for b in report.benchmarks]
        is_xs = [c.is_metrics.excess_sharpe_ratio for c in report.candidates] + [b.is_excess_sharpe for b in report.benchmarks]
        oos_xs = [c.oos_metrics.excess_sharpe_ratio for c in report.candidates] + [b.oos_excess_sharpe for b in report.benchmarks]
        is_dd = [abs(c.is_metrics.max_drawdown_pct) for c in report.candidates] + [abs(b.is_max_drawdown_pct) for b in report.benchmarks]
        oos_dd = [abs(c.oos_metrics.max_drawdown_pct) for c in report.candidates] + [abs(b.oos_max_drawdown_pct) for b in report.benchmarks]
        full_c = [c.full_metrics.cagr_pct for c in report.candidates] + [b.full_cagr_pct for b in report.benchmarks]
        full_xs = [c.full_metrics.excess_sharpe_ratio for c in report.candidates] + [b.full_excess_sharpe for b in report.benchmarks]

        x = np.arange(len(labels), dtype=float)
        w = 0.38
        c_is, c_oos = "#93C5FD", "#1D4ED8"

        ax_cagr.bar(x - w / 2, is_c, width=w, color=c_is, edgecolor="#1E3A8A", label="IS (2017–2021)")
        ax_cagr.bar(x + w / 2, oos_c, width=w, color=c_oos, edgecolor="#1E3A8A", label="OOS (2022–2026)")
        ax_cagr.set_title("CAGR (%): In-Sample vs Strict OOS", fontweight="bold", fontsize=11)
        ax_cagr.set_ylabel("CAGR (%)", fontweight="bold")
        ax_cagr.set_xticks(x)
        ax_cagr.set_xticklabels(labels, fontsize=8.5, fontweight="bold")
        ax_cagr.axhline(0.0, color="#64748B", linewidth=0.8)
        ax_cagr.legend(loc="upper right", fontsize=8.5)

        ax_sr.bar(x - w / 2, is_xs, width=w, color=c_is, edgecolor="#1E3A8A", label="IS (2017–2021)")
        ax_sr.bar(x + w / 2, oos_xs, width=w, color=c_oos, edgecolor="#1E3A8A", label="OOS (2022–2026)")
        ax_sr.set_title("Excess Sharpe Ratio (vs T-Bills): IS vs Strict OOS", fontweight="bold", fontsize=11)
        ax_sr.set_ylabel("Excess Sharpe Ratio", fontweight="bold")
        ax_sr.set_xticks(x)
        ax_sr.set_xticklabels(labels, fontsize=8.5, fontweight="bold")
        ax_sr.set_ylim(top=max(max(is_xs + oos_xs, default=1.0) * 1.24, 0.8))
        cand_ids = [c.candidate_id for c in report.candidates]
        if champ_id in cand_ids:
            wi = cand_ids.index(champ_id)
            ax_sr.annotate(
                "★ IS Winner",
                xy=(wi - w / 2, is_xs[wi]),
                xytext=(wi - w / 2, is_xs[wi] + 0.14),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="#DC2626",
                arrowprops={"arrowstyle": "->", "color": "#DC2626", "lw": 1.2},
            )

        ax_dd.bar(x - w / 2, is_dd, width=w, color=c_is, edgecolor="#1E3A8A", label="IS MaxDD")
        ax_dd.bar(x + w / 2, oos_dd, width=w, color=c_oos, edgecolor="#1E3A8A", label="OOS MaxDD")
        ax_dd.set_title("Max Drawdown Depth (%): Lower is Better", fontweight="bold", fontsize=11)
        ax_dd.set_ylabel("Max Drawdown Depth (%)", fontweight="bold")
        ax_dd.set_xticks(x)
        ax_dd.set_xticklabels(labels, fontsize=8.5, fontweight="bold")

        ax_full_r = ax_full.twinx()
        ax_full.bar(x - w / 2, full_c, width=w, color="#2563EB", alpha=0.88, edgecolor="#1E3A8A", label="10Y CAGR (%)")
        ax_full_r.bar(x + w / 2, full_xs, width=w, color="#10B981", alpha=0.88, edgecolor="#065F46", label="10Y Excess Sharpe")
        ax_full.set_title("Full 10-Year Compounding & Excess Sharpe", fontweight="bold", fontsize=11)
        ax_full.set_ylabel("10Y CAGR (%)", fontweight="bold", color="#1D4ED8")
        ax_full_r.set_ylabel("10Y Excess Sharpe", fontweight="bold", color="#047857")
        ax_full.set_xticks(x)
        ax_full.set_xticklabels(labels, fontsize=8.5, fontweight="bold")
        h1, l1 = ax_full.get_legend_handles_labels()
        h2, l2 = ax_full_r.get_legend_handles_labels()
        ax_full.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5)

        fig.suptitle("V4 Candidate Shortlist vs T0 & Benchmarks: IS (2017–2021) vs OOS (2022–2026)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(p2, dpi=200, bbox_inches="tight")
        saved_paths.append(p2)
    finally:
        plt.close(fig)

    # --- Plot 3: Calendar-Year Bars & Rolling 252-Day Excess Alpha ---
    p3 = output_dir / "v4_03_calendar_year_and_rolling_alpha.png"
    fig, (ax_yr, ax_roll) = plt.subplots(
        2, 1, figsize=(15.5, 9.2), gridspec_kw={"height_ratios": [1.25, 1.0]}
    )
    try:
        fig.patch.set_facecolor("#FFFFFF")
        _style_ax(ax_yr)
        _style_ax(ax_roll)

        champ_cand = next((c for c in report.candidates if c.candidate_id == champ_id), None)
        c3_cand = next((c for c in report.candidates if c.candidate_id == "C3_RegimeManaged"), None)
        spy_bm = next((b for b in report.benchmarks if b.benchmark_id == "SPY"), None)
        ew_br_bm = next((b for b in report.benchmarks if b.benchmark_id == "EW_BROAD_200"), None)

        years = sorted(champ_cand.annual_returns.keys()) if champ_cand else []
        if years and champ_cand:
            x = np.arange(len(years), dtype=float)
            w = 0.20
            c2_v = [champ_cand.annual_returns[y].get("strategy", 0.0) for y in years]
            c3_v = [c3_cand.annual_returns[y].get("strategy", 0.0) for y in years] if c3_cand else c2_v
            ew_v = [ew_br_bm.annual_returns.get(y, 0.0) for y in years] if ew_br_bm else [champ_cand.annual_returns[y].get("ew_pit", 0.0) for y in years]
            spy_v = [spy_bm.annual_returns.get(y, 0.0) for y in years] if spy_bm else [champ_cand.annual_returns[y].get("spy", 0.0) for y in years]

            b1 = ax_yr.bar(x - 1.5 * w, c2_v, width=w, label=f"{champ_id} ★", color="#2563EB", alpha=0.92)
            ax_yr.bar(x - 0.5 * w, c3_v, width=w, label="C3_RegimeManaged", color="#7C3AED", alpha=0.88)
            ax_yr.bar(x + 0.5 * w, ew_v, width=w, label="Dynamic EW-200 (Broad)", color="#D97706", alpha=0.88)
            b4 = ax_yr.bar(x + 1.5 * w, spy_v, width=w, label="SPY Baseline", color="#059669", alpha=0.88)

            for bar_col, vals in [(b1, c2_v), (b4, spy_v)]:
                for rect, val in zip(bar_col, vals, strict=True):
                    va = "bottom" if val >= 0 else "top"
                    y_txt = (val + 0.8) if val >= 0 else (val - 0.8)
                    ax_yr.text(rect.get_x() + rect.get_width() / 2.0, y_txt, f"{val:+.1f}%", ha="center", va=va, fontsize=7.5, fontweight="bold")

            all_v = c2_v + c3_v + ew_v + spy_v
            y_hi = max(max(all_v, default=20.0) * 1.25, 42.0)
            y_lo = min(min(all_v, default=-10.0) * 1.20, -24.0)
            ax_yr.set_ylim(y_lo, y_hi)

            if "2022" in years:
                sep = years.index("2022") - 0.5
                ax_yr.axvline(sep, color="#DC2626", linestyle="--", linewidth=1.4, alpha=0.85)
                ax_yr.text(sep - 0.12, y_hi * 0.96, "IS (2017–2021) ◄", ha="right", va="top", color="#475569", fontweight="bold", fontsize=9.0)
                ax_yr.text(sep + 0.12, y_hi * 0.96, "► OOS Holdout (2022–2026)", ha="left", va="top", color="#DC2626", fontweight="bold", fontsize=9.0)

            ax_yr.axhline(0.0, color="#64748B", linewidth=0.8)
            x_lbls = [f"{y}\n(YTD)" if y == years[-1] and report.oos_window[1] < f"{y}-12-31" else y for y in years]
            ax_yr.set_xticks(x)
            ax_yr.set_xticklabels(x_lbls, fontsize=9.5, fontweight="bold")
            ax_yr.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
            ax_yr.set_ylabel("Calendar Return (%)", fontweight="bold")
            ax_yr.set_title(f"Calendar-Year Performance: {champ_id} vs C3, EW-200, and SPY", fontsize=12, fontweight="bold", pad=32)
            ax_yr.legend(loc="lower center", bbox_to_anchor=(0.5, 1.008), ncol=4, framealpha=0.95, fontsize=9.0)

        # Bottom: Rolling 252-day Excess Return vs SPY & EW-PIT-SP500 & EW-Broad-200
        if champ_series and len(champ_series.result.equity_curve) > 255:
            c2_arr = np.asarray([pt.equity for pt in champ_series.result.equity_curve], dtype=float)
            spy_arr = np.asarray(_extract_spy_equities(champ_series, init_cap), dtype=float)
            ew_pit_arr = np.asarray(champ_series.ew_universe_equities, dtype=float)

            win = 252
            r_dates = dates[win:]
            c2_1y = (c2_arr[win:] / np.maximum(c2_arr[:-win], 1e-6) - 1.0) * 100.0
            spy_1y = (spy_arr[win:] / np.maximum(spy_arr[:-win], 1e-6) - 1.0) * 100.0
            ewp_1y = (ew_pit_arr[win:] / np.maximum(ew_pit_arr[:-win], 1e-6) - 1.0) * 100.0

            diff_spy = c2_1y - spy_1y
            diff_ewp = c2_1y - ewp_1y
            ax_roll.plot(r_dates, diff_spy, color="#059669", linewidth=1.8, label=f"{champ_id} - SPY (Rolling 252d Alpha)")
            ax_roll.plot(r_dates, diff_ewp, color="#0EA5E9", linewidth=1.6, linestyle="--", label=f"{champ_id} - EW PIT-SP500 (Rolling 252d Alpha)")
            if bm_br_series and len(bm_br_series.ew_universe_equities) == len(dates):
                ewb_arr = np.asarray(bm_br_series.ew_universe_equities, dtype=float)
                diff_ewb = c2_1y - ((ewb_arr[win:] / np.maximum(ewb_arr[:-win], 1e-6) - 1.0) * 100.0)
                ax_roll.plot(r_dates, diff_ewb, color="#D97706", linewidth=1.5, linestyle=":", label=f"{champ_id} - EW-200 Broad (Rolling 252d Alpha)")

            ax_roll.fill_between(r_dates, diff_spy, 0.0, where=(diff_spy >= 0.0), color="#10B981", alpha=0.14)
            ax_roll.fill_between(r_dates, diff_spy, 0.0, where=(diff_spy < 0.0), color="#EF4444", alpha=0.14)
            ax_roll.axhline(0.0, color="#334155", linewidth=1.0)
            if r_dates[0] < oos_start_date < r_dates[-1]:
                ax_roll.axvline(oos_start_date, color="#DC2626", linestyle="--", linewidth=1.4, alpha=0.85)
            pct_pos_spy = float(np.mean(diff_spy > 0.0) * 100.0)
            ax_roll.set_title(f"Rolling 12-Month (252d) Excess Return vs Indices (Beat SPY on {pct_pos_spy:.1f}% of rolling 1Y windows)", fontsize=12, fontweight="bold")
            ax_roll.set_ylabel("Rolling 1Y Alpha (%)", fontweight="bold")
            ax_roll.set_xlabel("Date", fontweight="bold")
            ax_roll.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
            ax_roll.xaxis.set_major_locator(mdates.YearLocator())
            ax_roll.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax_roll.legend(loc="upper left", fontsize=8.8, framealpha=0.92)

        fig.tight_layout()
        fig.savefig(p3, dpi=200, bbox_inches="tight")
        saved_paths.append(p3)
    finally:
        plt.close(fig)

    # --- Plot 4: Stress Suite & 15-Path CPCV ---
    p4 = output_dir / "v4_04_stress_and_cpcv.png"
    fig, (ax_stress, ax_cpcv) = plt.subplots(1, 2, figsize=(16, 6.8))
    try:
        fig.patch.set_facecolor("#FFFFFF")
        _style_ax(ax_stress)
        _style_ax(ax_cpcv)

        stress_items: list[tuple[str, float, float, float, str]] = [
            (
                f"{champ_id} (Base PIT SP500-200)",
                champ_cand.full_metrics.cagr_pct if champ_cand else 0.0,
                champ_cand.full_metrics.excess_sharpe_ratio if champ_cand else 0.0,
                abs(champ_cand.full_metrics.max_drawdown_pct) if champ_cand else 0.0,
                "#2563EB",
            )
        ]
        for sr in report.stress_rows:
            c = "#7C3AED" if sr.stress_id.startswith("C3") else ("#EF4444" if "Delist" in sr.stress_id or "Slip" in sr.stress_id else "#3B82F6")
            stress_items.append((
                sr.label,
                sr.full_metrics.cagr_pct,
                sr.full_metrics.excess_sharpe_ratio,
                abs(sr.full_metrics.max_drawdown_pct),
                c,
            ))
        if ew_br_bm:
            stress_items.append(("Dynamic EW-200 (Broad)", ew_br_bm.full_cagr_pct, ew_br_bm.full_excess_sharpe, abs(ew_br_bm.full_max_drawdown_pct), "#D97706"))
        if spy_bm:
            stress_items.append(("SPY Market Baseline", spy_bm.full_cagr_pct, spy_bm.full_excess_sharpe, abs(spy_bm.full_max_drawdown_pct), "#059669"))

        y_s = np.arange(len(stress_items))
        s_cagrs = [it[1] for it in stress_items]
        ax_stress.barh(y_s, s_cagrs, height=0.58, color=[it[4] for it in stress_items], alpha=0.88, edgecolor="#1E293B")
        for idx, (_lbl, cg, xs, dd, _) in enumerate(stress_items):
            ax_stress.text(cg + 0.35, idx, f"{cg:+.1f}% | XS {xs:.2f} | DD -{dd:.1f}%", va="center", ha="left", fontsize=7.8, fontweight="bold")
        ax_stress.set_xlim(0.0, max(max(s_cagrs, default=20.0) * 1.48, 25.0))
        ax_stress.set_yticks(y_s)
        ax_stress.set_yticklabels([it[0] for it in stress_items], fontsize=8.3, fontweight="bold")
        ax_stress.invert_yaxis()
        ax_stress.set_xlabel("Full 10-Year CAGR (%)", fontweight="bold")
        ax_stress.set_title("Execution, Cost, Delisting & Universe Stress Suite", fontweight="bold", fontsize=11)

        cpcv = report.cpcv_full_summary
        if cpcv and cpcv.fold_metrics:
            sharpes = [m.sharpe_ratio for m in cpcv.fold_metrics]
            xs_sharpes = [m.excess_sharpe_ratio for m in cpcv.fold_metrics]
            y_p = np.arange(len(sharpes))
            ax_cpcv.barh(y_p - 0.18, sharpes, height=0.35, label="Sharpe Ratio", color="#3B82F6", alpha=0.88)
            ax_cpcv.barh(y_p + 0.18, xs_sharpes, height=0.35, label="Excess Sharpe (vs T-Bills)", color="#10B981", alpha=0.88)
            ax_cpcv.axvline(cpcv.median_sharpe, color="#1D4ED8", linestyle="--", linewidth=1.2, label=f"Median Sharpe ({cpcv.median_sharpe:.2f})")
            ax_cpcv.axvline(cpcv.median_excess_sharpe, color="#047857", linestyle="--", linewidth=1.2, label=f"Median Excess ({cpcv.median_excess_sharpe:.2f})")
            ax_cpcv.axvline(0.0, color="#64748B", linewidth=0.8)
            ax_cpcv.set_yticks(y_p)
            ax_cpcv.set_yticklabels([f"Path {i + 1}" for i in y_p], fontsize=8.2)
            ax_cpcv.set_xlim(0.0, max(max(sharpes, default=1.0) * 1.28, 1.80))
            ax_cpcv.invert_yaxis()
            ax_cpcv.set_xlabel("Ratio", fontweight="bold")
            ax_cpcv.set_title(
                f"15-Path CPCV ({cpcv.n_groups} Groups, k={cpcv.k_test_groups}, {cpcv.purge_days}d Purge, {cpcv.embargo_days}d Embargo)\nPositive Paths: {cpcv.pct_positive_paths:.1f}% | Min Sharpe: {cpcv.min_sharpe:.2f}",
                fontweight="bold",
                fontsize=10.5,
            )
            ax_cpcv.legend(loc="lower right", fontsize=8.1, framealpha=0.95)
        else:
            ax_cpcv.text(0.5, 0.5, "CPCV Skipped", ha="center", va="center", color="#94A3B8")

        fig.suptitle(f"{champ_id} Robustness Validation: Severe Stress Suite & Purged/Embargoed CPCV", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(p4, dpi=200, bbox_inches="tight")
        saved_paths.append(p4)
    finally:
        plt.close(fig)

    # --- Plot 5: Parameter Stability Plateau Heatmap ---
    p5 = output_dir / "v4_05_parameter_stability_plateau.png"
    fig, (ax_hm_sr, ax_hm_cg) = plt.subplots(1, 2, figsize=(15, 6.2))
    try:
        fig.patch.set_facecolor("#FFFFFF")
        if report.plateau_cells:
            ks = sorted({c.top_k_long for c in report.plateau_cells})
            bufs = sorted({c.hold_rank_buffer for c in report.plateau_cells})
            mat_sr = np.zeros((len(ks), len(bufs)), dtype=float)
            mat_cg = np.zeros((len(ks), len(bufs)), dtype=float)
            mat_oos_sr = np.zeros((len(ks), len(bufs)), dtype=float)
            lookup = {(c.top_k_long, c.hold_rank_buffer): c for c in report.plateau_cells}
            for i, kv in enumerate(ks):
                for j, bv in enumerate(bufs):
                    cell = lookup.get((kv, bv))
                    if cell is not None:
                        mat_sr[i, j] = cell.full_excess_sharpe
                        mat_cg[i, j] = cell.full_cagr_pct
                        mat_oos_sr[i, j] = cell.oos_excess_sharpe

            im1 = ax_hm_sr.imshow(mat_sr, cmap="YlGnBu", aspect="auto")
            im2 = ax_hm_cg.imshow(mat_cg, cmap="YlGn", aspect="auto")
            fig.colorbar(im1, ax=ax_hm_sr, fraction=0.046, pad=0.04, label="10Y Excess Sharpe")
            fig.colorbar(im2, ax=ax_hm_cg, fraction=0.046, pad=0.04, label="10Y CAGR (%)")

            sr_mid = float(np.min(mat_sr) + 0.60 * (np.max(mat_sr) - np.min(mat_sr)))
            cg_mid = float(np.min(mat_cg) + 0.62 * (np.max(mat_cg) - np.min(mat_cg)))
            for i, kv in enumerate(ks):
                for j, bv in enumerate(bufs):
                    tc1 = "white" if mat_sr[i, j] > sr_mid else "#0F172A"
                    tc2 = "white" if mat_cg[i, j] > cg_mid else "#0F172A"
                    ax_hm_sr.text(j, i, f"{mat_sr[i, j]:.3f}\n(OOS {mat_oos_sr[i, j]:.2f})", ha="center", va="center", color=tc1, fontsize=8.5, fontweight="bold")
                    ax_hm_cg.text(j, i, f"{mat_cg[i, j]:+.1f}%", ha="center", va="center", color=tc2, fontsize=9.0, fontweight="bold")

            for ax in (ax_hm_sr, ax_hm_cg):
                ax.set_xticks(np.arange(len(bufs)))
                ax.set_xticklabels([f"buf={b}" for b in bufs], fontweight="bold")
                ax.set_yticks(np.arange(len(ks)))
                ax.set_yticklabels([f"k={k}" for k in ks], fontweight="bold")
                ax.set_xlabel("Hold Rank Hysteresis Buffer", fontweight="bold")
                ax.set_ylabel("Target Long Positions (top_k_long)", fontweight="bold")

            spy_bm_p5 = next((b for b in report.benchmarks if b.benchmark_id == "SPY"), None)
            ew_br_bm_p5 = next(
                (b for b in report.benchmarks if b.benchmark_id == "EW_BROAD_200"), None
            )
            spy_sr_str = f"{spy_bm_p5.full_excess_sharpe:.3f}" if spy_bm_p5 is not None else "N/A"
            ew_sr_str = f"{ew_br_bm_p5.full_excess_sharpe:.3f}" if ew_br_bm_p5 is not None else "N/A"
            spy_cg_str = f"{spy_bm_p5.full_cagr_pct:+.1f}%" if spy_bm_p5 is not None else "N/A"
            ew_cg_str = f"{ew_br_bm_p5.full_cagr_pct:+.1f}%" if ew_br_bm_p5 is not None else "N/A"
            ax_hm_sr.set_title(
                f"10Y Excess Sharpe Plateau (vs SPY {spy_sr_str}, EW-200 {ew_sr_str})",
                fontweight="bold",
                fontsize=11,
            )
            ax_hm_cg.set_title(
                f"10Y CAGR (%) Plateau (vs SPY {spy_cg_str}, EW-200 {ew_cg_str})",
                fontweight="bold",
                fontsize=11,
            )
        else:
            ax_hm_sr.text(0.5, 0.5, "Plateau Grid Skipped", ha="center", va="center", color="#94A3B8")
            ax_hm_cg.text(0.5, 0.5, "Plateau Grid Skipped", ha="center", va="center", color="#94A3B8")

        fig.suptitle(f"{champ_id} Parameter Stability Plateau: top_k_long × hold_rank_buffer", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(p5, dpi=200, bbox_inches="tight")
        saved_paths.append(p5)
    finally:
        plt.close(fig)

    return saved_paths


def render_v4_monthly_markdown(report: V4MonthlyResearchReport) -> str:
    """Render concise Markdown research summary report for V4 monthly index beater."""
    lines = [
        "# V4 Research: Monthly-Rebalance Dynamic US Equity Index-Beater (`2017–2026`)",
        "",
        "## 1. Universe, Safeguards & Protocol",
        "",
        f"- **Pool / Bars / Delisted:** `{report.pool_total}` pool equities, `{report.bars_loaded}` cached assets (`{report.core_etfs_loaded}/{len(V3_CORE_ETFS)}` Core ETFs), `{report.delisted_count}` delisted/acquired names (`SIVB`, `FRC`, `TWTR`, `ATVI`, etc.)",
        f"- **Monthly PIT Schedule:** `{report.reconstitution_months}` months on strict `pit_sp500_mddv` (`avg_active_universe_size={report.avg_active_universe_size:.1f}`, zero S&P 400 survivorship bias)",
        f"- **Windows:** IS `{report.is_window[0]}..{report.is_window[1]}` | Strict OOS `{report.oos_window[0]}..{report.oos_window[1]}` | Full 10Y `{report.full_window[0]}..{report.full_window[1]}`",
        f"- **Selected Champion:** `{report.champion_id}` (`1.0x` gross long-only, monthly rebalance, cumulative discovery trials tracked = `{report.n_is_trials_cumulative}`)",
        "",
        "## 2. Locked Candidates vs Benchmarks (Net of 10 bps/side + SEC Fees + Delisting Haircuts)",
        "",
        "| Strategy / Benchmark | IS CAGR | IS XS-SR | IS MaxDD | OOS CAGR | OOS XS-SR | OOS MaxDD | 10Y CAGR | 10Y XS-SR | 10Y MaxDD | Turn |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for c in report.candidates:
        star = " ★" if c.candidate_id == report.champion_id else ""
        lines.append(
            f"| **{c.candidate_id}{star}** | {c.is_metrics.cagr_pct:+.2f}% | {c.is_metrics.excess_sharpe_ratio:.3f} | {c.is_metrics.max_drawdown_pct:.2f}% | "
            f"{c.oos_metrics.cagr_pct:+.2f}% | {c.oos_metrics.excess_sharpe_ratio:.3f} | {c.oos_metrics.max_drawdown_pct:.2f}% | "
            f"{c.full_metrics.cagr_pct:+.2f}% | {c.full_metrics.excess_sharpe_ratio:.3f} | {c.full_metrics.max_drawdown_pct:.2f}% | {c.full_metrics.annual_turnover:.2f}x |"
        )
    for b in report.benchmarks:
        lines.append(
            f"| *{b.name}* | {b.is_cagr_pct:+.2f}% | {b.is_excess_sharpe:.3f} | {b.is_max_drawdown_pct:.2f}% | "
            f"{b.oos_cagr_pct:+.2f}% | {b.oos_excess_sharpe:.3f} | {b.oos_max_drawdown_pct:.2f}% | "
            f"{b.full_cagr_pct:+.2f}% | {b.full_excess_sharpe:.3f} | {b.full_max_drawdown_pct:.2f}% | — |"
        )

    lines.extend(["", f"> {report.selection_reason}", ""])
    if report.cpcv_full_summary:
        cpcv = report.cpcv_full_summary
        lines.extend([
            "## 3. CPCV & DSR Validation",
            "",
            f"- **15-Path CPCV (`6` groups, `k=2`, `{cpcv.purge_days}d` purge, `{cpcv.embargo_days}d` embargo):** Median Sharpe `{cpcv.median_sharpe:.3f}` (IQR `[{cpcv.p25_sharpe:.3f}, {cpcv.p75_sharpe:.3f}]`, Min `{cpcv.min_sharpe:.3f}`, Max `{cpcv.max_sharpe:.3f}`), Median Excess Sharpe `{cpcv.median_excess_sharpe:.3f}`, **`{cpcv.pct_positive_paths:.1f}%` positive paths**.",
        ])
        champ = next((x for x in report.candidates if x.candidate_id == report.champion_id), None)
        if champ:
            lines.append(
                f"- **Deflated Sharpe (`DSR`):** Full-10Y Shortlist (`n={report.n_shortlist}`) `{champ.dsr_shortlist:.4f}` | OOS Shortlist (`n={report.n_shortlist}`) `{champ.dsr_oos_shortlist:.4f}` | Full-10Y Cumulative (`n={report.n_is_trials_cumulative}`) `{champ.dsr_all_trials:.4f}`."
            )

    if report.stress_rows:
        lines.extend([
            "",
            "## 4. Stress Matrix",
            "",
            "| Variant | Universe | Timing | Slippage | Full CAGR | Full XS-SR | Full MaxDD | OOS CAGR | OOS XS-SR |",
            "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for sr in report.stress_rows:
            lines.append(
                f"| {sr.label} | {sr.universe_mode} ({sr.top_n}) | {sr.execution_timing} | {sr.equity_slippage_bps:.0f} bps | "
                f"{sr.full_metrics.cagr_pct:+.2f}% | {sr.full_metrics.excess_sharpe_ratio:.3f} | {sr.full_metrics.max_drawdown_pct:.2f}% | "
                f"{sr.oos_metrics.cagr_pct:+.2f}% | {sr.oos_metrics.excess_sharpe_ratio:.3f} |"
            )

    if report.plateau_cells:
        lines.extend([
            "",
            "## 5. Parameter Stability Plateau (`top_k_long × hold_rank_buffer`)",
            "",
            "| `top_k_long` | `hold_rank_buffer` | Full 10Y CAGR | Full XS-SR | Full MaxDD | OOS CAGR | OOS XS-SR |",
            "| :---: | :---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for pc in report.plateau_cells:
            lines.append(
                f"| {pc.top_k_long} | {pc.hold_rank_buffer} | {pc.full_cagr_pct:+.2f}% | {pc.full_excess_sharpe:.3f} | {pc.full_max_drawdown_pct:.2f}% | "
                f"{pc.oos_cagr_pct:+.2f}% | {pc.oos_excess_sharpe:.3f} |"
            )

    return "\n".join(lines) + "\n"
