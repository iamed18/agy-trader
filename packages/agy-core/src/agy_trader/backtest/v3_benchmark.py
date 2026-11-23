from __future__ import annotations

import copy
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from agy_trader.backtest.engine import BacktestEngine, BacktestRunSeries
from agy_trader.backtest.metrics import (
    BacktestMetrics,
    CPCVSummary,
)
from agy_trader.backtest.sp500_cohorts import atomic_write_text
from agy_trader.backtest.v3_universe import (
    HISTORICAL_DELISTED_SECTORS,
    SECTOR_TO_ETF,
    V3_CORE_ETFS,
    V3EquityCandidate,
    build_dynamic_pit_universe_schedule,
)
from agy_trader.core.config import AppConfig, RiskConfig, StrategyConfig
from agy_trader.core.models import AssetType, MacroPoint, UniverseSymbol, ensure_utc

logger = logging.getLogger(__name__)


def _ensure_matplotlib_agg() -> Any:
    """Lazily configure matplotlib Agg backend without clobbering existing configuration."""
    if "matplotlib.pyplot" not in sys.modules:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


class TrackSpec(BaseModel):
    """Specification for an incremental V3 strategy track."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    track_id: str
    name: str
    description: str
    strategy_overrides: dict[str, Any] = Field(default_factory=dict)
    risk_overrides: dict[str, Any] = Field(default_factory=dict)
    use_sector_residual: bool = False


class WindowMetrics(BaseModel):
    """Summarized metrics for a single backtest time window."""

    model_config = ConfigDict(allow_inf_nan=False)

    cagr_pct: float
    sharpe_ratio: float
    excess_sharpe_ratio: float
    sortino_ratio: float = 0.0
    max_drawdown_pct: float
    annual_turnover: float
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    total_trades: int = 0
    trading_days: int = 0
    final_equity: float = 0.0

    @classmethod
    def from_metrics(cls, m: BacktestMetrics) -> WindowMetrics:
        return cls(
            cagr_pct=m.cagr_pct,
            sharpe_ratio=m.sharpe_ratio,
            excess_sharpe_ratio=m.excess_sharpe_ratio,
            sortino_ratio=m.sortino_ratio,
            max_drawdown_pct=m.max_drawdown_pct,
            annual_turnover=m.annual_turnover,
            win_rate_pct=m.win_rate_pct,
            profit_factor=m.profit_factor,
            calmar_ratio=m.calmar_ratio,
            total_trades=m.closed_trades,
            trading_days=m.trading_days,
            final_equity=m.final_equity,
        )


class TrackComparisonRow(BaseModel):
    """Comparative row capturing IS, OOS, and Full 10Y metrics for a track alongside benchmarks."""

    model_config = ConfigDict(allow_inf_nan=False)

    track_id: str
    name: str
    description: str
    is_metrics: BacktestMetrics
    oos_metrics: BacktestMetrics
    full_metrics: BacktestMetrics
    ew_is_cagr_pct: float
    ew_oos_cagr_pct: float
    ew_full_cagr_pct: float
    spy_is_cagr_pct: float
    spy_oos_cagr_pct: float
    spy_full_cagr_pct: float
    ew_is_excess_sharpe: float = 0.0
    ew_oos_excess_sharpe: float = 0.0
    ew_full_excess_sharpe: float = 0.0
    spy_is_excess_sharpe: float = 0.0
    spy_oos_excess_sharpe: float = 0.0
    spy_full_excess_sharpe: float = 0.0
    annual_returns: dict[str, dict[str, float]] = Field(default_factory=dict)


class V3BenchmarkReport(BaseModel):
    """Comprehensive artifact summarizing V3 10-year backtest benchmark results."""

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
    tracks: list[TrackComparisonRow]
    is_selected_track_id: str
    is_selected_reason: str
    sp500_only_stress: TrackComparisonRow | None = None
    delisting_wipeout_stress: TrackComparisonRow | None = None
    cpcv_full_summary: CPCVSummary | None = None
    dsr_n_trials: int = 12
    plot_paths: list[str] = Field(default_factory=list)


def build_default_track_specs() -> list[TrackSpec]:
    """Return the progressive 6-track ladder (T0 through T5) for V3 research."""
    t0 = TrackSpec(
        track_id="T0",
        name="v0.2 Prod Baseline",
        description="v0.2 Prod Baseline on Broad Top-200 (k=6, 1d reb, 0.70 corr, 3.5x ATR, inv_atr, legacy_stacked)",
        strategy_overrides={},
        risk_overrides={},
        use_sector_residual=False,
    )
    t1 = TrackSpec(
        track_id="T1",
        name="+ Sizing & Risk (Tr 1&4)",
        description="Broad Single-Stock Sizing & Risk (top_k=12, capped_inv_vol [18%-45%], vol_target=14%, score_tilt=0.75, stop=6.0x ATR, dd_cooldown=60d)",
        strategy_overrides={
            "top_k_long": 12,
            "max_pair_corr_60d": 0.85,
            "corr_penalty_eta": 0.05,
            "vol_sizing_mode": "capped_inv_vol",
            "vol_sizing_floor": 0.18,
            "vol_sizing_cap": 0.45,
            "score_tilt_power": 0.75,
            "portfolio_vol_target_ann": 0.14,
        },
        risk_overrides={
            "max_long_position_pct": 0.12,
            "max_sector_gross_pct": 0.35,
            "stop_loss_atr_mult_long": 6.0,
            "max_drawdown_cooldown_days": 60,
        },
        use_sector_residual=False,
    )
    t2_strat = dict(t1.strategy_overrides)
    t2_strat.update({"rebalance_every_n_days": 5, "rebalance_tranches": 4})
    t2 = TrackSpec(
        track_id="T2",
        name="+ 4-Tranche Weekly (Tr 5)",
        description="4-Tranche Staggered Weekly Rebalancing (rebalance_every_n_days=5, rebalance_tranches=4)",
        strategy_overrides=t2_strat,
        risk_overrides=dict(t1.risk_overrides),
        use_sector_residual=False,
    )
    t3_strat = dict(t2_strat)
    t3_strat.update({
        "momentum_mode": "m2_residual",
        "use_sector_residual": True,
        "w_fip": 0.20,
        "w_jump_penalty": 0.10,
    })
    t3 = TrackSpec(
        track_id="T3",
        name="+ Sector Residual & FIP (Tr 2)",
        description="2-Factor [SPY, SectorETF] Residual + Frog-in-the-Pan + Jump Penalty",
        strategy_overrides=t3_strat,
        risk_overrides=dict(t2.risk_overrides),
        use_sector_residual=True,
    )
    t4_strat = dict(t3_strat)
    t4_strat.update({"regime_mode": "daniel_moskowitz"})
    t4 = TrackSpec(
        track_id="T4",
        name="+ Smooth Regime (Tr 6)",
        description="Daniel-Moskowitz Smooth Bear-Panic Regime (regime_mode='daniel_moskowitz')",
        strategy_overrides=t4_strat,
        risk_overrides=dict(t3.risk_overrides),
        use_sector_residual=True,
    )
    t5_strat = dict(t4_strat)
    t5_strat.update({"finalist_pool_size": 40, "w_pead": 0.15})
    t5 = TrackSpec(
        track_id="T5",
        name="+ Finalist & PEAD (Tr 3)",
        description="Stage-2 Top-40 Finalist Funnel + PIT PEAD Drift",
        strategy_overrides=t5_strat,
        risk_overrides=dict(t4.risk_overrides),
        use_sector_residual=True,
    )
    return [t0, t1, t2, t3, t4, t5]


def _compute_annual_returns(
    timestamps: Sequence[datetime],
    series_map: Mapping[str, Sequence[float] | np.ndarray],
    initial_capital: float,
) -> dict[str, dict[str, float]]:
    """Compute per-calendar-year percentage returns for aligned equity series."""
    if not timestamps:
        return {}

    year_last_idx: dict[int, int] = {}
    for idx, ts in enumerate(timestamps):
        year_last_idx[ts.year] = idx

    years = sorted(year_last_idx.keys())
    annual: dict[str, dict[str, float]] = {}
    prev_end_idx: int | None = None
    for yr in years:
        end_idx = year_last_idx[yr]
        yr_dict: dict[str, float] = {}
        for name, raw_arr in series_map.items():
            arr = np.asarray(raw_arr, dtype=float)
            if len(arr) == 0:
                yr_dict[name] = 0.0
                continue
            anchor = initial_capital if prev_end_idx is None else float(arr[prev_end_idx])
            end_val = float(arr[end_idx])
            ret_pct = ((end_val / anchor) - 1.0) * 100.0 if anchor > 0.0 else 0.0
            yr_dict[name] = round(ret_pct, 2)
        annual[str(yr)] = yr_dict
        prev_end_idx = end_idx
    return annual


def _calc_series_cagr_pct(equities: Sequence[float], n_days: int) -> float:
    """Compute annualized CAGR in percent from an equity curve and trading days."""
    if len(equities) < 2 or equities[0] <= 0.0 or n_days <= 0:
        return 0.0
    val = (float(equities[-1]) / float(equities[0])) ** (252.0 / float(n_days)) - 1.0
    return round(val * 100.0, 2)


def _calc_excess_sharpe(
    equities: Sequence[float], daily_rf_rates: Sequence[float]
) -> float:
    """Compute annualized excess Sharpe ratio over PIT daily T-bill rates."""
    arr = np.asarray(equities, dtype=float)
    if arr.size < 2 or arr[0] <= 0.0 or np.any(arr <= 0.0):
        return 0.0
    rets = np.diff(arr) / arr[:-1]
    rets = rets[np.isfinite(rets)]
    if rets.size < 2:
        return 0.0
    rf_arr = np.asarray(daily_rf_rates, dtype=float)
    if rf_arr.size >= rets.size + 1:
        excess = rets - rf_arr[1 : 1 + rets.size]
    elif rf_arr.size == rets.size:
        excess = rets - rf_arr
    else:
        excess = rets
    excess = excess[np.isfinite(excess)]
    if excess.size < 2:
        return 0.0
    std_e = float(np.std(excess, ddof=1))
    if not math.isfinite(std_e) or std_e <= 1e-12:
        return 0.0
    return round(float(np.mean(excess) / std_e * math.sqrt(252.0)), 3)


def _extract_spy_equities(run_series: BacktestRunSeries, initial_capital: float) -> list[float]:
    """Convert benchmark close prices into an aligned starting-capital equity curve."""
    closes = run_series.benchmark_closes
    if not closes or closes[0] <= 0.0:
        return [initial_capital] * len(closes)
    c0 = closes[0]
    return [initial_capital * (c / c0) for c in closes]


def select_is_winning_track(tracks: Sequence[TrackComparisonRow]) -> tuple[str, str]:
    """Select winning track strictly by highest In-Sample (IS 2017–2021) excess Sharpe ratio."""
    candidates = [t for t in tracks if t.track_id != "T0" and t.is_metrics.trading_days > 0]
    if not candidates:
        candidates = [t for t in tracks if t.is_metrics.trading_days > 0]
    if not candidates:
        fallback_id = tracks[0].track_id if tracks else "T0"
        return fallback_id, "Fallback: no track produced valid trading days"

    winner = max(
        candidates,
        key=lambda t: (t.is_metrics.excess_sharpe_ratio, t.is_metrics.sharpe_ratio),
    )
    reason = (
        f"Highest In-Sample (IS 2017–2021) excess Sharpe ratio ({winner.is_metrics.excess_sharpe_ratio:.3f}) "
        f"among progressive tracks T1–T5 (IS Sharpe: {winner.is_metrics.sharpe_ratio:.3f}, "
        f"IS CAGR: {winner.is_metrics.cagr_pct:+.2f}%, IS MaxDD: {winner.is_metrics.max_drawdown_pct:.1f}%). "
        f"Selected strictly without inspecting OOS performance."
    )
    return winner.track_id, reason


def _build_variant_app_config(
    base_config: AppConfig,
    strat_overrides: Mapping[str, Any],
    risk_overrides: Mapping[str, Any],
) -> AppConfig:
    """Create a validated AppConfig with applied strategy and risk overrides."""
    strat_dict = base_config.strategy.model_dump()
    strat_dict.update(strat_overrides)
    risk_dict = base_config.risk.model_dump()
    risk_dict.update(risk_overrides)

    new_strat = StrategyConfig.model_validate(strat_dict)
    new_risk = RiskConfig.model_validate(risk_dict)

    return AppConfig(
        strategy=new_strat,
        risk=new_risk,
        backtest=copy.deepcopy(base_config.backtest),
        llm_quota=copy.deepcopy(base_config.llm_quota),
    )


def run_v3_benchmark(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    pool: Sequence[V3EquityCandidate],
    *,
    app_config: AppConfig | None = None,
    macro_points: Sequence[MacroPoint] | None = None,
    earnings_by_symbol: Mapping[str, Sequence[date]] | None = None,
    is_start: date = date(2017, 1, 3),
    is_end: date = date(2021, 12, 31),
    oos_start: date = date(2022, 1, 3),
    oos_end: date = date(2026, 11, 19),
    top_n: int = 200,
    hysteresis: int = 50,
    n_trials: int = 12,
    run_cpcv: bool = True,
    run_stress: bool = True,
    track_specs: Sequence[TrackSpec] | None = None,
    output_dir: Path | None = None,
) -> tuple[V3BenchmarkReport, dict[str, Any]]:
    """Execute progressive 6-track benchmark ladder with strict anti-peeking selection and stress checks."""
    base_app_cfg = app_config or AppConfig()
    specs = list(track_specs or build_default_track_specs())

    # 1. Monthly dynamic schedule on Broad MDDV
    broad_schedule = build_dynamic_pit_universe_schedule(
        bars_by_symbol,
        pool,
        mode="broad_mddv",
        top_n=top_n,
        hysteresis_buffer=hysteresis,
    )
    if not broad_schedule:
        raise ValueError("Dynamic schedule returned 0 reconstitution dates")

    eligible_union: set[str] = set().union(*broad_schedule.values()) if broad_schedule else set()
    active_syms = eligible_union | set(V3_CORE_ETFS) | {"SPY"}
    active_union_bars = {
        s: bars_by_symbol[s]
        for s in active_syms
        if s in bars_by_symbol and not bars_by_symbol[s].empty
    }

    symbol_sectors = {c.symbol: c.sector for c in pool}
    equity_universe_meta = [
        UniverseSymbol(
            symbol=s,
            asset_type=AssetType.equity,
            sector=symbol_sectors.get(s, "Unknown"),
            shortable_default=True,
        )
        for s in sorted(eligible_union)
    ]

    # Calculate dataset stats
    core_etfs_loaded = sum(1 for etf in V3_CORE_ETFS if etf in active_union_bars)
    bars_loaded_count = len(active_union_bars)
    reconstitutions_count = len(broad_schedule)
    avg_universe_size = (
        float(np.mean([len(v) for v in broad_schedule.values()])) if broad_schedule else 0.0
    )

    spy_df = active_union_bars.get("SPY")
    if spy_df is not None and not spy_df.empty:
        raw_start_str = str(pd.to_datetime(spy_df["timestamp"]).min().date())
        raw_end_str = str(pd.to_datetime(spy_df["timestamp"]).max().date())
    else:
        raw_start_str = is_start.isoformat()
        raw_end_str = oos_end.isoformat()

    # Count delisted/historical names
    delisted_count = 0
    ref_end_date = oos_end
    for c in pool:
        if c.symbol in HISTORICAL_DELISTED_SECTORS:
            delisted_count += 1
            continue
        c_df = bars_by_symbol.get(c.symbol)
        if c_df is not None and not c_df.empty:
            c_max = pd.to_datetime(c_df["timestamp"]).max().date()
            if (ref_end_date - c_max) > timedelta(days=90):
                delisted_count += 1

    tracks_rows: list[TrackComparisonRow] = []
    full_series_by_track: dict[str, BacktestRunSeries] = {}
    init_cap = base_app_cfg.backtest.initial_capital

    for spec in specs:
        track_cfg = _build_variant_app_config(
            base_app_cfg,
            spec.strategy_overrides,
            spec.risk_overrides,
        )
        engine = BacktestEngine(track_cfg, universe=equity_universe_meta)

        # IS Run
        is_series = engine.run_with_series(
            active_union_bars,
            macro_points=macro_points,
            earnings_by_symbol=earnings_by_symbol,
            start_date=is_start,
            end_date=is_end,
            label=f"{spec.track_id}_IS",
            allow_shorting=True,
            use_news=False,
            n_trials=n_trials,
            eligible_symbols_by_date=broad_schedule,
            symbol_sectors=symbol_sectors,
            sector_etf_map=SECTOR_TO_ETF,
        )

        # OOS Run
        oos_series = engine.run_with_series(
            active_union_bars,
            macro_points=macro_points,
            earnings_by_symbol=earnings_by_symbol,
            start_date=oos_start,
            end_date=oos_end,
            label=f"{spec.track_id}_OOS",
            allow_shorting=True,
            use_news=False,
            n_trials=n_trials,
            eligible_symbols_by_date=broad_schedule,
            symbol_sectors=symbol_sectors,
            sector_etf_map=SECTOR_TO_ETF,
        )

        # Full 10Y Run
        full_series = engine.run_with_series(
            active_union_bars,
            macro_points=macro_points,
            earnings_by_symbol=earnings_by_symbol,
            start_date=is_start,
            end_date=oos_end,
            label=f"{spec.track_id}_Full",
            allow_shorting=True,
            use_news=False,
            n_trials=n_trials,
            eligible_symbols_by_date=broad_schedule,
            symbol_sectors=symbol_sectors,
            sector_etf_map=SECTOR_TO_ETF,
        )
        full_series_by_track[spec.track_id] = full_series

        is_spy_eq = _extract_spy_equities(is_series, init_cap)
        oos_spy_eq = _extract_spy_equities(oos_series, init_cap)
        full_spy_eq = _extract_spy_equities(full_series, init_cap)

        ew_is_cagr = _calc_series_cagr_pct(
            is_series.ew_universe_equities, is_series.result.metrics.trading_days
        )
        ew_oos_cagr = _calc_series_cagr_pct(
            oos_series.ew_universe_equities, oos_series.result.metrics.trading_days
        )
        ew_full_cagr = _calc_series_cagr_pct(
            full_series.ew_universe_equities, full_series.result.metrics.trading_days
        )

        spy_is_cagr = _calc_series_cagr_pct(is_spy_eq, is_series.result.metrics.trading_days)
        spy_oos_cagr = _calc_series_cagr_pct(oos_spy_eq, oos_series.result.metrics.trading_days)
        spy_full_cagr = _calc_series_cagr_pct(full_spy_eq, full_series.result.metrics.trading_days)

        full_timestamps = [ensure_utc(pt.timestamp) for pt in full_series.result.equity_curve]
        full_strat_eq = [pt.equity for pt in full_series.result.equity_curve]
        annual_returns = _compute_annual_returns(
            full_timestamps,
            {
                "strategy": full_strat_eq,
                "ew": full_series.ew_universe_equities,
                "spy": full_spy_eq,
            },
            initial_capital=init_cap,
        )

        row = TrackComparisonRow(
            track_id=spec.track_id,
            name=spec.name,
            description=spec.description,
            is_metrics=is_series.result.metrics,
            oos_metrics=oos_series.result.metrics,
            full_metrics=full_series.result.metrics,
            ew_is_cagr_pct=ew_is_cagr,
            ew_oos_cagr_pct=ew_oos_cagr,
            ew_full_cagr_pct=ew_full_cagr,
            spy_is_cagr_pct=spy_is_cagr,
            spy_oos_cagr_pct=spy_oos_cagr,
            spy_full_cagr_pct=spy_full_cagr,
            ew_is_excess_sharpe=_calc_excess_sharpe(
                is_series.ew_universe_equities, is_series.daily_rf_rates
            ),
            ew_oos_excess_sharpe=_calc_excess_sharpe(
                oos_series.ew_universe_equities, oos_series.daily_rf_rates
            ),
            ew_full_excess_sharpe=_calc_excess_sharpe(
                full_series.ew_universe_equities, full_series.daily_rf_rates
            ),
            spy_is_excess_sharpe=_calc_excess_sharpe(is_spy_eq, is_series.daily_rf_rates),
            spy_oos_excess_sharpe=_calc_excess_sharpe(oos_spy_eq, oos_series.daily_rf_rates),
            spy_full_excess_sharpe=_calc_excess_sharpe(full_spy_eq, full_series.daily_rf_rates),
            annual_returns=annual_returns,
        )
        tracks_rows.append(row)

    # Winner selection strictly on IS metrics (zero OOS peeking!)
    is_winner_id, winner_reason = select_is_winning_track(tracks_rows)
    winner_spec = next((s for s in specs if s.track_id == is_winner_id), specs[0])
    winner_cfg = _build_variant_app_config(
        base_app_cfg,
        winner_spec.strategy_overrides,
        winner_spec.risk_overrides,
    )
    winner_engine = BacktestEngine(winner_cfg, universe=equity_universe_meta)

    # T6 Stress Checks
    sp500_stress_row: TrackComparisonRow | None = None
    delist_stress_row: TrackComparisonRow | None = None
    cpcv_summary: CPCVSummary | None = None

    if run_stress:
        # (a) sp500_only_stress
        try:
            sp500_schedule = build_dynamic_pit_universe_schedule(
                bars_by_symbol,
                pool,
                mode="pit_sp500_mddv",
                top_n=top_n,
                hysteresis_buffer=hysteresis,
            )
            sp_is = winner_engine.run_with_series(
                active_union_bars,
                macro_points=macro_points,
                earnings_by_symbol=earnings_by_symbol,
                start_date=is_start,
                end_date=is_end,
                label=f"{is_winner_id}_SP500_IS",
                allow_shorting=True,
                use_news=False,
                n_trials=n_trials,
                eligible_symbols_by_date=sp500_schedule,
                symbol_sectors=symbol_sectors,
                sector_etf_map=SECTOR_TO_ETF,
            )
            sp_oos = winner_engine.run_with_series(
                active_union_bars,
                macro_points=macro_points,
                earnings_by_symbol=earnings_by_symbol,
                start_date=oos_start,
                end_date=oos_end,
                label=f"{is_winner_id}_SP500_OOS",
                allow_shorting=True,
                use_news=False,
                n_trials=n_trials,
                eligible_symbols_by_date=sp500_schedule,
                symbol_sectors=symbol_sectors,
                sector_etf_map=SECTOR_TO_ETF,
            )
            sp_full = winner_engine.run_with_series(
                active_union_bars,
                macro_points=macro_points,
                earnings_by_symbol=earnings_by_symbol,
                start_date=is_start,
                end_date=oos_end,
                label=f"{is_winner_id}_SP500_Full",
                allow_shorting=True,
                use_news=False,
                n_trials=n_trials,
                eligible_symbols_by_date=sp500_schedule,
                symbol_sectors=symbol_sectors,
                sector_etf_map=SECTOR_TO_ETF,
            )
            full_series_by_track["STRESS_SP500"] = sp_full
            sp_full_ts = [ensure_utc(pt.timestamp) for pt in sp_full.result.equity_curve]
            sp_full_strat_eq = [pt.equity for pt in sp_full.result.equity_curve]
            sp_full_spy_eq = _extract_spy_equities(sp_full, init_cap)
            sp500_stress_row = TrackComparisonRow(
                track_id="STRESS_SP500",
                name="Strict PIT S&P 500 Top-200",
                description=f"Winning track {is_winner_id} evaluated strictly on PIT S&P 500 constituents",
                is_metrics=sp_is.result.metrics,
                oos_metrics=sp_oos.result.metrics,
                full_metrics=sp_full.result.metrics,
                ew_is_cagr_pct=_calc_series_cagr_pct(
                    sp_is.ew_universe_equities, sp_is.result.metrics.trading_days
                ),
                ew_oos_cagr_pct=_calc_series_cagr_pct(
                    sp_oos.ew_universe_equities, sp_oos.result.metrics.trading_days
                ),
                ew_full_cagr_pct=_calc_series_cagr_pct(
                    sp_full.ew_universe_equities, sp_full.result.metrics.trading_days
                ),
                spy_is_cagr_pct=_calc_series_cagr_pct(
                    _extract_spy_equities(sp_is, init_cap), sp_is.result.metrics.trading_days
                ),
                spy_oos_cagr_pct=_calc_series_cagr_pct(
                    _extract_spy_equities(sp_oos, init_cap), sp_oos.result.metrics.trading_days
                ),
                spy_full_cagr_pct=_calc_series_cagr_pct(
                    sp_full_spy_eq, sp_full.result.metrics.trading_days
                ),
                annual_returns=_compute_annual_returns(
                    sp_full_ts,
                    {
                        "strategy": sp_full_strat_eq,
                        "ew": sp_full.ew_universe_equities,
                        "spy": sp_full_spy_eq,
                    },
                    initial_capital=init_cap,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("sp500_only_stress failed: %s", exc)

        # (b) delisting_wipeout_stress (100% loss on delisting)
        try:
            delist_risk = dict(winner_spec.risk_overrides)
            delist_risk["delisting_haircut_pct"] = 1.0
            delist_cfg = _build_variant_app_config(
                base_app_cfg,
                winner_spec.strategy_overrides,
                delist_risk,
            )
            delist_engine = BacktestEngine(delist_cfg, universe=equity_universe_meta)
            dl_is = delist_engine.run_with_series(
                active_union_bars,
                macro_points=macro_points,
                earnings_by_symbol=earnings_by_symbol,
                start_date=is_start,
                end_date=is_end,
                label=f"{is_winner_id}_DELIST100_IS",
                allow_shorting=True,
                use_news=False,
                n_trials=n_trials,
                eligible_symbols_by_date=broad_schedule,
                symbol_sectors=symbol_sectors,
                sector_etf_map=SECTOR_TO_ETF,
            )
            dl_oos = delist_engine.run_with_series(
                active_union_bars,
                macro_points=macro_points,
                earnings_by_symbol=earnings_by_symbol,
                start_date=oos_start,
                end_date=oos_end,
                label=f"{is_winner_id}_DELIST100_OOS",
                allow_shorting=True,
                use_news=False,
                n_trials=n_trials,
                eligible_symbols_by_date=broad_schedule,
                symbol_sectors=symbol_sectors,
                sector_etf_map=SECTOR_TO_ETF,
            )
            dl_full = delist_engine.run_with_series(
                active_union_bars,
                macro_points=macro_points,
                earnings_by_symbol=earnings_by_symbol,
                start_date=is_start,
                end_date=oos_end,
                label=f"{is_winner_id}_DELIST100_Full",
                allow_shorting=True,
                use_news=False,
                n_trials=n_trials,
                eligible_symbols_by_date=broad_schedule,
                symbol_sectors=symbol_sectors,
                sector_etf_map=SECTOR_TO_ETF,
            )
            full_series_by_track["STRESS_DELIST_100"] = dl_full
            dl_full_ts = [ensure_utc(pt.timestamp) for pt in dl_full.result.equity_curve]
            dl_full_strat_eq = [pt.equity for pt in dl_full.result.equity_curve]
            dl_full_spy_eq = _extract_spy_equities(dl_full, init_cap)
            delist_stress_row = TrackComparisonRow(
                track_id="STRESS_DELIST_100",
                name="100% Delisting Wipeout Stress",
                description=f"Winning track {is_winner_id} evaluated with 100% haircut on forced delisting exit",
                is_metrics=dl_is.result.metrics,
                oos_metrics=dl_oos.result.metrics,
                full_metrics=dl_full.result.metrics,
                ew_is_cagr_pct=_calc_series_cagr_pct(
                    dl_is.ew_universe_equities, dl_is.result.metrics.trading_days
                ),
                ew_oos_cagr_pct=_calc_series_cagr_pct(
                    dl_oos.ew_universe_equities, dl_oos.result.metrics.trading_days
                ),
                ew_full_cagr_pct=_calc_series_cagr_pct(
                    dl_full.ew_universe_equities, dl_full.result.metrics.trading_days
                ),
                spy_is_cagr_pct=_calc_series_cagr_pct(
                    _extract_spy_equities(dl_is, init_cap), dl_is.result.metrics.trading_days
                ),
                spy_oos_cagr_pct=_calc_series_cagr_pct(
                    _extract_spy_equities(dl_oos, init_cap), dl_oos.result.metrics.trading_days
                ),
                spy_full_cagr_pct=_calc_series_cagr_pct(
                    dl_full_spy_eq, dl_full.result.metrics.trading_days
                ),
                annual_returns=_compute_annual_returns(
                    dl_full_ts,
                    {
                        "strategy": dl_full_strat_eq,
                        "ew": dl_full.ew_universe_equities,
                        "spy": dl_full_spy_eq,
                    },
                    initial_capital=init_cap,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("delisting_wipeout_stress failed: %s", exc)

    if run_cpcv:
        # (c) cpcv_full_summary over full window
        try:
            cpcv_summary = winner_engine.run_cpcv(
                active_union_bars,
                eligible_symbols_by_date=broad_schedule,
                symbol_sectors=symbol_sectors,
                sector_etf_map=SECTOR_TO_ETF,
                macro_points=macro_points,
                start_date=is_start,
                end_date=oos_end,
                n_groups=6,
                k_test_groups=2,
                purge_days=21,
                embargo_days=5,
                allow_shorting=True,
                use_news=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("run_cpcv failed: %s", exc)

    report = V3BenchmarkReport(
        pool_total=len(pool),
        bars_loaded=bars_loaded_count,
        core_etfs_loaded=core_etfs_loaded,
        delisted_count=delisted_count,
        reconstitution_months=reconstitutions_count,
        avg_active_universe_size=round(avg_universe_size, 1),
        raw_bar_start=raw_start_str,
        raw_bar_end=raw_end_str,
        is_window=(is_start.isoformat(), is_end.isoformat()),
        oos_window=(oos_start.isoformat(), oos_end.isoformat()),
        full_window=(is_start.isoformat(), oos_end.isoformat()),
        tracks=tracks_rows,
        is_selected_track_id=is_winner_id,
        is_selected_reason=winner_reason,
        sp500_only_stress=sp500_stress_row,
        delisting_wipeout_stress=delist_stress_row,
        cpcv_full_summary=cpcv_summary,
        dsr_n_trials=n_trials,
        plot_paths=[],
    )

    series_bundle: dict[str, Any] = {
        "full_series_by_track": full_series_by_track,
        "is_start": is_start,
        "is_end": is_end,
        "oos_start": oos_start,
        "oos_end": oos_end,
        "initial_capital": init_cap,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = generate_v3_benchmark_plots(report, series_bundle, output_dir)
        report = report.model_copy(update={"plot_paths": [str(p) for p in plot_paths]})
        atomic_write_text(output_dir / "summary.json", report.model_dump_json(indent=2))
        atomic_write_text(output_dir / "summary.md", render_v3_benchmark_markdown(report))

    return report, series_bundle


def generate_v3_benchmark_plots(
    report: V3BenchmarkReport,
    series_bundle: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Generate 4 clean, high-DPI summary plots for V3 research."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _ensure_matplotlib_agg()
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    full_series = series_bundle.get("full_series_by_track", {})
    init_cap = series_bundle.get("initial_capital", 100000.0)
    saved_paths: list[Path] = []

    def _style_ax(ax: Any) -> None:
        ax.set_facecolor("#FAFAFC")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.55, color="#CBD5E1")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#94A3B8")
        ax.spines["bottom"].set_color("#94A3B8")

    winner_id = report.is_selected_track_id
    t0_series = full_series.get("T0")
    winner_series = full_series.get(winner_id)
    t5_series = full_series.get("T5")

    # Pick representative series for baseline / comparison
    ref_run = winner_series or t0_series or (next(iter(full_series.values())) if full_series else None)
    if ref_run is None or not ref_run.result.equity_curve:
        logger.warning("No equity curve available to generate plots")
        return []

    timestamps = [ensure_utc(pt.timestamp) for pt in ref_run.result.equity_curve]
    dates = [ts.date() for ts in timestamps]

    # --- Plot 1: 10-Year Cumulative Equity & Underwater Drawdown (15x9) ---
    p1 = output_dir / "v3_01_10yr_equity_and_drawdown.png"
    fig, (ax_eq, ax_dd) = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    try:
        fig.patch.set_facecolor("#FFFFFF")
        _style_ax(ax_eq)
        _style_ax(ax_dd)

        # Plot candidate series: T0, Winner, T5, EW-200, SPY
        plot_items = []
        if t0_series:
            eq = [pt.equity for pt in t0_series.result.equity_curve]
            plot_items.append(("T0 Baseline", eq, "#64748B", 1.8, "--"))
        if winner_series:
            eq = [pt.equity for pt in winner_series.result.equity_curve]
            plot_items.append((f"{winner_id} (v0.3 IS Winner)", eq, "#2563EB", 2.6, "-"))
        if t5_series and winner_id != "T5":
            eq = [pt.equity for pt in t5_series.result.equity_curve]
            plot_items.append(("T5 Full Stack", eq, "#7C3AED", 2.0, "-."))

        ew_eq = np.asarray(ref_run.ew_universe_equities, dtype=float)
        if len(ew_eq) == len(dates):
            plot_items.append(("Dynamic EW-200", list(ew_eq), "#D97706", 1.8, ":"))

        spy_eq = _extract_spy_equities(ref_run, init_cap)
        if len(spy_eq) == len(dates):
            plot_items.append(("SPY Market Baseline", spy_eq, "#059669", 1.8, "-"))

        oos_start_date = date.fromisoformat(report.oos_window[0])

        for label, eq_curve, color, lw, ls in plot_items:
            if len(eq_curve) != len(dates):
                continue
            arr = np.asarray(eq_curve, dtype=float)
            ax_eq.plot(dates, arr, label=label, color=color, linewidth=lw, linestyle=ls)

            # Drawdown curve
            running_max = np.maximum.accumulate(arr)
            dd_pct = ((arr - running_max) / np.maximum(running_max, 1e-6)) * 100.0
            ax_dd.plot(dates, dd_pct, label=label, color=color, linewidth=lw, linestyle=ls)

            # Right edge annotation on equity curve
            final_val = arr[-1]
            cagr = ((final_val / arr[0]) ** (252.0 / max(len(arr), 1)) - 1.0) * 100.0
            tag = (
                "EW-200"
                if label.startswith("Dynamic EW-200")
                else ("SPY" if label.startswith("SPY") else label.split()[0])
            )
            ax_eq.text(
                dates[-1] + timedelta(days=25),
                final_val,
                f"{tag}: ${final_val / 1000.0:.0f}k ({cagr:+.1f}%)",
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                color=color,
            )

        # IS vs OOS vertical divider and background shading
        if dates[0] < oos_start_date < dates[-1]:
            ax_eq.axvline(oos_start_date, color="#DC2626", linestyle="--", linewidth=1.5, alpha=0.85)
            ax_dd.axvline(oos_start_date, color="#DC2626", linestyle="--", linewidth=1.5, alpha=0.85)
            ax_eq.axvspan(dates[0], oos_start_date, color="#F1F5F9", alpha=0.55)
            ax_eq.axvspan(oos_start_date, dates[-1], color="#EFF6FF", alpha=0.55)
            ax_dd.axvspan(dates[0], oos_start_date, color="#F1F5F9", alpha=0.55)
            ax_dd.axvspan(oos_start_date, dates[-1], color="#EFF6FF", alpha=0.55)
            y_top = ax_eq.get_ylim()[1]
            ax_eq.text(
                oos_start_date - timedelta(days=45),
                y_top * 0.97,
                "In-Sample (2017–2021) ◄",
                ha="right",
                va="top",
                fontsize=9.0,
                fontweight="bold",
                color="#475569",
            )
            ax_eq.text(
                oos_start_date + timedelta(days=45),
                y_top * 0.97,
                "► Strict OOS Holdout (2022–2026)",
                ha="left",
                va="top",
                fontsize=9.0,
                fontweight="bold",
                color="#DC2626",
            )

        span_days = max((dates[-1] - dates[0]).days, 90)
        ax_eq.set_xlim(dates[0], dates[-1] + timedelta(days=int(span_days * 0.16)))
        ax_eq.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        ax_eq.set_ylabel("Portfolio Value ($)", fontsize=11, fontweight="bold")
        ax_eq.set_title(
            f"V3 10-Year Track Comparison: {report.raw_bar_start} to {report.raw_bar_end} (Strict Anti-Peeking Winner: {winner_id})",
            fontsize=13,
            fontweight="bold",
        )
        ax_eq.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

        # Underwater drawdown styling
        ax_dd.axhline(-14.0, color="#EF4444", linestyle="--", linewidth=1.2, label="-14.0% Max Drawdown Halt")
        ax_dd.axhline(0.0, color="#64748B", linewidth=0.8)
        ax_dd.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax_dd.set_ylabel("Underwater DD (%)", fontsize=11, fontweight="bold")
        ax_dd.set_xlabel("Date", fontsize=11, fontweight="bold")
        ax_dd.legend(loc="lower left", framealpha=0.9, fontsize=9.0)

        ax_dd.xaxis.set_major_locator(mdates.YearLocator())
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(p1, dpi=200, bbox_inches="tight")
        saved_paths.append(p1)
    finally:
        plt.close(fig)

    # --- Plot 2: 2x2 Track Ablation IS vs OOS (16x9) ---
    p2 = output_dir / "v3_02_track_ablation_is_vs_oos.png"
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    try:
        fig.patch.set_facecolor("#FFFFFF")
        ax_cagr, ax_sr = axes[0, 0], axes[0, 1]
        ax_dd, ax_turn = axes[1, 0], axes[1, 1]
        for ax in (ax_cagr, ax_sr, ax_dd, ax_turn):
            _style_ax(ax)

        track_labels = [t.track_id for t in report.tracks] + ["EW-200", "SPY"]
        n_bars = len(track_labels)
        x = np.arange(n_bars, dtype=float)
        width = 0.38

        # Extract metrics for IS vs OOS
        ref_row = report.tracks[0] if report.tracks else None
        is_cagrs = [t.is_metrics.cagr_pct for t in report.tracks] + [
            ref_row.ew_is_cagr_pct if ref_row else 0.0,
            ref_row.spy_is_cagr_pct if ref_row else 0.0,
        ]
        oos_cagrs = [t.oos_metrics.cagr_pct for t in report.tracks] + [
            ref_row.ew_oos_cagr_pct if ref_row else 0.0,
            ref_row.spy_oos_cagr_pct if ref_row else 0.0,
        ]

        is_xs_sr = [t.is_metrics.excess_sharpe_ratio for t in report.tracks] + [
            ref_row.ew_is_excess_sharpe if ref_row else 0.0,
            ref_row.spy_is_excess_sharpe if ref_row else 0.0,
        ]
        oos_xs_sr = [t.oos_metrics.excess_sharpe_ratio for t in report.tracks] + [
            ref_row.ew_oos_excess_sharpe if ref_row else 0.0,
            ref_row.spy_oos_excess_sharpe if ref_row else 0.0,
        ]

        is_dds = [abs(t.is_metrics.max_drawdown_pct) for t in report.tracks] + [
            abs(ref_row.is_metrics.ew_universe_max_drawdown_pct) if ref_row else 0.0,
            abs(ref_row.is_metrics.benchmark_max_drawdown_pct) if ref_row else 0.0,
        ]
        oos_dds = [abs(t.oos_metrics.max_drawdown_pct) for t in report.tracks] + [
            abs(ref_row.oos_metrics.ew_universe_max_drawdown_pct) if ref_row else 0.0,
            abs(ref_row.oos_metrics.benchmark_max_drawdown_pct) if ref_row else 0.0,
        ]

        is_turns = [t.is_metrics.annual_turnover for t in report.tracks] + [0.0, 0.0]
        oos_turns = [t.oos_metrics.annual_turnover for t in report.tracks] + [0.0, 0.0]

        c_is = "#93C5FD"  # lighter blue
        c_oos = "#1D4ED8"  # darker blue

        # (a) CAGR
        ax_cagr.bar(x - width / 2, is_cagrs, width=width, label="IS (2017–2021)", color=c_is, edgecolor="#1E3A8A")
        ax_cagr.bar(x + width / 2, oos_cagrs, width=width, label="OOS (2022–2026)", color=c_oos, edgecolor="#1E3A8A")
        ax_cagr.set_ylabel("CAGR (%)", fontweight="bold")
        ax_cagr.set_title("CAGR (%): IS vs Strict OOS", fontweight="bold", fontsize=11)
        ax_cagr.set_xticks(x)
        ax_cagr.set_xticklabels(track_labels, fontweight="bold")
        ax_cagr.axhline(0.0, color="#64748B", linewidth=0.8)
        ax_cagr.legend(loc="upper left")

        # (b) Excess Sharpe
        ax_sr.bar(x - width / 2, is_xs_sr, width=width, label="IS (2017–2021)", color=c_is, edgecolor="#1E3A8A")
        ax_sr.bar(x + width / 2, oos_xs_sr, width=width, label="OOS (2022–2026)", color=c_oos, edgecolor="#1E3A8A")
        ax_sr.set_ylabel("Excess Sharpe Ratio (vs T-Bills)", fontweight="bold")
        ax_sr.set_title("Excess Sharpe Ratio (vs T-Bills): IS vs Strict OOS", fontweight="bold", fontsize=11)
        ax_sr.set_xticks(x)
        ax_sr.set_xticklabels(track_labels, fontweight="bold")
        ax_sr.axhline(0.0, color="#64748B", linewidth=0.8)
        ax_sr.set_ylim(top=max(max(is_xs_sr + oos_xs_sr, default=0.5) * 1.24, 0.5))

        # Highlight winner on Excess Sharpe
        if winner_id in track_labels:
            w_idx = track_labels.index(winner_id)
            ax_sr.annotate(
                "★ IS Winner",
                xy=(w_idx - width / 2, is_xs_sr[w_idx]),
                xytext=(w_idx - width / 2, is_xs_sr[w_idx] + 0.12),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="#DC2626",
                arrowprops={"arrowstyle": "->", "color": "#DC2626", "lw": 1.2},
            )

        # (c) Max Drawdown
        ax_dd.bar(x - width / 2, is_dds, width=width, label="IS (2017–2021)", color=c_is, edgecolor="#1E3A8A")
        ax_dd.bar(x + width / 2, oos_dds, width=width, label="OOS (2022–2026)", color=c_oos, edgecolor="#1E3A8A")
        ax_dd.set_ylabel("Max Drawdown (% depth)", fontweight="bold")
        ax_dd.set_title("Max Drawdown Depth (%): Lower is Better", fontweight="bold", fontsize=11)
        ax_dd.set_xticks(x)
        ax_dd.set_xticklabels(track_labels, fontweight="bold")
        ax_dd.axhline(14.0, color="#EF4444", linestyle="--", linewidth=1.0, label="14% Halt")

        # (d) Annual Turnover
        ax_turn.bar(x - width / 2, is_turns, width=width, label="IS (2017–2021)", color=c_is, edgecolor="#1E3A8A")
        ax_turn.bar(x + width / 2, oos_turns, width=width, label="OOS (2022–2026)", color=c_oos, edgecolor="#1E3A8A")
        ax_turn.set_ylabel("Annual Turnover (x)", fontweight="bold")
        ax_turn.set_title("Annual Turnover (x): Lower is Better", fontweight="bold", fontsize=11)
        ax_turn.set_xticks(x)
        ax_turn.set_xticklabels(track_labels, fontweight="bold")

        fig.suptitle("V3 Track Ablation Matrix: In-Sample (2017–2021) vs Out-Of-Sample (2022–2026)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(p2, dpi=200, bbox_inches="tight")
        saved_paths.append(p2)
    finally:
        plt.close(fig)

    # --- Plot 3: 10 Calendar Years Comparison (15x6.5) ---
    p3 = output_dir / "v3_03_calendar_year_10yr.png"
    fig, ax = plt.subplots(figsize=(15, 6.5))
    try:
        fig.patch.set_facecolor("#FFFFFF")
        _style_ax(ax)

        winner_row = next((t for t in report.tracks if t.track_id == winner_id), report.tracks[0] if report.tracks else None)
        t0_row = next((t for t in report.tracks if t.track_id == "T0"), report.tracks[0] if report.tracks else None)

        years = sorted(winner_row.annual_returns.keys()) if winner_row else []
        if years:
            x = np.arange(len(years), dtype=float)
            width = 0.20

            strat_vals = [winner_row.annual_returns[y].get("strategy", 0.0) for y in years] if winner_row else []
            t0_vals = [t0_row.annual_returns[y].get("strategy", 0.0) for y in years] if t0_row else []
            ew_vals = [winner_row.annual_returns[y].get("ew", 0.0) for y in years] if winner_row else []
            spy_vals = [winner_row.annual_returns[y].get("spy", 0.0) for y in years] if winner_row else []

            b1 = ax.bar(x - 1.5 * width, strat_vals, width=width, label=f"{winner_id} (IS Winner)", color="#2563EB", alpha=0.9)
            ax.bar(x - 0.5 * width, t0_vals, width=width, label="T0 Baseline", color="#64748B", alpha=0.85)
            ax.bar(x + 0.5 * width, ew_vals, width=width, label="Dynamic EW-200", color="#D97706", alpha=0.85)
            b4 = ax.bar(x + 1.5 * width, spy_vals, width=width, label="SPY Baseline", color="#059669", alpha=0.85)

            # Bar text labels
            for bar_col, vals in [(b1, strat_vals), (b4, spy_vals)]:
                for rect, val in zip(bar_col, vals, strict=True):
                    va = "bottom" if val >= 0 else "top"
                    y_text = (val + 0.8) if val >= 0 else (val - 0.8)
                    ax.text(
                        rect.get_x() + rect.get_width() / 2.0,
                        y_text,
                        f"{val:+.1f}%",
                        ha="center",
                        va=va,
                        fontsize=7.5,
                        fontweight="bold",
                    )

            all_vals = strat_vals + t0_vals + ew_vals + spy_vals
            y_hi = max(max(all_vals, default=20.0) * 1.25, 39.0)
            y_lo = min(min(all_vals, default=-10.0) * 1.18, -21.0)
            ax.set_ylim(y_lo, y_hi)

            # Mark 2022 OOS start divider
            if "2022" in years:
                sep_idx = years.index("2022") - 0.5
                ax.axvline(sep_idx, color="#DC2626", linestyle="--", linewidth=1.5, alpha=0.85)
                ax.text(sep_idx - 0.10, y_hi * 0.96, "IS Window ◄", ha="right", va="top", color="#475569", fontweight="bold", fontsize=9.5)
                ax.text(sep_idx + 0.10, y_hi * 0.96, "► OOS Holdout", ha="left", va="top", color="#DC2626", fontweight="bold", fontsize=9.5)

            ax.axhline(0.0, color="#64748B", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(years, fontsize=10, fontweight="bold")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
            ax.set_ylabel("Calendar-Year Return (%)", fontsize=11, fontweight="bold")
            ax.set_title(
                f"10-Year Calendar-Year Performance: {winner_id} Winner vs T0, EW-200, and SPY ({years[0]}–{years[-1]})",
                fontsize=13,
                fontweight="bold",
                pad=24,
            )
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.005),
                ncol=4,
                framealpha=0.95,
                fontsize=9.5,
            )

        fig.tight_layout()
        fig.savefig(p3, dpi=200, bbox_inches="tight")
        saved_paths.append(p3)
    finally:
        plt.close(fig)

    # --- Plot 4: CPCV & Stress Tests (15x6.2) ---
    p4 = output_dir / "v3_04_cpcv_and_delisting_stress.png"
    fig, (ax_cpcv, ax_stress) = plt.subplots(1, 2, figsize=(15, 6.2))
    try:
        fig.patch.set_facecolor("#FFFFFF")
        _style_ax(ax_cpcv)
        _style_ax(ax_stress)

        # Left: CPCV distribution
        cpcv = report.cpcv_full_summary
        if cpcv and cpcv.fold_metrics:
            xs_sharpes = [m.excess_sharpe_ratio for m in cpcv.fold_metrics]
            sharpes = [m.sharpe_ratio for m in cpcv.fold_metrics]
            y_pos = np.arange(len(sharpes))

            ax_cpcv.barh(y_pos - 0.18, sharpes, height=0.35, label="Sharpe Ratio", color="#3B82F6", alpha=0.85)
            ax_cpcv.barh(y_pos + 0.18, xs_sharpes, height=0.35, label="Excess Sharpe", color="#10B981", alpha=0.85)

            med_sr = cpcv.median_sharpe
            med_xs = cpcv.median_excess_sharpe
            ax_cpcv.axvline(med_sr, color="#1D4ED8", linestyle="--", linewidth=1.2, label=f"Median Sharpe ({med_sr:.2f})")
            ax_cpcv.axvline(med_xs, color="#047857", linestyle="--", linewidth=1.2, label=f"Median Excess ({med_xs:.2f})")
            ax_cpcv.axvline(0.0, color="#64748B", linewidth=0.8)

            ax_cpcv.set_yticks(y_pos)
            ax_cpcv.set_yticklabels([f"Path {i+1}" for i in y_pos], fontsize=8.0)
            ax_cpcv.invert_yaxis()
            ax_cpcv.set_xlabel("Ratio", fontweight="bold")
            ax_cpcv.set_title(
                f"15-Path CPCV Distribution (6 Groups, k=2, 21d Purge, 5d Embargo)\nPositive Paths: {cpcv.pct_positive_paths:.1f}%",
                fontweight="bold",
                fontsize=11,
            )
            ax_cpcv.legend(loc="lower right", fontsize=8.5)
        else:
            ax_cpcv.text(0.5, 0.5, "CPCV Summary Not Generated", ha="center", va="center", color="#94A3B8")

        # Right: Stress Matrix comparison
        stress_specs = []
        winner_row = next((t for t in report.tracks if t.track_id == winner_id), None)
        if winner_row:
            stress_specs.append((
                f"{winner_id} (Broad Top-200)",
                winner_row.full_metrics.cagr_pct,
                winner_row.full_metrics.excess_sharpe_ratio,
                abs(winner_row.full_metrics.max_drawdown_pct),
                winner_row.full_metrics.annual_turnover,
                "#2563EB",
            ))

        if report.delisting_wipeout_stress:
            d_m = report.delisting_wipeout_stress.full_metrics
            stress_specs.append((
                "100% Delisting Wipeout",
                d_m.cagr_pct,
                d_m.excess_sharpe_ratio,
                abs(d_m.max_drawdown_pct),
                d_m.annual_turnover,
                "#EF4444",
            ))

        if report.sp500_only_stress:
            s_m = report.sp500_only_stress.full_metrics
            stress_specs.append((
                "Strict PIT S&P 500",
                s_m.cagr_pct,
                s_m.excess_sharpe_ratio,
                abs(s_m.max_drawdown_pct),
                s_m.annual_turnover,
                "#8B5CF6",
            ))

        if winner_row:
            stress_specs.append((
                "Dynamic EW-200",
                winner_row.ew_full_cagr_pct,
                winner_row.ew_full_excess_sharpe,
                abs(winner_row.full_metrics.ew_universe_max_drawdown_pct),
                0.0,
                "#D97706",
            ))
            stress_specs.append((
                "SPY Baseline",
                winner_row.spy_full_cagr_pct,
                winner_row.spy_full_excess_sharpe,
                abs(winner_row.full_metrics.benchmark_max_drawdown_pct),
                0.0,
                "#059669",
            ))

        if stress_specs:
            y_s = np.arange(len(stress_specs))
            cagrs = [s[1] for s in stress_specs]
            colors = [s[5] for s in stress_specs]
            bars_s = ax_stress.barh(y_s, cagrs, height=0.55, color=colors, alpha=0.88, edgecolor="#1E293B")

            for idx, (b, spec) in enumerate(zip(bars_s, stress_specs, strict=True)):
                _lbl, cagr, xs, dd, _turn = spec[0], spec[1], spec[2], spec[3], spec[4]
                ha = "left" if cagr >= 0 else "right"
                x_pos = (cagr + 0.35) if cagr >= 0 else (cagr - 0.35)
                extra = f"  XS-SR: {xs:.2f} | MaxDD: {dd:.1f}%"
                ax_stress.text(x_pos, idx, f"{cagr:+.1f}%{extra}", va="center", ha=ha, fontsize=8.0, fontweight="bold")

            max_pos_cagr = max(max(cagrs, default=5.0), 1.0)
            min_neg_cagr = min(min(cagrs, default=0.0), 0.0)
            ax_stress.set_xlim(min_neg_cagr * 1.45 if min_neg_cagr < 0 else 0.0, max_pos_cagr * 1.65)
            ax_stress.set_yticks(y_s)
            ax_stress.set_yticklabels([s[0] for s in stress_specs], fontsize=9.0, fontweight="bold")
            ax_stress.invert_yaxis()
            ax_stress.axvline(0.0, color="#64748B", linewidth=0.8)
            ax_stress.set_xlabel("Full 10-Year CAGR (%)", fontweight="bold")
            ax_stress.set_title("Robustness & Stress Matrix (Full 10Y)", fontweight="bold", fontsize=11)

        fig.suptitle(f"Track {winner_id} Validation: CPCV Stability & Severe Stress Tests", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(p4, dpi=200, bbox_inches="tight")
        saved_paths.append(p4)
    finally:
        plt.close(fig)

    return saved_paths


def render_v3_benchmark_markdown(report: V3BenchmarkReport) -> str:
    """Render comprehensive Markdown summary report for V3 benchmarks."""
    lines: list[str] = [
        "# V3 Research: 11-Year Progressive 6-Track Benchmarks & Robustness Audit",
        "",
        "## 1. Dataset & Reconstitution Summary",
        "",
        f"- **Equity Candidate Pool:** {report.pool_total} unique common stocks (S&P 500 historical 2016–2026 + current S&P 500 + S&P 400)",
        f"- **Historical Delisted Names Verified:** {report.delisted_count} tickers",
        f"- **Core Sector/Macro ETFs Loaded:** {report.core_etfs_loaded} / {len(V3_CORE_ETFS)} ({', '.join(V3_CORE_ETFS)})",
        f"- **Active Bar Data Range:** `{report.raw_bar_start}` to `{report.raw_bar_end}` ({report.bars_loaded} total cached parquet assets)",
        f"- **Monthly Reconstitution Schedule:** {report.reconstitution_months} months (Average active universe size: {report.avg_active_universe_size:.1f} symbols)",
        "- **Evaluation Windows:**",
        f"  - **In-Sample (IS):** `{report.is_window[0]}` to `{report.is_window[1]}` (5 Years)",
        f"  - **Strict Out-Of-Sample (OOS):** `{report.oos_window[0]}` to `{report.oos_window[1]}` (4.8 Years)",
        f"  - **Full 10-Year Horizon:** `{report.full_window[0]}` to `{report.full_window[1]}`",
        "",
        "---",
        "",
        "## 2. Progressive Ladder Performance (IS 2017–2021 vs Strict OOS 2022–2026 vs Full 10Y)",
        "",
        "| Track | Name | IS CAGR (%) | IS Excess Sharpe | IS MaxDD (%) | OOS CAGR (%) | OOS Excess Sharpe | OOS MaxDD (%) | Full CAGR (%) | Full Excess Sharpe | Full MaxDD (%) | Turnover (x) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for t in report.tracks:
        star = " ★" if t.track_id == report.is_selected_track_id else ""
        lines.append(
            f"| **{t.track_id}{star}** | {t.name} | "
            f"{t.is_metrics.cagr_pct:+.1f}% | {t.is_metrics.excess_sharpe_ratio:.2f} | {t.is_metrics.max_drawdown_pct:.1f}% | "
            f"{t.oos_metrics.cagr_pct:+.1f}% | {t.oos_metrics.excess_sharpe_ratio:.2f} | {t.oos_metrics.max_drawdown_pct:.1f}% | "
            f"{t.full_metrics.cagr_pct:+.1f}% | {t.full_metrics.excess_sharpe_ratio:.2f} | {t.full_metrics.max_drawdown_pct:.1f}% | "
            f"{t.full_metrics.annual_turnover:.1f}x |"
        )

    # Add Dynamic EW-200 and SPY rows
    if report.tracks:
        ref_t = report.tracks[0]
        lines.append(
            f"| **EW-200** | Dynamic Broad Top-200 | "
            f"{ref_t.ew_is_cagr_pct:+.1f}% | {ref_t.ew_is_excess_sharpe:.2f} | {ref_t.is_metrics.ew_universe_max_drawdown_pct:.1f}% | "
            f"{ref_t.ew_oos_cagr_pct:+.1f}% | {ref_t.ew_oos_excess_sharpe:.2f} | {ref_t.oos_metrics.ew_universe_max_drawdown_pct:.1f}% | "
            f"{ref_t.ew_full_cagr_pct:+.1f}% | {ref_t.ew_full_excess_sharpe:.2f} | {ref_t.full_metrics.ew_universe_max_drawdown_pct:.1f}% | "
            f"0.0x |"
        )
        lines.append(
            f"| **SPY** | S&P 500 Market ETF | "
            f"{ref_t.spy_is_cagr_pct:+.1f}% | {ref_t.spy_is_excess_sharpe:.2f} | {ref_t.is_metrics.benchmark_max_drawdown_pct:.1f}% | "
            f"{ref_t.spy_oos_cagr_pct:+.1f}% | {ref_t.spy_oos_excess_sharpe:.2f} | {ref_t.oos_metrics.benchmark_max_drawdown_pct:.1f}% | "
            f"{ref_t.spy_full_cagr_pct:+.1f}% | {ref_t.spy_full_excess_sharpe:.2f} | {ref_t.full_metrics.benchmark_max_drawdown_pct:.1f}% | "
            f"0.0x |"
        )

    lines.extend([
        "",
        "> [!IMPORTANT]",
        f"> **Strict Anti-Peeking In-Sample Winner:** `{report.is_selected_track_id}`",
        f"> {report.is_selected_reason}",
        "",
        "---",
        "",
        "## 3. Robustness & Stress Audit (T6 Checks)",
        "",
    ])

    if report.cpcv_full_summary:
        c = report.cpcv_full_summary
        lines.extend([
            "### Combinatorial Purged Cross-Validation (CPCV)",
            f"- **Configuration:** {c.n_groups} groups, k={c.k_test_groups} test groups ({c.n_paths} combinatorial paths), {c.purge_days}d purge, {c.embargo_days}d embargo",
            f"- **Median Sharpe:** `{c.median_sharpe:.2f}` (Mean: `{c.mean_sharpe:.2f}`, IQR: `[{c.p25_sharpe:.2f}, {c.p75_sharpe:.2f}]`, Min: `{c.min_sharpe:.2f}`, Max: `{c.max_sharpe:.2f}`)",
            f"- **Median Excess Sharpe (vs T-Bills):** `{c.median_excess_sharpe:.2f}`",
            f"- **Positive Sharpe Paths:** `{c.pct_positive_paths:.1f}%`",
            "",
        ])

    lines.extend([
        "### Stress Matrix",
        "",
        "| Evaluation Variant | Universe Mode | Delisting Haircut | Full CAGR (%) | Full Excess Sharpe | Full MaxDD (%) | Turnover (x) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    winner_row = next((t for t in report.tracks if t.track_id == report.is_selected_track_id), None)
    if winner_row:
        lines.append(
            f"| **{report.is_selected_track_id} (Production Target)** | Broad Top-200 MDDV | 10% | "
            f"{winner_row.full_metrics.cagr_pct:+.1f}% | {winner_row.full_metrics.excess_sharpe_ratio:.2f} | "
            f"{winner_row.full_metrics.max_drawdown_pct:.1f}% | {winner_row.full_metrics.annual_turnover:.1f}x |"
        )

    if report.delisting_wipeout_stress:
        dm = report.delisting_wipeout_stress.full_metrics
        lines.append(
            f"| **100% Delisting Wipeout Stress** | Broad Top-200 MDDV | 100% | "
            f"{dm.cagr_pct:+.1f}% | {dm.excess_sharpe_ratio:.2f} | "
            f"{dm.max_drawdown_pct:.1f}% | {dm.annual_turnover:.1f}x |"
        )

    if report.sp500_only_stress:
        sm = report.sp500_only_stress.full_metrics
        lines.append(
            f"| **Strict PIT S&P 500 Stress** | PIT S&P 500 MDDV | 10% | "
            f"{sm.cagr_pct:+.1f}% | {sm.excess_sharpe_ratio:.2f} | "
            f"{sm.max_drawdown_pct:.1f}% | {sm.annual_turnover:.1f}x |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Calendar-Year Returns (2017–2026)",
        "",
    ])

    if winner_row and winner_row.annual_returns:
        years = sorted(winner_row.annual_returns.keys())
        lines.extend([
            f"| Year | {report.is_selected_track_id} (%) | T0 Baseline (%) | EW-200 (%) | SPY (%) | Regime Window |",
            "| :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        t0_row = next((t for t in report.tracks if t.track_id == "T0"), None)
        for y in years:
            strat_val = winner_row.annual_returns[y].get("strategy", 0.0)
            t0_val = t0_row.annual_returns[y].get("strategy", 0.0) if t0_row else 0.0
            ew_val = winner_row.annual_returns[y].get("ew", 0.0)
            spy_val = winner_row.annual_returns[y].get("spy", 0.0)
            regime = "OOS Holdout" if int(y) >= 2022 else "In-Sample (IS)"
            lines.append(
                f"| **{y}** | {strat_val:+.1f}% | {t0_val:+.1f}% | {ew_val:+.1f}% | {spy_val:+.1f}% | {regime} |"
            )

    if report.plot_paths:
        lines.extend([
            "",
            "---",
            "",
            "## 5. Visual Summary Charts",
            "",
        ])
        for p in report.plot_paths:
            lines.append(f"- `![]({p})`")

    lines.append("")
    return "\n".join(lines)
