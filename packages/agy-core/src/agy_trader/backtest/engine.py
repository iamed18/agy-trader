from __future__ import annotations

import bisect
import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_NY_TZ = ZoneInfo("America/New_York")

from agy_trader.core.config import AppConfig, BacktestConfig, RiskConfig, StrategyConfig
from agy_trader.core.models import (
    AssetRiskMetadata,
    AssetType,
    ControlDials,
    MacroPoint,
    NewsArticle,
    OrderSide,
    RiskVerdict,
    SecFilingEvent,
    UniverseSymbol,
    ensure_utc,
)
from agy_trader.risk.engine import HardRiskEngine, PortfolioRiskState
from agy_trader.strategy.features import (
    MIN_HISTORY_BARS,
    SAFE_HAVEN_SYMBOLS,
    FeatureEngine,
    latest_pit_rf_apr,
)
from agy_trader.strategy.finalist_funnel import apply_stage2_finalist_overlay
from agy_trader.strategy.portfolio import PortfolioConstructor

from .metrics import (
    BacktestMetrics,
    BacktestResult,
    CPCVSummary,
    EquityPoint,
    SimulatedTrade,
    compute_backtest_metrics,
)

# Trailing bar window passed to FeatureEngine (~300 bars -> <0.0005% boundary weight on EMA50)
_FEATURE_LOOKBACK_BARS = 300


@dataclass(frozen=True, slots=True)
class BacktestRunSeries:
    """Backtest result bundled with aligned daily RF, benchmark, and equal-weight series."""

    result: BacktestResult
    daily_rf_rates: list[float]
    benchmark_closes: list[float]
    ew_universe_equities: list[float]


@dataclass
class _PositionLot:
    symbol: str
    qty: float  # Signed (+long, -short)
    avg_entry_price: float
    entry_time: datetime
    stop_loss_price: float | None
    accrued_borrow_usd: float = 0.0
    accrued_sec_fee_usd: float = 0.0


@dataclass
class _PendingExecution:
    verdict: RiskVerdict
    stop_loss_dist_pct: float | None  # Distance from signal-bar close to stop, re-anchored at fill


@dataclass
class _SimState:
    cash: float
    positions: dict[str, _PositionLot] = field(default_factory=dict)
    carried_exits: list[_PendingExecution] = field(default_factory=list)
    tranche_weights: list[dict[str, float]] = field(default_factory=list)
    rebalance_count: int = 0
    bankrupt: bool = False
    trades: list[SimulatedTrade] = field(default_factory=list)
    total_traded_notional: float = 0.0
    total_orders_filled: int = 0
    stop_loss_exits: int = 0
    risk_rejections: int = 0
    total_slippage_usd: float = 0.0
    total_borrow_cost_usd: float = 0.0
    total_sec_fees_usd: float = 0.0
    total_cash_yield_usd: float = 0.0


class BacktestEngine:
    """Event-driven bar-by-bar backtester sharing exact live Feature/Portfolio/Risk code.

    Supports two execution timing modes:
      - ``"same_close"`` (default, matching ``15:40 ET`` live execution):
        1. Overnight short borrow + positive-cash T-bill yield accrual (prev_ts -> ts).
        2. Open-gap stop-loss checks on existing positions at bar t open.
        3. Intraday stop-loss checks on existing positions using bar t high/low.
        4. Mark-to-market at bar t close -> PIT signals -> PortfolioConstructor ->
           HardRiskEngine -> execute approved orders at bar t close + adverse slippage
           (stop-losses for new entries anchor to fill price and start on bar t+1).
        5. Re-compute post-trade EOD mark-to-market equity/exposures -> record EquityPoint.
      - ``"next_open"`` (bar t close signal -> bar t+1 open fill):
        1. Overnight short borrow + positive-cash T-bill yield accrual (prev_ts -> ts).
        2. Open-gap stop-loss checks on existing positions at bar t open.
        3. Execute queued orders from bar t-1 close at bar t open (Pass 1 exits first,
           Pass 2 entries with gap-past-stop cancellation and live open headroom checks).
        4. Intraday stop-loss checks on all open positions using bar t high/low.
        5. Mark-to-market at bar t close -> record EquityPoint.
        6. Generate strictly PIT signals at bar t close -> queue orders for bar t+1 open.
    """

    def __init__(self, app_config: AppConfig, universe: Sequence[UniverseSymbol]) -> None:
        self.config = app_config
        self.universe = list(universe)
        self.universe_meta: dict[str, AssetRiskMetadata] = {
            u.symbol: AssetRiskMetadata(
                symbol=u.symbol,
                asset_type=u.asset_type,
                sector=u.sector,
                shortable=u.shortable_default,
                easy_to_borrow=u.shortable_default,
            )
            for u in self.universe
        }

    def run(
        self,
        bars_by_symbol: Mapping[str, pd.DataFrame],
        *,
        news_by_symbol: Mapping[str, Sequence[NewsArticle]] | None = None,
        macro_points: Sequence[MacroPoint] | None = None,
        earnings_by_symbol: Mapping[str, Sequence[date]] | None = None,
        sec_filings_by_symbol: Mapping[str, Sequence[SecFilingEvent]] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        label: str = "Strategy",
        allow_shorting: bool = True,
        use_news: bool = True,
        exposure_multiplier: float = 1.0,
        close_open_positions_at_end: bool = False,
        execution_timing: Literal["same_close", "next_open"] | None = None,
        n_trials: int = 1,
        eligible_symbols_by_date: Mapping[date, frozenset[str]] | None = None,
        symbol_sectors: Mapping[str, str] | None = None,
        sector_etf_map: Mapping[str, str] | None = None,
    ) -> BacktestResult:
        res, _, _, _ = self._run_with_series(
            bars_by_symbol,
            news_by_symbol=news_by_symbol,
            macro_points=macro_points,
            earnings_by_symbol=earnings_by_symbol,
            sec_filings_by_symbol=sec_filings_by_symbol,
            start_date=start_date,
            end_date=end_date,
            label=label,
            allow_shorting=allow_shorting,
            use_news=use_news,
            exposure_multiplier=exposure_multiplier,
            close_open_positions_at_end=close_open_positions_at_end,
            execution_timing=execution_timing,
            n_trials=n_trials,
            eligible_symbols_by_date=eligible_symbols_by_date,
            symbol_sectors=symbol_sectors,
            sector_etf_map=sector_etf_map,
        )
        return res

    def run_with_series(
        self,
        bars_by_symbol: Mapping[str, pd.DataFrame],
        *,
        news_by_symbol: Mapping[str, Sequence[NewsArticle]] | None = None,
        macro_points: Sequence[MacroPoint] | None = None,
        earnings_by_symbol: Mapping[str, Sequence[date]] | None = None,
        sec_filings_by_symbol: Mapping[str, Sequence[SecFilingEvent]] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        label: str = "Strategy",
        allow_shorting: bool = True,
        use_news: bool = True,
        exposure_multiplier: float = 1.0,
        close_open_positions_at_end: bool = False,
        execution_timing: Literal["same_close", "next_open"] | None = None,
        n_trials: int = 1,
        eligible_symbols_by_date: Mapping[date, frozenset[str]] | None = None,
        symbol_sectors: Mapping[str, str] | None = None,
        sector_etf_map: Mapping[str, str] | None = None,
    ) -> BacktestRunSeries:
        """Run backtest and return result alongside aligned RF, benchmark, and EW series."""
        res, daily_rf_rates, benchmark_closes, ew_universe_equities = self._run_with_series(
            bars_by_symbol,
            news_by_symbol=news_by_symbol,
            macro_points=macro_points,
            earnings_by_symbol=earnings_by_symbol,
            sec_filings_by_symbol=sec_filings_by_symbol,
            start_date=start_date,
            end_date=end_date,
            label=label,
            allow_shorting=allow_shorting,
            use_news=use_news,
            exposure_multiplier=exposure_multiplier,
            close_open_positions_at_end=close_open_positions_at_end,
            execution_timing=execution_timing,
            n_trials=n_trials,
            eligible_symbols_by_date=eligible_symbols_by_date,
            symbol_sectors=symbol_sectors,
            sector_etf_map=sector_etf_map,
        )
        return BacktestRunSeries(
            result=res,
            daily_rf_rates=daily_rf_rates,
            benchmark_closes=benchmark_closes,
            ew_universe_equities=ew_universe_equities,
        )

    def _run_with_series(
        self,
        bars_by_symbol: Mapping[str, pd.DataFrame],
        *,
        news_by_symbol: Mapping[str, Sequence[NewsArticle]] | None = None,
        macro_points: Sequence[MacroPoint] | None = None,
        earnings_by_symbol: Mapping[str, Sequence[date]] | None = None,
        sec_filings_by_symbol: Mapping[str, Sequence[SecFilingEvent]] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        label: str = "Strategy",
        allow_shorting: bool = True,
        use_news: bool = True,
        exposure_multiplier: float = 1.0,
        close_open_positions_at_end: bool = False,
        execution_timing: Literal["same_close", "next_open"] | None = None,
        n_trials: int = 1,
        eligible_symbols_by_date: Mapping[date, frozenset[str]] | None = None,
        symbol_sectors: Mapping[str, str] | None = None,
        sector_etf_map: Mapping[str, str] | None = None,
    ) -> tuple[BacktestResult, list[float], list[float], list[float]]:
        # Validate variant strategy config (guards news-only config + use_news=False -> sum=0)
        strat_payload = self.config.strategy.model_dump()
        strat_payload["w_news_sentiment"] = (
            self.config.strategy.w_news_sentiment if use_news else 0.0
        )
        strat_cfg = StrategyConfig.model_validate(strat_payload)
        risk_cfg = self.config.risk
        bt_cfg = self.config.backtest
        eff_timing: Literal["same_close", "next_open"] = (
            execution_timing if execution_timing is not None else bt_cfg.execution_timing
        )

        feature_engine = FeatureEngine(strat_cfg)
        portfolio_ctor = PortfolioConstructor(strat_cfg, risk_cfg)
        risk_engine = HardRiskEngine(
            risk_cfg, allow_fractional_longs=strat_cfg.allow_fractional_longs
        )

        sorted_bars, ts_arrays, bar_lookup, all_timestamps = self._prepare_bars(bars_by_symbol)
        if not all_timestamps:
            empty_metrics = compute_backtest_metrics(
                label=label,
                equity_curve=[],
                trades=[],
                initial_capital=bt_cfg.initial_capital,
                total_traded_notional=0.0,
                total_orders_filled=0,
                stop_loss_exits=0,
                risk_rejections=0,
                total_slippage_usd=0.0,
                total_borrow_cost_usd=0.0,
                total_sec_fees_usd=0.0,
                total_cash_yield_usd=0.0,
                benchmark_symbol=bt_cfg.benchmark_symbol,
                benchmark_closes=[],
                n_trials=n_trials,
                execution_timing=eff_timing,
            )
            return BacktestResult(metrics=empty_metrics), [], [], []

        # Determine active evaluation window (ensure MIN_HISTORY_BARS warm-up before first trade)
        first_eval_idx = min(MIN_HISTORY_BARS, max(len(all_timestamps) - 1, 0))
        sim_timestamps: list[datetime] = []
        for idx, ts in enumerate(all_timestamps):
            if idx < first_eval_idx:
                continue
            d = ts.date()
            if start_date is not None and d < start_date:
                continue
            if end_date is not None and d > end_date:
                continue
            sim_timestamps.append(ts)

        if not sim_timestamps:
            empty_metrics = compute_backtest_metrics(
                label=label,
                equity_curve=[],
                trades=[],
                initial_capital=bt_cfg.initial_capital,
                total_traded_notional=0.0,
                total_orders_filled=0,
                stop_loss_exits=0,
                risk_rejections=0,
                total_slippage_usd=0.0,
                total_borrow_cost_usd=0.0,
                total_sec_fees_usd=0.0,
                total_cash_yield_usd=0.0,
                benchmark_symbol=bt_cfg.benchmark_symbol,
                benchmark_closes=[],
                n_trials=n_trials,
                execution_timing=eff_timing,
            )
            return BacktestResult(metrics=empty_metrics), [], [], []

        news_sorted, news_ts_sorted = self._prepare_news(news_by_symbol if use_news else None)
        sec_sorted, sec_ts_sorted = self._prepare_sec_filings(sec_filings_by_symbol)

        state = _SimState(cash=float(bt_cfg.initial_capital))
        last_known_close: dict[str, float] = {}
        last_bar_ts: dict[str, datetime] = {}
        pending_queue: list[_PendingExecution] = []
        consecutive_missing_bars: dict[str, int] = defaultdict(int)
        halt_streak_days = 0

        norm_eligible_by_date: dict[date, frozenset[str]] | None = None
        reconst_dates: list[date] = []
        if eligible_symbols_by_date is not None:
            norm_eligible_by_date = {
                d: frozenset(s.strip().upper() for s in syms if s.strip())
                for d, syms in eligible_symbols_by_date.items()
            }
            reconst_dates = sorted(norm_eligible_by_date.keys())

        def _active_universe_for_date(d: date) -> frozenset[str] | None:
            if norm_eligible_by_date is None:
                return None
            if not reconst_dates:
                return frozenset()
            idx = bisect.bisect_right(reconst_dates, d)
            if idx > 0:
                return norm_eligible_by_date[reconst_dates[idx - 1]]
            return frozenset()

        reporting_hwm = state.cash
        risk_hwm = state.cash
        prev_day_equity = state.cash
        prev_ts: datetime | None = None

        equity_curve: list[EquityPoint] = []
        benchmark_closes: list[float] = []
        daily_rf_rates: list[float] = []
        ew_universe_equities: list[float] = []
        ew_equity = float(bt_cfg.initial_capital)
        prev_ew_closes: dict[str, float] = {}

        # Pre-seed last_known_close & last_bar_ts from the last bar before sim_timestamps[0]
        first_sim_ns = int(pd.Timestamp(sim_timestamps[0]).as_unit("ns").value)
        for sym, df in sorted_bars.items():
            pos = int(np.searchsorted(ts_arrays[sym], first_sim_ns, side="left"))
            if pos > 0:
                last_known_close[sym] = float(df["close"].iloc[pos - 1])
                last_bar_ts[sym] = ensure_utc(df["timestamp"].iloc[pos - 1].to_pydatetime())

        for bar_idx, ts in enumerate(sim_timestamps):
            active_universe_syms = _active_universe_for_date(ts.date())

            # 1. Overnight borrow accrual + positive-cash T-bill yield accrual
            period_rf = 0.0
            if prev_ts is not None and not state.bankrupt:
                period_rf = self._accrue_overnight_carry(
                    state=state,
                    prev_ts=prev_ts,
                    ts=ts,
                    last_known_close=last_known_close,
                    macro_points=macro_points,
                    bt_cfg=bt_cfg,
                )
            daily_rf_rates.append(period_rf)

            # 1b. Strictly causal delisting / prolonged-halt forced exit (>= 5 consecutive missing bars)
            if not state.bankrupt and state.positions:
                for sym in list(state.positions.keys()):
                    lot = state.positions[sym]
                    if abs(lot.qty) <= 1e-9:
                        continue
                    if bar_lookup.get(sym, {}).get(ts) is None:
                        consecutive_missing_bars[sym] += 1
                    else:
                        consecutive_missing_bars[sym] = 0
                    if consecutive_missing_bars[sym] >= 5:
                        base_px = last_known_close.get(sym, lot.avg_entry_price)
                        self._force_close_delisted(
                            state=state,
                            sym=sym,
                            lot=lot,
                            base_px=base_px,
                            ts=ts,
                            haircut_pct=risk_cfg.delisting_haircut_pct,
                            bt_cfg=bt_cfg,
                        )
                        consecutive_missing_bars.pop(sym, None)
                        if state.carried_exits:
                            state.carried_exits = [
                                item
                                for item in state.carried_exits
                                if item.verdict.order.symbol != sym
                            ]
                        if pending_queue:
                            pending_queue = [
                                item
                                for item in pending_queue
                                if item.verdict.order.symbol != sym
                            ]

            # 2. Open-gap stop-losses on existing positions at bar t open
            stopped_at_open: set[str] = set()
            if not state.bankrupt:
                stopped_at_open = self._check_stop_losses(
                    state=state,
                    ts=ts,
                    bar_lookup=bar_lookup,
                    mode="open_gap",
                    bt_cfg=bt_cfg,
                )

            if eff_timing == "next_open":
                # 3. Execute queued approved orders from bar t-1 close at bar t open
                open_marks: dict[str, float] = dict(last_known_close)
                for sym, sym_bars in bar_lookup.items():
                    b = sym_bars.get(ts)
                    if b is not None and math.isfinite(b[0]) and b[0] > 0.0:
                        open_marks[sym] = b[0]

                if not state.bankrupt and pending_queue:
                    state.carried_exits = self._execute_order_batch(
                        state=state,
                        items=pending_queue,
                        ts=ts,
                        bar_lookup=bar_lookup,
                        exec_price_idx=0,
                        marks=open_marks,
                        strat_cfg=strat_cfg,
                        risk_cfg=risk_cfg,
                        bt_cfg=bt_cfg,
                        exposure_multiplier=exposure_multiplier,
                        skip_symbols=stopped_at_open,
                        check_stop_gap=True,
                    )
                else:
                    state.carried_exits = []
                pending_queue = []

                # 4. Intra-bar stop-loss checks on all open positions (bar t high/low)
                if not state.bankrupt:
                    self._check_stop_losses(
                        state=state,
                        ts=ts,
                        bar_lookup=bar_lookup,
                        mode="intraday",
                        bt_cfg=bt_cfg,
                    )
            else:
                # same_close Step 3: Intra-bar stops on existing positions BEFORE close signal/fill
                if not state.bankrupt:
                    self._check_stop_losses(
                        state=state,
                        ts=ts,
                        bar_lookup=bar_lookup,
                        mode="intraday",
                        bt_cfg=bt_cfg,
                    )

            # Mark-to-market at bar t close (distinguish live bars vs carried marks!)
            live_close_prices: dict[str, float] = {}
            mtm_prices: dict[str, float] = {}
            for sym, sym_bars in bar_lookup.items():
                bar = sym_bars.get(ts)
                if bar is not None:
                    last_known_close[sym] = bar[3]
                    last_bar_ts[sym] = ts
                    live_close_prices[sym] = bar[3]
                    mtm_prices[sym] = bar[3]
                elif sym in last_known_close:
                    mtm_prices[sym] = last_known_close[sym]

            if strat_cfg.rebalance_schedule == "monthly_first":
                ts_et = ts.astimezone(_NY_TZ)
                prev_et = sim_timestamps[bar_idx - 1].astimezone(_NY_TZ) if bar_idx > 0 else ts_et
                is_rebalance_day = (
                    bar_idx == 0
                    or (ts_et.year, ts_et.month) != (prev_et.year, prev_et.month)
                )
            elif strat_cfg.rebalance_schedule == "monthly_last":
                ts_et = ts.astimezone(_NY_TZ)
                is_rebalance_day = bar_idx == 0 or (
                    bar_idx + 1 < len(sim_timestamps)
                    and (
                        sim_timestamps[bar_idx + 1].astimezone(_NY_TZ).year,
                        sim_timestamps[bar_idx + 1].astimezone(_NY_TZ).month,
                    )
                    != (ts_et.year, ts_et.month)
                )
            else:
                is_rebalance_day = (bar_idx % strat_cfg.rebalance_every_n_days) == 0
            is_last_bar = ts == sim_timestamps[-1]

            if eff_timing == "same_close":
                # Step 4 (same_close): pre-trade MTM -> PIT signals -> Risk -> fill at bar t close
                pre_mtm = sum(
                    lot.qty * mtm_prices.get(sym, lot.avg_entry_price)
                    for sym, lot in state.positions.items()
                )
                pre_equity = 0.0 if state.bankrupt else (state.cash + pre_mtm)
                if pre_equity <= 0.0:
                    self._mark_bankrupt(state)
                    pre_equity = 0.0

                if not state.bankrupt and pre_equity > 0.0:
                    pre_drawdown = (
                        (pre_equity - max(risk_hwm, pre_equity))
                        / max(risk_hwm, pre_equity)
                        if max(risk_hwm, pre_equity) > 1e-9
                        else 0.0
                    )
                    pre_daily_pnl = (
                        (pre_equity - prev_day_equity) / prev_day_equity
                        if prev_day_equity > 1e-9
                        else 0.0
                    )
                    batch_to_exec = list(state.carried_exits)
                    carried_symbols = {item.verdict.order.symbol for item in state.carried_exits}
                    if is_rebalance_day and not (close_open_positions_at_end and is_last_bar):
                        new_items = self._generate_signals_and_verdicts(
                            ts=ts,
                            sorted_bars=sorted_bars,
                            ts_arrays=ts_arrays,
                            last_bar_ts=last_bar_ts,
                            news_sorted=news_sorted,
                            news_ts_sorted=news_ts_sorted,
                            sec_sorted=sec_sorted,
                            sec_ts_sorted=sec_ts_sorted,
                            macro_points=macro_points,
                            earnings_by_symbol=earnings_by_symbol,
                            feature_engine=feature_engine,
                            portfolio_ctor=portfolio_ctor,
                            risk_engine=risk_engine,
                            strat_cfg=strat_cfg,
                            risk_cfg=risk_cfg,
                            bt_cfg=bt_cfg,
                            state=state,
                            mtm_prices=mtm_prices,
                            equity=pre_equity,
                            daily_pnl_pct=pre_daily_pnl,
                            drawdown_pct=pre_drawdown,
                            allow_shorting=allow_shorting,
                            use_news=use_news,
                            exposure_multiplier=exposure_multiplier,
                            label=label,
                            carried_symbols=carried_symbols,
                            active_universe_syms=active_universe_syms,
                            symbol_sectors=symbol_sectors,
                            sector_etf_map=sector_etf_map,
                        )
                        batch_to_exec.extend(new_items)

                    if batch_to_exec:
                        state.carried_exits = self._execute_order_batch(
                            state=state,
                            items=batch_to_exec,
                            ts=ts,
                            bar_lookup=bar_lookup,
                            exec_price_idx=3,
                            marks=mtm_prices,
                            strat_cfg=strat_cfg,
                            risk_cfg=risk_cfg,
                            bt_cfg=bt_cfg,
                            exposure_multiplier=exposure_multiplier,
                            skip_symbols=set(),
                            check_stop_gap=False,
                        )
                    else:
                        state.carried_exits = []

            # Optional end-of-window mark-to-close liquidation for exact trade reconciliation
            if close_open_positions_at_end and is_last_bar and state.positions and not state.bankrupt:
                self._liquidate_positions_at_end(
                    state=state,
                    ts=ts,
                    mtm_prices=mtm_prices,
                    bt_cfg=bt_cfg,
                )

            # Final post-trade EOD mark-to-market
            mtm_value = 0.0
            gross_notional = 0.0
            net_notional = 0.0
            for sym, lot in state.positions.items():
                px = mtm_prices.get(sym, lot.avg_entry_price)
                val = lot.qty * px
                mtm_value += val
                gross_notional += abs(val)
                net_notional += val

            equity = 0.0 if state.bankrupt else (state.cash + mtm_value)
            if equity <= 0.0:
                self._mark_bankrupt(state)
                equity = 0.0
                mtm_value = 0.0
                gross_notional = 0.0
                net_notional = 0.0

            reporting_hwm = max(reporting_hwm, equity)
            reporting_drawdown_pct = (
                (equity - reporting_hwm) / reporting_hwm if reporting_hwm > 1e-9 else 0.0
            )
            risk_hwm = max(risk_hwm, equity)
            risk_drawdown_pct = (
                (equity - risk_hwm) / risk_hwm if risk_hwm > 1e-9 else 0.0
            )
            daily_pnl_pct = (
                (equity - prev_day_equity) / prev_day_equity if prev_day_equity > 1e-9 else 0.0
            )
            gross_exp = gross_notional / equity if equity > 1e-9 else 0.0
            net_exp = net_notional / equity if equity > 1e-9 else 0.0

            equity_curve.append(
                EquityPoint(
                    timestamp=ts,
                    equity=round(equity, 2),
                    cash=round(state.cash, 2),
                    gross_exposure=round(gross_exp, 4),
                    net_exposure=round(net_exp, 4),
                    daily_pnl_pct=round(daily_pnl_pct, 6),
                    drawdown_pct=round(min(0.0, reporting_drawdown_pct), 6),
                )
            )
            if risk_drawdown_pct <= -risk_cfg.max_drawdown_halt_pct:
                halt_streak_days += 1
            else:
                halt_streak_days = 0
            if (
                risk_cfg.max_drawdown_cooldown_days is not None
                and not state.bankrupt
                and equity > 0.0
                and halt_streak_days >= risk_cfg.max_drawdown_cooldown_days
            ):
                risk_hwm = equity
                halt_streak_days = 0
                risk_drawdown_pct = 0.0

            # Only append genuine live benchmark bars (never carry forward stale SPY closes!)
            if bt_cfg.benchmark_symbol in live_close_prices:
                benchmark_closes.append(live_close_prices[bt_cfg.benchmark_symbol])

            # Track equal-weight universe index using MTM prices so missing bars
            # hold 0% return on day t and realize exact multi-day return on day t+1
            all_valid_mtm = {
                sym: px
                for sym, px in mtm_prices.items()
                if (not self.universe_meta or sym in self.universe_meta)
                and math.isfinite(px)
                and px > 0.0
            }
            if bar_idx == 0:
                ew_universe_equities.append(ew_equity)
            else:
                sym_rets = [
                    close_px / prev_ew_closes[sym] - 1.0
                    for sym, close_px in all_valid_mtm.items()
                    if sym in prev_ew_closes and prev_ew_closes[sym] > 0.0
                ]
                if sym_rets:
                    ew_equity = max(0.0, ew_equity * (1.0 + float(np.mean(sym_rets))))
                ew_universe_equities.append(ew_equity)
            if active_universe_syms is not None:
                prev_ew_closes = {
                    sym: px
                    for sym, px in all_valid_mtm.items()
                    if sym in active_universe_syms
                }
            else:
                prev_ew_closes = all_valid_mtm

            if eff_timing == "next_open":
                # 6. Compute strictly point-in-time signals & queue approved orders for bar t+1 open
                pending_queue = list(state.carried_exits)
                carried_symbols = {item.verdict.order.symbol for item in state.carried_exits}
                if is_rebalance_day and not is_last_bar and not state.bankrupt and equity > 0.0:
                    new_items = self._generate_signals_and_verdicts(
                        ts=ts,
                        sorted_bars=sorted_bars,
                        ts_arrays=ts_arrays,
                        last_bar_ts=last_bar_ts,
                        news_sorted=news_sorted,
                        news_ts_sorted=news_ts_sorted,
                        sec_sorted=sec_sorted,
                        sec_ts_sorted=sec_ts_sorted,
                        macro_points=macro_points,
                        earnings_by_symbol=earnings_by_symbol,
                        feature_engine=feature_engine,
                        portfolio_ctor=portfolio_ctor,
                        risk_engine=risk_engine,
                        strat_cfg=strat_cfg,
                        risk_cfg=risk_cfg,
                        bt_cfg=bt_cfg,
                        state=state,
                        mtm_prices=mtm_prices,
                        equity=equity,
                        daily_pnl_pct=daily_pnl_pct,
                        drawdown_pct=risk_drawdown_pct,
                        allow_shorting=allow_shorting,
                        use_news=use_news,
                        exposure_multiplier=exposure_multiplier,
                        label=label,
                        carried_symbols=carried_symbols,
                        active_universe_syms=active_universe_syms,
                        symbol_sectors=symbol_sectors,
                        sector_etf_map=sector_etf_map,
                    )
                    pending_queue.extend(new_items)

            prev_day_equity = equity
            prev_ts = ts

        metrics = compute_backtest_metrics(
            label=label,
            equity_curve=equity_curve,
            trades=state.trades,
            initial_capital=bt_cfg.initial_capital,
            total_traded_notional=state.total_traded_notional,
            total_orders_filled=state.total_orders_filled,
            stop_loss_exits=state.stop_loss_exits,
            risk_rejections=state.risk_rejections,
            total_slippage_usd=state.total_slippage_usd,
            total_borrow_cost_usd=state.total_borrow_cost_usd,
            total_sec_fees_usd=state.total_sec_fees_usd,
            total_cash_yield_usd=state.total_cash_yield_usd,
            benchmark_symbol=bt_cfg.benchmark_symbol,
            benchmark_closes=benchmark_closes,
            daily_rf_rates=daily_rf_rates,
            ew_universe_equities=ew_universe_equities,
            n_trials=n_trials,
            execution_timing=eff_timing,
        )
        result = BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            trades=state.trades,
            config_snapshot={
                "label": label,
                "allow_shorting": allow_shorting,
                "use_news": use_news,
                "exposure_multiplier": exposure_multiplier,
                "execution_timing": eff_timing,
                "n_trials": n_trials,
            },
        )
        return result, daily_rf_rates, benchmark_closes, ew_universe_equities

    @staticmethod
    def _prepare_bars(
        bars_by_symbol: Mapping[str, pd.DataFrame],
    ) -> tuple[
        dict[str, pd.DataFrame],
        dict[str, np.ndarray],
        dict[str, dict[datetime, tuple[float, float, float, float]]],
        list[datetime],
    ]:
        sorted_bars: dict[str, pd.DataFrame] = {}
        ts_arrays: dict[str, np.ndarray] = {}
        bar_lookup: dict[str, dict[datetime, tuple[float, float, float, float]]] = {}
        all_timestamps_set: set[datetime] = set()

        for raw_sym, raw_df in bars_by_symbol.items():
            sym = raw_sym.strip().upper()
            if not sym or raw_df.empty:
                continue
            df = raw_df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
            sorted_bars[sym] = df
            ts_ns = df["timestamp"].dt.as_unit("ns").astype("int64").to_numpy()
            ts_arrays[sym] = ts_ns
            sym_map: dict[datetime, tuple[float, float, float, float]] = {}
            for row in df.itertuples(index=False):
                ts_dt = ensure_utc(row.timestamp.to_pydatetime())
                sym_map[ts_dt] = (
                    float(row.open),
                    float(row.high),
                    float(row.low),
                    float(row.close),
                )
                all_timestamps_set.add(ts_dt)
            bar_lookup[sym] = sym_map

        return sorted_bars, ts_arrays, bar_lookup, sorted(all_timestamps_set)

    @staticmethod
    def _prepare_news(
        news_by_symbol: Mapping[str, Sequence[NewsArticle]] | None,
    ) -> tuple[dict[str, list[NewsArticle]], dict[str, list[datetime]]]:
        news_sorted: dict[str, list[NewsArticle]] = {}
        news_ts_sorted: dict[str, list[datetime]] = {}
        if not news_by_symbol:
            return news_sorted, news_ts_sorted
        for raw_sym, arts in news_by_symbol.items():
            sym = raw_sym.strip().upper()
            scored = sorted(
                (a for a in arts if a.sentiment_score is not None),
                key=lambda a: a.published_at,
            )
            if scored:
                news_sorted[sym] = scored
                news_ts_sorted[sym] = [a.published_at for a in scored]
        return news_sorted, news_ts_sorted

    @staticmethod
    def _prepare_sec_filings(
        sec_filings_by_symbol: Mapping[str, Sequence[SecFilingEvent]] | None,
    ) -> tuple[dict[str, list[SecFilingEvent]], dict[str, list[datetime]]]:
        sec_sorted: dict[str, list[SecFilingEvent]] = {}
        sec_ts_sorted: dict[str, list[datetime]] = {}
        if not sec_filings_by_symbol:
            return sec_sorted, sec_ts_sorted
        for raw_sym, events in sec_filings_by_symbol.items():
            sym = raw_sym.strip().upper()
            ordered = sorted(events, key=lambda e: e.accepted_at)
            if ordered:
                sec_sorted[sym] = ordered
                sec_ts_sorted[sym] = [e.accepted_at for e in ordered]
        return sec_sorted, sec_ts_sorted

    @staticmethod
    def _mark_bankrupt(state: _SimState) -> None:
        state.bankrupt = True
        state.cash = 0.0
        state.positions.clear()
        state.carried_exits.clear()

    @staticmethod
    def _accrue_overnight_carry(
        *,
        state: _SimState,
        prev_ts: datetime,
        ts: datetime,
        last_known_close: Mapping[str, float],
        macro_points: Sequence[MacroPoint] | None,
        bt_cfg: BacktestConfig,
    ) -> float:
        elapsed_days = max(0.0, (ts - prev_ts).total_seconds() / 86400.0)
        day_frac = elapsed_days / 365.0
        if bt_cfg.use_fred_tbill_yield:
            pit_rf_apr = latest_pit_rf_apr(
                prev_ts, macro_points, fallback_apr=bt_cfg.risk_free_rate_apr
            )
            pit_cash_apr = latest_pit_rf_apr(
                prev_ts, macro_points, fallback_apr=bt_cfg.cash_yield_apr
            )
        else:
            pit_rf_apr = bt_cfg.risk_free_rate_apr
            pit_cash_apr = bt_cfg.cash_yield_apr

        if day_frac > 0.0:
            if bt_cfg.short_borrow_apr > 0.0:
                for sym, lot in state.positions.items():
                    if lot.qty < -1e-9:
                        mark_px = last_known_close.get(sym, lot.avg_entry_price)
                        borrow_fee = abs(lot.qty) * mark_px * bt_cfg.short_borrow_apr * day_frac
                        state.cash -= borrow_fee
                        lot.accrued_borrow_usd += borrow_fee
                        state.total_borrow_cost_usd += borrow_fee
            if pit_cash_apr > 0.0:
                short_notional = sum(
                    abs(lot.qty) * last_known_close.get(sym, lot.avg_entry_price)
                    for sym, lot in state.positions.items()
                    if lot.qty < -1e-9
                )
                unencumbered_cash = max(0.0, state.cash - short_notional)
                if unencumbered_cash > 0.0:
                    cash_yield = unencumbered_cash * pit_cash_apr * day_frac
                    state.cash += cash_yield
                    state.total_cash_yield_usd += cash_yield

        return pit_rf_apr * day_frac

    def _force_close_delisted(
        self,
        *,
        state: _SimState,
        sym: str,
        lot: _PositionLot,
        base_px: float,
        ts: datetime,
        haircut_pct: float,
        bt_cfg: BacktestConfig,
    ) -> None:
        haircut = max(0.0, min(1.0, float(haircut_pct)))
        if lot.qty > 1e-9:
            fill_px = round(base_px * (1.0 - haircut), 6)
            qty = lot.qty
            slip_usd = qty * max(0.0, base_px - fill_px)
            sec_fee = qty * fill_px * (bt_cfg.sec_sell_fee_bps / 10_000.0)
            state.cash += qty * fill_px - sec_fee
            gross_pnl = (fill_px - lot.avg_entry_price) * qty
            net_pnl = gross_pnl - sec_fee - lot.accrued_sec_fee_usd - lot.accrued_borrow_usd
            side = "long"
        else:
            fill_px = round(base_px * (1.0 + haircut), 6)
            qty = abs(lot.qty)
            slip_usd = qty * max(0.0, fill_px - base_px)
            sec_fee = 0.0
            state.cash -= qty * fill_px
            gross_pnl = (lot.avg_entry_price - fill_px) * qty
            net_pnl = gross_pnl - lot.accrued_sec_fee_usd - lot.accrued_borrow_usd
            side = "short"

        state.total_traded_notional += qty * fill_px
        state.total_orders_filled += 1
        state.total_slippage_usd += slip_usd
        state.total_sec_fees_usd += sec_fee
        state.trades.append(
            SimulatedTrade(
                symbol=sym,
                side=side,
                entry_time=lot.entry_time,
                exit_time=ts,
                qty=qty,
                entry_price=round(lot.avg_entry_price, 6),
                exit_price=round(fill_px, 6),
                gross_pnl=round(gross_pnl, 4),
                net_pnl=round(net_pnl, 4),
                holding_days=max(0.0, (ts - lot.entry_time).total_seconds() / 86400.0),
                exit_reason="delisted_forced_exit",
            )
        )
        del state.positions[sym]
        for tw in state.tranche_weights:
            tw.pop(sym, None)

    def _close_lot_full(
        self,
        *,
        state: _SimState,
        sym: str,
        lot: _PositionLot,
        trigger_px: float,
        ts: datetime,
        exit_reason: str,
        bt_cfg: BacktestConfig,
        count_as_stop: bool = False,
    ) -> None:
        slip_rate = self._slippage_rate(sym)
        if lot.qty > 1e-9:
            fill_px = trigger_px * (1.0 - slip_rate)
            qty = lot.qty
            slip_usd = qty * (trigger_px - fill_px)
            sec_fee = qty * fill_px * (bt_cfg.sec_sell_fee_bps / 10_000.0)
            state.cash += qty * fill_px - sec_fee
            gross_pnl = (fill_px - lot.avg_entry_price) * qty
            net_pnl = gross_pnl - sec_fee - lot.accrued_sec_fee_usd - lot.accrued_borrow_usd
            side = "long"
        else:
            fill_px = trigger_px * (1.0 + slip_rate)
            qty = abs(lot.qty)
            slip_usd = qty * (fill_px - trigger_px)
            sec_fee = 0.0
            state.cash -= qty * fill_px
            gross_pnl = (lot.avg_entry_price - fill_px) * qty
            net_pnl = gross_pnl - lot.accrued_sec_fee_usd - lot.accrued_borrow_usd
            side = "short"

        state.total_traded_notional += qty * fill_px
        state.total_orders_filled += 1
        if count_as_stop:
            state.stop_loss_exits += 1
        state.total_slippage_usd += slip_usd
        state.total_sec_fees_usd += sec_fee
        state.trades.append(
            SimulatedTrade(
                symbol=sym,
                side=side,
                entry_time=lot.entry_time,
                exit_time=ts,
                qty=qty,
                entry_price=round(lot.avg_entry_price, 6),
                exit_price=round(fill_px, 6),
                gross_pnl=round(gross_pnl, 4),
                net_pnl=round(net_pnl, 4),
                holding_days=max(0.0, (ts - lot.entry_time).total_seconds() / 86400.0),
                exit_reason=exit_reason,
            )
        )
        del state.positions[sym]

    def _check_stop_losses(
        self,
        *,
        state: _SimState,
        ts: datetime,
        bar_lookup: Mapping[str, Mapping[datetime, tuple[float, float, float, float]]],
        mode: Literal["open_gap", "intraday"],
        bt_cfg: BacktestConfig,
    ) -> set[str]:
        stopped_syms: set[str] = set()
        for sym in list(state.positions.keys()):
            lot = state.positions[sym]
            if lot.stop_loss_price is None:
                continue
            bar = bar_lookup.get(sym, {}).get(ts)
            if bar is None:
                continue
            bar_open, bar_high, bar_low, _ = bar
            trigger_px: float | None = None
            if mode == "open_gap":
                if (lot.qty > 1e-9 and bar_open <= lot.stop_loss_price) or (
                    lot.qty < -1e-9 and bar_open >= lot.stop_loss_price
                ):
                    trigger_px = bar_open
                reason = "stop_loss_open_gap"
            else:
                if lot.qty > 1e-9 and bar_low <= lot.stop_loss_price:
                    trigger_px = min(bar_open, lot.stop_loss_price)
                elif lot.qty < -1e-9 and bar_high >= lot.stop_loss_price:
                    trigger_px = max(bar_open, lot.stop_loss_price)
                reason = "stop_loss_intraday"

            if trigger_px is not None:
                self._close_lot_full(
                    state=state,
                    sym=sym,
                    lot=lot,
                    trigger_px=trigger_px,
                    ts=ts,
                    exit_reason=reason,
                    bt_cfg=bt_cfg,
                    count_as_stop=True,
                )
                stopped_syms.add(sym)
        return stopped_syms

    def _liquidate_positions_at_end(
        self,
        *,
        state: _SimState,
        ts: datetime,
        mtm_prices: dict[str, float],
        bt_cfg: BacktestConfig,
    ) -> None:
        for sym in list(state.positions.keys()):
            lot = state.positions[sym]
            px = mtm_prices.get(sym, lot.avg_entry_price)
            self._close_lot_full(
                state=state,
                sym=sym,
                lot=lot,
                trigger_px=px,
                ts=ts,
                exit_reason="end_of_window",
                bt_cfg=bt_cfg,
                count_as_stop=False,
            )

    def _execute_order_batch(
        self,
        *,
        state: _SimState,
        items: Sequence[_PendingExecution],
        ts: datetime,
        bar_lookup: Mapping[str, Mapping[datetime, tuple[float, float, float, float]]],
        exec_price_idx: int,
        marks: Mapping[str, float],
        strat_cfg: StrategyConfig,
        risk_cfg: RiskConfig,
        bt_cfg: BacktestConfig,
        exposure_multiplier: float,
        skip_symbols: set[str],
        check_stop_gap: bool,
    ) -> list[_PendingExecution]:
        ordered = sorted(
            items,
            key=lambda item: 0 if item.verdict.order.is_risk_reducing else 1,
        )
        carried_exits: list[_PendingExecution] = []

        for item in ordered:
            verdict = item.verdict
            order = verdict.order
            sym = order.symbol
            if sym in skip_symbols:
                continue
            bar = bar_lookup.get(sym, {}).get(ts)
            if bar is None or not math.isfinite(bar[exec_price_idx]) or bar[exec_price_idx] <= 0.0:
                if order.is_risk_reducing and sym in state.positions:
                    carried_exits.append(item)
                continue
            exec_px = bar[exec_price_idx]

            qty = verdict.adjusted_qty
            if qty <= 1e-9:
                continue

            existing = state.positions.get(sym)
            slip_rate = self._slippage_rate(sym)
            is_reducing = False

            if order.is_risk_reducing:
                if existing is None:
                    continue
                if existing.qty > 0.0 and order.side != OrderSide.sell:
                    continue
                if existing.qty < 0.0 and order.side != OrderSide.buy:
                    continue
                qty = min(qty, abs(existing.qty))
                if qty <= 1e-9:
                    continue
                is_reducing = True
            else:
                is_short_entry = order.side == OrderSide.sell and (
                    existing is None or existing.qty <= 1e-9
                )
                if check_stop_gap and verdict.stop_loss_price is not None:
                    if not is_short_entry and exec_px <= verdict.stop_loss_price:
                        state.risk_rejections += 1
                        continue
                    if is_short_entry and exec_px >= verdict.stop_loss_price:
                        state.risk_rejections += 1
                        continue

                live_equity = state.cash + sum(
                    lot.qty * marks.get(s, lot.avg_entry_price)
                    for s, lot in state.positions.items()
                )
                if live_equity <= 0.0:
                    state.risk_rejections += 1
                    continue

                est_fill_px = (
                    exec_px * (1.0 + slip_rate)
                    if order.side == OrderSide.buy
                    else exec_px * (1.0 - slip_rate)
                )
                cur_gross = sum(
                    abs(lot.qty * marks.get(s, lot.avg_entry_price))
                    for s, lot in state.positions.items()
                )
                cur_net = sum(
                    lot.qty * marks.get(s, lot.avg_entry_price)
                    for s, lot in state.positions.items()
                )
                eff_gross_cap = (
                    risk_cfg.max_gross_exposure * min(1.0, max(0.0, exposure_multiplier))
                ) * live_equity
                gross_headroom = max(0.0, eff_gross_cap - cur_gross)
                if is_short_entry:
                    net_headroom = max(0.0, cur_net - risk_cfg.min_net_exposure * live_equity)
                    pos_cap = risk_cfg.max_short_position_pct * live_equity
                else:
                    net_headroom = max(0.0, risk_cfg.max_net_exposure * live_equity - cur_net)
                    pos_cap = risk_cfg.max_long_position_pct * live_equity

                cur_pos_val = abs(existing.qty) * exec_px if existing is not None else 0.0
                pos_headroom = max(0.0, pos_cap - cur_pos_val)
                sym_meta = self.universe_meta.get(sym)
                if sym_meta is not None:
                    cur_sector_gross = sum(
                        abs(lot.qty * marks.get(s, lot.avg_entry_price))
                        for s, lot in state.positions.items()
                        if (m := self.universe_meta.get(s)) is not None
                        and m.sector == sym_meta.sector
                    )
                    sector_headroom = max(
                        0.0, risk_cfg.max_sector_gross_pct * live_equity - cur_sector_gross
                    )
                else:
                    sector_headroom = float("inf")
                cash_headroom = (
                    max(0.0, state.cash) if order.side == OrderSide.buy else float("inf")
                )
                sizing_px = max(exec_px, est_fill_px) if is_short_entry else est_fill_px

                max_notional = min(
                    qty * sizing_px,
                    gross_headroom,
                    net_headroom,
                    pos_headroom,
                    sector_headroom,
                    cash_headroom,
                )
                if max_notional <= 1e-6:
                    state.risk_rejections += 1
                    continue

                if is_short_entry or not strat_cfg.allow_fractional_longs:
                    qty = float(math.floor(max_notional / sizing_px + 1e-12))
                    if qty < 1.0:
                        state.risk_rejections += 1
                        continue
                else:
                    qty = math.floor((max_notional / sizing_px) * 1e6) / 1e6
                    if qty <= 1e-6:
                        state.risk_rejections += 1
                        continue

            if order.side == OrderSide.buy:
                fill_px = exec_px * (1.0 + slip_rate)
                slip_usd = qty * (fill_px - exec_px)
                sec_fee = 0.0
                state.cash -= qty * fill_px
            else:
                fill_px = exec_px * (1.0 - slip_rate)
                slip_usd = qty * (exec_px - fill_px)
                sec_fee = qty * fill_px * (bt_cfg.sec_sell_fee_bps / 10_000.0)
                state.cash += qty * fill_px - sec_fee

            state.total_traded_notional += qty * fill_px
            state.total_orders_filled += 1
            state.total_slippage_usd += slip_usd
            state.total_sec_fees_usd += sec_fee

            delta_signed = qty if order.side == OrderSide.buy else -qty
            if existing is None:
                new_stop = self._reanchor_stop(
                    fill_px=fill_px,
                    is_short=(delta_signed < 0.0),
                    dist_pct=item.stop_loss_dist_pct,
                    fallback_stop=verdict.stop_loss_price,
                )
                state.positions[sym] = _PositionLot(
                    symbol=sym,
                    qty=delta_signed,
                    avg_entry_price=fill_px,
                    entry_time=ts,
                    stop_loss_price=new_stop,
                    accrued_borrow_usd=0.0,
                    accrued_sec_fee_usd=sec_fee,
                )
            else:
                same_sign = (existing.qty > 0.0 and delta_signed > 0.0) or (
                    existing.qty < 0.0 and delta_signed < 0.0
                )
                if same_sign:
                    old_abs = abs(existing.qty)
                    add_abs = abs(delta_signed)
                    new_abs = old_abs + add_abs
                    existing.avg_entry_price = (
                        existing.avg_entry_price * old_abs + fill_px * add_abs
                    ) / new_abs
                    existing.qty += delta_signed
                    existing.accrued_sec_fee_usd += sec_fee
                    new_stop = self._reanchor_stop(
                        fill_px=fill_px,
                        is_short=(existing.qty < 0.0),
                        dist_pct=item.stop_loss_dist_pct,
                        fallback_stop=verdict.stop_loss_price,
                    )
                    if new_stop is not None:
                        if existing.stop_loss_price is None:
                            existing.stop_loss_price = new_stop
                        elif existing.qty > 0.0:
                            existing.stop_loss_price = max(existing.stop_loss_price, new_stop)
                        else:
                            existing.stop_loss_price = min(existing.stop_loss_price, new_stop)
                else:
                    old_abs = abs(existing.qty)
                    close_abs = min(old_abs, abs(delta_signed))
                    frac_closed = close_abs / max(old_abs, 1e-12)
                    pro_rata_borrow = existing.accrued_borrow_usd * frac_closed
                    pro_rata_entry_sec = existing.accrued_sec_fee_usd * frac_closed
                    existing.accrued_borrow_usd -= pro_rata_borrow
                    existing.accrued_sec_fee_usd -= pro_rata_entry_sec

                    if existing.qty > 0.0:
                        gross_pnl = (fill_px - existing.avg_entry_price) * close_abs
                        trade_side = "long"
                    else:
                        gross_pnl = (existing.avg_entry_price - fill_px) * close_abs
                        trade_side = "short"
                    net_pnl = gross_pnl - sec_fee - pro_rata_entry_sec - pro_rata_borrow

                    state.trades.append(
                        SimulatedTrade(
                            symbol=sym,
                            side=trade_side,
                            entry_time=existing.entry_time,
                            exit_time=ts,
                            qty=close_abs,
                            entry_price=round(existing.avg_entry_price, 6),
                            exit_price=round(fill_px, 6),
                            gross_pnl=round(gross_pnl, 4),
                            net_pnl=round(net_pnl, 4),
                            holding_days=max(
                                0.0, (ts - existing.entry_time).total_seconds() / 86400.0
                            ),
                            exit_reason=order.reason
                            or ("rebalance_exit" if is_reducing else "rebalance_reduce"),
                        )
                    )
                    new_q = existing.qty + delta_signed
                    if (
                        abs(new_q) <= 1e-9
                        or (existing.qty > 0.0 and new_q < 0.0)
                        or (existing.qty < 0.0 and new_q > 0.0)
                    ):
                        del state.positions[sym]
                    else:
                        existing.qty = new_q

        return carried_exits

    def _generate_signals_and_verdicts(
        self,
        *,
        ts: datetime,
        sorted_bars: Mapping[str, pd.DataFrame],
        ts_arrays: Mapping[str, np.ndarray],
        last_bar_ts: Mapping[str, datetime],
        news_sorted: Mapping[str, list[NewsArticle]],
        news_ts_sorted: Mapping[str, list[datetime]],
        sec_sorted: Mapping[str, list[SecFilingEvent]],
        sec_ts_sorted: Mapping[str, list[datetime]],
        macro_points: Sequence[MacroPoint] | None,
        earnings_by_symbol: Mapping[str, Sequence[date]] | None,
        feature_engine: FeatureEngine,
        portfolio_ctor: PortfolioConstructor,
        risk_engine: HardRiskEngine,
        strat_cfg: StrategyConfig,
        risk_cfg: RiskConfig,
        bt_cfg: BacktestConfig,
        state: _SimState,
        mtm_prices: dict[str, float],
        equity: float,
        daily_pnl_pct: float,
        drawdown_pct: float,
        allow_shorting: bool,
        use_news: bool,
        exposure_multiplier: float,
        label: str,
        carried_symbols: set[str],
        active_universe_syms: frozenset[str] | None = None,
        symbol_sectors: Mapping[str, str] | None = None,
        sector_etf_map: Mapping[str, str] | None = None,
    ) -> list[_PendingExecution]:
        ts_ns = int(pd.Timestamp(ts).as_unit("ns").value)
        pit_bars: dict[str, pd.DataFrame] = {}
        fresh_bar_ages: list[float] = []

        safe_havens_in_meta: set[str] = (
            {s for s in SAFE_HAVEN_SYMBOLS if s in self.universe_meta}
            if strat_cfg.safe_haven_rotation_fraction > 0.0
            else set()
        )
        needed_bar_syms: set[str] | None = None
        if active_universe_syms is not None:
            needed_bar_syms = (
                set(active_universe_syms)
                | {bt_cfg.benchmark_symbol.strip().upper()}
                | {
                    s.strip().upper()
                    for s in (sector_etf_map.values() if sector_etf_map else ())
                    if s.strip()
                }
                | set(state.positions.keys())
                | safe_havens_in_meta
            )

        feature_lb = max(_FEATURE_LOOKBACK_BARS, strat_cfg.m2_long_lookback + 48)
        for sym, df in sorted_bars.items():
            if needed_bar_syms is not None and sym not in needed_bar_syms:
                continue
            sym_last_ts = last_bar_ts.get(sym)
            if sym_last_ts is None:
                continue
            age_min = (ts - sym_last_ts).total_seconds() / 60.0
            if age_min > risk_cfg.max_data_staleness_minutes:
                continue
            end_idx = int(np.searchsorted(ts_arrays[sym], ts_ns, side="right"))
            if end_idx >= MIN_HISTORY_BARS:
                start_idx = max(0, end_idx - feature_lb)
                pit_bars[sym] = df.iloc[start_idx:end_idx]
                fresh_bar_ages.append(max(0.0, age_min))

        latest_age_min = (
            max(fresh_bar_ages)
            if fresh_bar_ages
            else float(risk_cfg.max_data_staleness_minutes + 1)
        )

        pit_news: dict[str, list[NewsArticle]] = {}
        if use_news and news_sorted:
            win_lo = ts - timedelta(days=30)
            for sym, arts in news_sorted.items():
                if sym not in pit_bars:
                    continue
                ts_list = news_ts_sorted[sym]
                hi_i = bisect.bisect_right(ts_list, ts)
                lo_i = bisect.bisect_left(ts_list, win_lo, 0, hi_i)
                if hi_i > lo_i:
                    pit_news[sym] = arts[lo_i:hi_i]

        pit_sec: dict[str, list[SecFilingEvent]] = {}
        if sec_sorted:
            sec_lookback = max(
                strat_cfg.insider_lookback_days,
                risk_cfg.restatement_blackout_days * 2 + 14,
            )
            sec_win_lo = ts - timedelta(days=sec_lookback)
            for sym, evs in sec_sorted.items():
                if sym not in pit_bars:
                    continue
                ts_list = sec_ts_sorted[sym]
                hi_i = bisect.bisect_right(ts_list, ts)
                lo_i = bisect.bisect_left(ts_list, sec_win_lo, 0, hi_i)
                if hi_i > lo_i:
                    pit_sec[sym] = evs[lo_i:hi_i]

        eff_universe_syms: frozenset[str] | None
        if active_universe_syms is not None:
            eff_universe_syms = (
                active_universe_syms | frozenset(safe_havens_in_meta)
                if safe_havens_in_meta
                else active_universe_syms
            )
        elif self.universe_meta:
            eff_universe_syms = frozenset(self.universe_meta.keys())
        else:
            eff_universe_syms = None

        eff_sectors: Mapping[str, str] | None = (
            symbol_sectors
            if symbol_sectors is not None
            else ({s: m.sector for s, m in self.universe_meta.items()} if self.universe_meta else None)
        )

        signals = feature_engine.build_signals(
            as_of=ts,
            bars_by_symbol=pit_bars,
            news_by_symbol=pit_news if use_news else None,
            macro_points=macro_points,
            benchmark_symbol=bt_cfg.benchmark_symbol,
            fallback_rf_apr=bt_cfg.risk_free_rate_apr,
            sec_filings_by_symbol=pit_sec if pit_sec else None,
            restatement_blackout_days=risk_cfg.restatement_blackout_days,
            universe_symbols=eff_universe_syms,
            symbol_sectors=eff_sectors,
            sector_etf_map=sector_etf_map,
        )

        if strat_cfg.w_pead > 0.0 and signals:
            signals = apply_stage2_finalist_overlay(
                signals=signals,
                bars_by_symbol=pit_bars,
                held_symbols=set(state.positions.keys()),
                as_of=ts,
                finalist_pool_size=strat_cfg.finalist_pool_size,
                w_pead=strat_cfg.w_pead,
                w_trend=strat_cfg.w_trend,
                w_pullback=strat_cfg.w_pullback,
                w_news_sentiment=strat_cfg.w_news_sentiment if use_news else 0.0,
                w_insider=strat_cfg.w_insider,
                w_fip=strat_cfg.w_fip,
                earnings_by_symbol=earnings_by_symbol,
            )

        dials = ControlDials(
            pause_new_entries=False,
            exposure_multiplier=exposure_multiplier,
            allow_shorting=allow_shorting,
            updated_by="backtest",
            reason=label,
            updated_at=ts,
        )
        signed_pos = {sym: lot.qty for sym, lot in state.positions.items()}

        if strat_cfg.rebalance_tranches > 1:
            n_tranches = strat_cfg.rebalance_tranches
            cur_weights: dict[str, float] = {}
            for sym, qty in signed_pos.items():
                px = mtm_prices.get(sym, 0.0)
                if px > 0.0 and math.isfinite(px) and equity > 0.0:
                    cur_weights[sym] = (qty * px) / equity

            if active_universe_syms is not None:
                allowed_tranche_syms = active_universe_syms | frozenset(safe_havens_in_meta)
                for tw in state.tranche_weights:
                    for sym in list(tw.keys()):
                        if sym not in allowed_tranche_syms:
                            tw.pop(sym, None)

            if not signals:
                # Do not initialize empty tranches or advance rebalance_count when signals are empty
                blended_weights = portfolio_ctor.compute_target_weights(
                    signals=(),
                    dials=dials,
                    as_of=ts,
                    universe_meta=self.universe_meta,
                    current_weights=cur_weights,
                )
                if active_universe_syms is not None:
                    allowed_tranche_syms = active_universe_syms | frozenset(safe_havens_in_meta)
                    for sym in list(signed_pos.keys()):
                        if sym not in allowed_tranche_syms:
                            blended_weights[sym] = 0.0
            else:
                in_halt = (
                    not math.isfinite(daily_pnl_pct)
                    or daily_pnl_pct <= -risk_cfg.daily_loss_halt_pct
                    or not math.isfinite(drawdown_pct)
                    or drawdown_pct <= -risk_cfg.max_drawdown_halt_pct
                )
                if in_halt:
                    # During kill-switch halt: do NOT insert phantom Pass-2 entries or advance tranche pointer;
                    # only purge existing held symbols whose fresh target exited to 0.0 so Pass-1 exits run.
                    fresh_weights = portfolio_ctor.compute_target_weights(
                        signals=signals,
                        dials=dials,
                        as_of=ts,
                        universe_meta=self.universe_meta,
                        current_weights=cur_weights,
                    )
                    for tw in state.tranche_weights:
                        for sym in list(tw.keys()):
                            if abs(fresh_weights.get(sym, 0.0)) <= 1e-9:
                                tw.pop(sym, None)
                    if state.tranche_weights:
                        all_tranche_syms: set[str] = set()
                        for tw in state.tranche_weights:
                            all_tranche_syms.update(tw.keys())
                        blended_weights = {}
                        for sym in all_tranche_syms:
                            avg_w = sum(tw.get(sym, 0.0) for tw in state.tranche_weights) / float(
                                len(state.tranche_weights)
                            )
                            blended_weights[sym] = 0.0 if abs(avg_w) < 1e-6 else avg_w
                    else:
                        blended_weights = {
                            sym: (w if abs(fresh_weights.get(sym, 0.0)) > 1e-9 else 0.0)
                            for sym, w in cur_weights.items()
                        }
                else:
                    if state.rebalance_count == 0 or len(state.tranche_weights) != n_tranches:
                        fresh_weights = portfolio_ctor.compute_target_weights(
                            signals=signals,
                            dials=dials,
                            as_of=ts,
                            universe_meta=self.universe_meta,
                            current_weights=cur_weights,
                        )
                        nz_fresh = {
                            k: v for k, v in fresh_weights.items() if abs(v) > 1e-9
                        }
                        state.tranche_weights = [dict(nz_fresh) for _ in range(n_tranches)]
                    else:
                        tranche_idx = state.rebalance_count % n_tranches
                        prev_tranche_w = state.tranche_weights[tranche_idx]
                        fresh_weights = portfolio_ctor.compute_target_weights(
                            signals=signals,
                            dials=dials,
                            as_of=ts,
                            universe_meta=self.universe_meta,
                            current_weights=prev_tranche_w,
                        )
                        state.tranche_weights[tranche_idx] = {
                            k: v for k, v in fresh_weights.items() if abs(v) > 1e-9
                        }
                    state.rebalance_count += 1

                    all_tranche_syms = set()
                    for tw in state.tranche_weights:
                        all_tranche_syms.update(tw.keys())
                    blended_weights = {}
                    for sym in all_tranche_syms:
                        avg_w = sum(tw.get(sym, 0.0) for tw in state.tranche_weights) / float(
                            n_tranches
                        )
                        blended_weights[sym] = 0.0 if abs(avg_w) < 1e-6 else avg_w

                if active_universe_syms is not None:
                    allowed_tranche_syms = active_universe_syms | frozenset(safe_havens_in_meta)
                    for sym in list(signed_pos.keys()):
                        if sym not in allowed_tranche_syms:
                            blended_weights[sym] = 0.0

                blended_weights = portfolio_ctor.apply_portfolio_overlays(
                    blended_weights,
                    signals=signals,
                    dials=dials,
                    universe_meta=self.universe_meta,
                    current_weights=cur_weights,
                    as_of=ts,
                )

            proposed = portfolio_ctor.build_proposed_orders(
                signals=signals,
                current_positions=signed_pos,
                prices=mtm_prices,
                equity=equity,
                dials=dials,
                as_of=ts,
                universe_meta=self.universe_meta,
                target_weights_override=blended_weights,
            )
        else:
            proposed = portfolio_ctor.build_proposed_orders(
                signals=signals,
                current_positions=signed_pos,
                prices=mtm_prices,
                equity=equity,
                dials=dials,
                as_of=ts,
                universe_meta=self.universe_meta,
            )
        proposed = [o for o in proposed if o.symbol not in carried_symbols]

        risk_state = PortfolioRiskState(
            as_of=ts,
            equity=equity,
            daily_pnl_pct=daily_pnl_pct,
            drawdown_pct=min(0.0, drawdown_pct),
            positions=signed_pos,
            prices=mtm_prices,
            latest_bar_age_minutes=latest_age_min,
        )
        verdicts = risk_engine.evaluate_batch(
            orders=proposed,
            state=risk_state,
            dials=dials,
            universe_meta=self.universe_meta,
            signals_by_symbol={s.symbol: s for s in signals},
            earnings_by_symbol=earnings_by_symbol,
        )

        approved_items: list[_PendingExecution] = []
        for v in verdicts:
            if v.approved and v.adjusted_qty > 0.0:
                dist_pct: float | None = None
                if v.stop_loss_price is not None and v.order.reference_price > 0.0:
                    dist_pct = abs(v.stop_loss_price - v.order.reference_price) / v.order.reference_price
                approved_items.append(_PendingExecution(verdict=v, stop_loss_dist_pct=dist_pct))
            else:
                state.risk_rejections += 1
        return approved_items

    def run_ablation(
        self,
        bars_by_symbol: Mapping[str, pd.DataFrame],
        *,
        news_by_symbol: Mapping[str, Sequence[NewsArticle]] | None = None,
        macro_points: Sequence[MacroPoint] | None = None,
        earnings_by_symbol: Mapping[str, Sequence[date]] | None = None,
        sec_filings_by_symbol: Mapping[str, Sequence[SecFilingEvent]] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        execution_timing: Literal["same_close", "next_open"] | None = None,
        n_trials: int = 1,
        eligible_symbols_by_date: Mapping[date, frozenset[str]] | None = None,
        symbol_sectors: Mapping[str, str] | None = None,
        sector_etf_map: Mapping[str, str] | None = None,
    ) -> list[BacktestResult]:
        """Run the 4-way A/B ablation matrix on identical market data."""
        variants = [
            ("Price+Vol Only (Long-Only)", False, False),
            ("Price+Vol + News (Long-Only)", False, True),
            ("Price+Vol Only (Long/Short)", True, False),
            ("Price+Vol + News (Long/Short)", True, True),
        ]
        return [
            self.run(
                bars_by_symbol,
                news_by_symbol=news_by_symbol,
                macro_points=macro_points,
                earnings_by_symbol=earnings_by_symbol,
                sec_filings_by_symbol=sec_filings_by_symbol,
                start_date=start_date,
                end_date=end_date,
                label=label,
                allow_shorting=allow_short,
                use_news=use_news,
                execution_timing=execution_timing,
                n_trials=n_trials,
                eligible_symbols_by_date=eligible_symbols_by_date,
                symbol_sectors=symbol_sectors,
                sector_etf_map=sector_etf_map,
            )
            for label, allow_short, use_news in variants
        ]

    @staticmethod
    def _extract_sorted_dates(bars_by_symbol: Mapping[str, pd.DataFrame]) -> list[date]:
        return sorted(
            {
                ensure_utc(pd.Timestamp(ts).tz_convert("UTC").to_pydatetime()).date()
                if pd.Timestamp(ts).tzinfo is not None
                else ensure_utc(pd.Timestamp(ts, tz="UTC").to_pydatetime()).date()
                for df in bars_by_symbol.values()
                if not df.empty
                for ts in df["timestamp"]
            }
        )

    def run_walk_forward(
        self,
        bars_by_symbol: Mapping[str, pd.DataFrame],
        *,
        news_by_symbol: Mapping[str, Sequence[NewsArticle]] | None = None,
        macro_points: Sequence[MacroPoint] | None = None,
        earnings_by_symbol: Mapping[str, Sequence[date]] | None = None,
        sec_filings_by_symbol: Mapping[str, Sequence[SecFilingEvent]] | None = None,
        n_folds: int = 3,
        allow_shorting: bool = True,
        use_news: bool = True,
        execution_timing: Literal["same_close", "next_open"] | None = None,
        eligible_symbols_by_date: Mapping[date, frozenset[str]] | None = None,
        symbol_sectors: Mapping[str, str] | None = None,
        sector_etf_map: Mapping[str, str] | None = None,
    ) -> list[BacktestMetrics]:
        """Evaluate fixed-parameter strategy across chronological out-of-sample calendar folds."""
        all_dates = self._extract_sorted_dates(bars_by_symbol)
        if len(all_dates) <= MIN_HISTORY_BARS + max(n_folds, 1) * 20:
            return []

        eval_dates = all_dates[MIN_HISTORY_BARS:]
        fold_size = len(eval_dates) // max(n_folds, 1)
        fold_metrics: list[BacktestMetrics] = []
        for f_idx in range(n_folds):
            lo = f_idx * fold_size
            hi = len(eval_dates) - 1 if f_idx == n_folds - 1 else (f_idx + 1) * fold_size - 1
            res = self.run(
                bars_by_symbol,
                news_by_symbol=news_by_symbol,
                macro_points=macro_points,
                earnings_by_symbol=earnings_by_symbol,
                sec_filings_by_symbol=sec_filings_by_symbol,
                start_date=eval_dates[lo],
                end_date=eval_dates[hi],
                label=f"Fold {f_idx + 1}",
                allow_shorting=allow_shorting,
                use_news=use_news,
                execution_timing=execution_timing,
                eligible_symbols_by_date=eligible_symbols_by_date,
                symbol_sectors=symbol_sectors,
                sector_etf_map=sector_etf_map,
            )
            fold_metrics.append(res.metrics)
        return fold_metrics

    def run_cpcv(
        self,
        bars_by_symbol: Mapping[str, pd.DataFrame],
        *,
        news_by_symbol: Mapping[str, Sequence[NewsArticle]] | None = None,
        macro_points: Sequence[MacroPoint] | None = None,
        earnings_by_symbol: Mapping[str, Sequence[date]] | None = None,
        sec_filings_by_symbol: Mapping[str, Sequence[SecFilingEvent]] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        n_groups: int = 6,
        k_test_groups: int = 2,
        purge_days: int = 21,
        embargo_days: int = 5,
        allow_shorting: bool = True,
        use_news: bool = True,
        execution_timing: Literal["same_close", "next_open"] | None = None,
        eligible_symbols_by_date: Mapping[date, frozenset[str]] | None = None,
        symbol_sectors: Mapping[str, str] | None = None,
        sector_etf_map: Mapping[str, str] | None = None,
    ) -> CPCVSummary:
        """Combinatorial Purged Cross-Validation (CPCV) across C(n_groups, k_test_groups) paths.

        Partitions evaluable dates into ``n_groups`` contiguous blocks and evaluates every
        combination of ``k_test_groups`` blocks with ``purge_days`` / ``embargo_days`` boundary
        exclusions where test blocks border non-test blocks. Each contiguous test span runs
        independently from ``initial_capital`` with full prior warm-up history.
        """
        if n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if k_test_groups < 1 or k_test_groups >= n_groups:
            raise ValueError("k_test_groups must satisfy 1 <= k_test_groups < n_groups")
        if purge_days < 0 or embargo_days < 0:
            raise ValueError("purge_days and embargo_days must be >= 0")

        all_dates = self._extract_sorted_dates(bars_by_symbol)
        eval_dates = all_dates[MIN_HISTORY_BARS:] if len(all_dates) > MIN_HISTORY_BARS else []
        eff_start = start_date
        if eligible_symbols_by_date:
            earliest_sched = min(eligible_symbols_by_date.keys())
            eff_start = (
                max(eff_start, earliest_sched) if eff_start is not None else earliest_sched
            )
        if eff_start is not None or end_date is not None:
            eval_dates = [
                d
                for d in eval_dates
                if (eff_start is None or d >= eff_start) and (end_date is None or d <= end_date)
            ]
        if len(eval_dates) < n_groups * 2:
            return CPCVSummary(
                n_groups=n_groups,
                k_test_groups=k_test_groups,
                purge_days=purge_days,
                embargo_days=embargo_days,
                n_paths=0,
                median_sharpe=0.0,
                mean_sharpe=0.0,
                p25_sharpe=0.0,
                p75_sharpe=0.0,
                min_sharpe=0.0,
                max_sharpe=0.0,
                pct_positive_paths=0.0,
                median_excess_sharpe=0.0,
                fold_metrics=[],
            )

        # Partition eval_dates into n_groups contiguous (start_idx, end_idx) index ranges
        group_indices = np.array_split(np.arange(len(eval_dates)), n_groups)
        group_bounds = [(int(arr[0]), int(arr[-1])) for arr in group_indices if arr.size > 0]

        span_cache: dict[
            tuple[int, int], tuple[BacktestResult, list[float], list[float], list[float]]
        ] = {}
        eff_timing: Literal["same_close", "next_open"] = (
            execution_timing
            if execution_timing is not None
            else self.config.backtest.execution_timing
        )

        fold_metrics: list[BacktestMetrics] = []
        for path_idx, combo in enumerate(itertools.combinations(range(len(group_bounds)), k_test_groups), start=1):
            # Merge adjacent test groups into contiguous spans before applying boundary purge/embargo
            merged_spans: list[tuple[int, int]] = []
            for grp_i in combo:
                lo, hi = group_bounds[grp_i]
                if merged_spans and lo == merged_spans[-1][1] + 1:
                    merged_spans[-1] = (merged_spans[-1][0], hi)
                else:
                    merged_spans.append((lo, hi))

            span_runs: list[tuple[BacktestResult, list[float], list[float], list[float]]] = []
            path_valid = True
            for raw_lo, raw_hi in merged_spans:
                eff_lo = raw_lo + purge_days if raw_lo > 0 else raw_lo
                eff_hi = raw_hi - embargo_days if raw_hi < len(eval_dates) - 1 else raw_hi
                if eff_hi - eff_lo < 1:
                    path_valid = False
                    break
                cache_key = (eff_lo, eff_hi)
                if cache_key not in span_cache:
                    span_cache[cache_key] = self._run_with_series(
                        bars_by_symbol,
                        news_by_symbol=news_by_symbol,
                        macro_points=macro_points,
                        earnings_by_symbol=earnings_by_symbol,
                        sec_filings_by_symbol=sec_filings_by_symbol,
                        start_date=eval_dates[eff_lo],
                        end_date=eval_dates[eff_hi],
                        label=f"Span {eff_lo}..{eff_hi}",
                        allow_shorting=allow_shorting,
                        use_news=use_news,
                        execution_timing=eff_timing,
                        eligible_symbols_by_date=eligible_symbols_by_date,
                        symbol_sectors=symbol_sectors,
                        sector_etf_map=sector_etf_map,
                    )
                span_runs.append(span_cache[cache_key])

            if not path_valid or not span_runs:
                continue

            combo_tag = "+".join(f"G{g + 1}" for g in combo)
            path_label = f"Path {path_idx} ({combo_tag})"
            path_metrics = self._combine_cpcv_spans(
                span_runs=span_runs,
                label=path_label,
                initial_capital=self.config.backtest.initial_capital,
                benchmark_symbol=self.config.backtest.benchmark_symbol,
                execution_timing=eff_timing,
            )
            fold_metrics.append(path_metrics)

        if not fold_metrics:
            return CPCVSummary(
                n_groups=n_groups,
                k_test_groups=k_test_groups,
                purge_days=purge_days,
                embargo_days=embargo_days,
                n_paths=0,
                median_sharpe=0.0,
                mean_sharpe=0.0,
                p25_sharpe=0.0,
                p75_sharpe=0.0,
                min_sharpe=0.0,
                max_sharpe=0.0,
                pct_positive_paths=0.0,
                median_excess_sharpe=0.0,
                fold_metrics=[],
            )

        sharpes = [m.sharpe_ratio for m in fold_metrics]
        ex_sharpes = [m.excess_sharpe_ratio for m in fold_metrics]
        return CPCVSummary(
            n_groups=n_groups,
            k_test_groups=k_test_groups,
            purge_days=purge_days,
            embargo_days=embargo_days,
            n_paths=len(fold_metrics),
            median_sharpe=round(float(np.median(sharpes)), 3),
            mean_sharpe=round(float(np.mean(sharpes)), 3),
            p25_sharpe=round(float(np.percentile(sharpes, 25)), 3),
            p75_sharpe=round(float(np.percentile(sharpes, 75)), 3),
            min_sharpe=round(float(np.min(sharpes)), 3),
            max_sharpe=round(float(np.max(sharpes)), 3),
            pct_positive_paths=round(sum(1 for s in sharpes if s > 0.0) / len(sharpes) * 100.0, 2),
            median_excess_sharpe=round(float(np.median(ex_sharpes)), 3),
            fold_metrics=fold_metrics,
        )

    @staticmethod
    def _combine_cpcv_spans(
        *,
        span_runs: Sequence[tuple[BacktestResult, list[float], list[float], list[float]]],
        label: str,
        initial_capital: float,
        benchmark_symbol: str,
        execution_timing: str,
    ) -> BacktestMetrics:
        if len(span_runs) == 1:
            return span_runs[0][0].metrics.model_copy(update={"label": label})

        stitched_curve: list[EquityPoint] = []
        stitched_rf: list[float] = [0.0]
        stitched_bm: list[float] = []
        stitched_ew: list[float] = []

        first_res = span_runs[0][0]
        cur_eq = (
            float(first_res.equity_curve[0].equity)
            if first_res.equity_curve
            else float(initial_capital)
        )
        cur_bm = 100.0
        cur_ew = (
            float(span_runs[0][3][0])
            if span_runs[0][3]
            else float(initial_capital)
        )

        if first_res.equity_curve:
            p0 = first_res.equity_curve[0]
            stitched_curve.append(
                p0.model_copy(update={"equity": round(cur_eq, 2)})
            )
        stitched_bm.append(cur_bm)
        stitched_ew.append(cur_ew)

        all_trades: list[SimulatedTrade] = []
        total_traded_notional = 0.0
        total_orders_filled = 0
        stop_loss_exits = 0
        risk_rejections = 0
        total_slippage_usd = 0.0
        total_borrow_cost_usd = 0.0
        total_sec_fees_usd = 0.0
        total_cash_yield_usd = 0.0

        for span_idx, (res, rf_rates, bm_closes, ew_equities) in enumerate(span_runs):
            all_trades.extend(res.trades)
            m = res.metrics
            total_orders_filled += m.total_orders_filled
            stop_loss_exits += m.stop_loss_exits
            risk_rejections += m.risk_rejections
            total_slippage_usd += m.total_slippage_usd
            total_borrow_cost_usd += m.total_borrow_cost_usd
            total_sec_fees_usd += m.total_sec_fees_usd
            total_cash_yield_usd += m.total_cash_yield_usd
            if m.trading_days > 0 and res.equity_curve:
                avg_eq = float(np.mean([pt.equity for pt in res.equity_curve]))
                years = max(m.trading_days / 252.0, 1.0 / 252.0)
                total_traded_notional += m.annual_turnover * 2.0 * avg_eq * years

            if span_idx > 0 and initial_capital > 0.0:
                if res.equity_curve:
                    ret0 = (res.equity_curve[0].equity / initial_capital) - 1.0
                    cur_eq = max(0.0, cur_eq * (1.0 + ret0))
                if ew_equities:
                    ew_ret0 = (ew_equities[0] / initial_capital) - 1.0
                    cur_ew = max(0.0, cur_ew * (1.0 + ew_ret0))

            for k in range(1, len(res.equity_curve)):
                prev_e = res.equity_curve[k - 1].equity
                cur_e = res.equity_curve[k].equity
                ret = (cur_e / prev_e) - 1.0 if prev_e > 0.0 else 0.0
                cur_eq = max(0.0, cur_eq * (1.0 + ret))
                pt = res.equity_curve[k]
                stitched_curve.append(pt.model_copy(update={"equity": round(cur_eq, 2)}))
                stitched_rf.append(rf_rates[k] if k < len(rf_rates) else 0.0)

            for k in range(1, len(bm_closes)):
                if bm_closes[k - 1] > 0.0:
                    cur_bm = max(0.0, cur_bm * (bm_closes[k] / bm_closes[k - 1]))
                    stitched_bm.append(cur_bm)

            for k in range(1, len(ew_equities)):
                if ew_equities[k - 1] > 0.0:
                    cur_ew = max(0.0, cur_ew * (ew_equities[k] / ew_equities[k - 1]))
                    stitched_ew.append(cur_ew)

        return compute_backtest_metrics(
            label=label,
            equity_curve=stitched_curve,
            trades=all_trades,
            initial_capital=initial_capital,
            total_traded_notional=total_traded_notional,
            total_orders_filled=total_orders_filled,
            stop_loss_exits=stop_loss_exits,
            risk_rejections=risk_rejections,
            total_slippage_usd=total_slippage_usd,
            total_borrow_cost_usd=total_borrow_cost_usd,
            total_sec_fees_usd=total_sec_fees_usd,
            total_cash_yield_usd=total_cash_yield_usd,
            benchmark_symbol=benchmark_symbol,
            benchmark_closes=stitched_bm,
            daily_rf_rates=stitched_rf,
            ew_universe_equities=stitched_ew,
            n_trials=1,
            execution_timing=execution_timing,
        )

    def _slippage_rate(self, symbol: str) -> float:
        meta = self.universe_meta.get(symbol)
        if meta is not None and meta.asset_type == AssetType.etf:
            return self.config.backtest.etf_slippage_bps / 10_000.0
        return self.config.backtest.equity_slippage_bps / 10_000.0

    @staticmethod
    def _reanchor_stop(
        fill_px: float,
        is_short: bool,
        dist_pct: float | None,
        fallback_stop: float | None,
    ) -> float | None:
        if dist_pct is not None and math.isfinite(dist_pct) and dist_pct > 0.0:
            if is_short:
                return round(fill_px * (1.0 + dist_pct), 4)
            return round(fill_px * max(0.05, 1.0 - dist_pct), 4)
        if fallback_stop is not None:
            if is_short and fallback_stop > fill_px:
                return round(fallback_stop, 4)
            if not is_short and 0.0 < fallback_stop < fill_px:
                return round(fallback_stop, 4)
        return None
