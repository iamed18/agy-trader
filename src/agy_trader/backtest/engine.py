from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from agy_trader.core.config import AppConfig, StrategyConfig
from agy_trader.core.models import (
    AssetRiskMetadata,
    AssetType,
    ControlDials,
    MacroPoint,
    NewsArticle,
    OrderSide,
    RiskVerdict,
    UniverseSymbol,
    ensure_utc,
)
from agy_trader.risk.engine import HardRiskEngine, PortfolioRiskState
from agy_trader.strategy.features import MIN_HISTORY_BARS, FeatureEngine
from agy_trader.strategy.portfolio import PortfolioConstructor

from .metrics import (
    BacktestMetrics,
    BacktestResult,
    EquityPoint,
    SimulatedTrade,
    compute_backtest_metrics,
)

# Trailing bar window passed to FeatureEngine (~300 bars -> <0.0005% boundary weight on EMA50)
_FEATURE_LOOKBACK_BARS = 300


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


class BacktestEngine:
    """Event-driven bar-by-bar backtester sharing exact live Feature/Portfolio/Risk code.

    Execution timeline for every trading day t:
      1. Overnight short borrow accrual from bar t-1 to bar t (exact elapsed days / 365).
      2. Open-gap stop-loss checks on existing positions (fill at bar t open + adverse slippage).
      3. Queued rebalance orders from bar t-1 close execute at bar t open:
         - Pass 1 risk-reducing exits execute first (carried forward if a bar is missing!).
         - Pass 2 entries are cancelled if bar t open already gapped past their stop-loss,
           and are re-checked against live t-open cash, gross, net, and position caps so
           overnight gaps or deferred exits can never breach cash or hard risk limits.
      4. Intra-bar stop-loss checks on all open positions (fill at stop or worse open + slippage).
      5. Mark-to-market at bar t close -> compute equity, daily PnL, drawdown -> record curve.
         Only live symbols with a fresh bar at t enter pit_bars & benchmark_closes.
      6. Generate strictly point-in-time signals at bar t close -> PortfolioConstructor ->
         HardRiskEngine (with real bar-age staleness) -> queue approved orders for bar t+1 open.
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
        start_date: date | None = None,
        end_date: date | None = None,
        label: str = "Strategy",
        allow_shorting: bool = True,
        use_news: bool = True,
        exposure_multiplier: float = 1.0,
        close_open_positions_at_end: bool = False,
    ) -> BacktestResult:
        # Validate variant strategy config (guards news-only config + use_news=False -> sum=0)
        strat_payload = self.config.strategy.model_dump()
        strat_payload["w_news_sentiment"] = (
            self.config.strategy.w_news_sentiment if use_news else 0.0
        )
        strat_cfg = StrategyConfig.model_validate(strat_payload)
        risk_cfg = self.config.risk
        bt_cfg = self.config.backtest

        feature_engine = FeatureEngine(strat_cfg)
        portfolio_ctor = PortfolioConstructor(strat_cfg, risk_cfg)
        risk_engine = HardRiskEngine(
            risk_cfg, allow_fractional_longs=strat_cfg.allow_fractional_longs
        )

        # Normalize keys, pre-sort bars once, and index ns timestamps safely
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

        all_timestamps = sorted(all_timestamps_set)
        if not all_timestamps:
            return BacktestResult(
                metrics=compute_backtest_metrics(
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
                    benchmark_symbol=bt_cfg.benchmark_symbol,
                    benchmark_closes=[],
                )
            )

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
            return BacktestResult(
                metrics=compute_backtest_metrics(
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
                    benchmark_symbol=bt_cfg.benchmark_symbol,
                    benchmark_closes=[],
                )
            )

        # Pre-sort scored news per symbol by published_at for O(log N) point-in-time windowing
        news_sorted: dict[str, list[NewsArticle]] = {}
        news_ts_sorted: dict[str, list[datetime]] = {}
        if use_news and news_by_symbol:
            for raw_sym, arts in news_by_symbol.items():
                sym = raw_sym.strip().upper()
                scored = sorted(
                    (a for a in arts if a.sentiment_score is not None),
                    key=lambda a: a.published_at,
                )
                if scored:
                    news_sorted[sym] = scored
                    news_ts_sorted[sym] = [a.published_at for a in scored]

        # Portfolio state
        cash = float(bt_cfg.initial_capital)
        positions: dict[str, _PositionLot] = {}
        last_known_close: dict[str, float] = {}
        last_bar_ts: dict[str, datetime] = {}
        pending_queue: list[_PendingExecution] = []

        high_water_mark = cash
        prev_day_equity = cash
        prev_ts: datetime | None = None
        bankrupt = False

        equity_curve: list[EquityPoint] = []
        trades: list[SimulatedTrade] = []
        benchmark_closes: list[float] = []

        total_traded_notional = 0.0
        total_orders_filled = 0
        stop_loss_exits = 0
        risk_rejections = 0
        total_slippage_usd = 0.0
        total_borrow_cost_usd = 0.0
        total_sec_fees_usd = 0.0

        # Pre-seed last_known_close & last_bar_ts from the last bar before sim_timestamps[0]
        first_sim_ns = int(pd.Timestamp(sim_timestamps[0]).as_unit("ns").value)
        for sym, df in sorted_bars.items():
            pos = int(np.searchsorted(ts_arrays[sym], first_sim_ns, side="left"))
            if pos > 0:
                last_known_close[sym] = float(df["close"].iloc[pos - 1])
                last_bar_ts[sym] = ensure_utc(df["timestamp"].iloc[pos - 1].to_pydatetime())

        for bar_idx, ts in enumerate(sim_timestamps):
            # 1. Accrue short borrow cost using exact elapsed calendar days / 365
            if prev_ts is not None and bt_cfg.short_borrow_apr > 0.0 and not bankrupt:
                elapsed_days = max(0.0, (ts - prev_ts).total_seconds() / 86400.0)
                day_frac = elapsed_days / 365.0
                if day_frac > 0.0:
                    for sym, lot in positions.items():
                        if lot.qty < -1e-9:
                            mark_px = last_known_close.get(sym, lot.avg_entry_price)
                            borrow_fee = abs(lot.qty) * mark_px * bt_cfg.short_borrow_apr * day_frac
                            cash -= borrow_fee
                            lot.accrued_borrow_usd += borrow_fee
                            total_borrow_cost_usd += borrow_fee

            # 2. Check open-gap stop-losses on existing positions at bar t open
            stopped_at_open: set[str] = set()
            if not bankrupt:
                for sym in list(positions.keys()):
                    lot = positions[sym]
                    if lot.stop_loss_price is None:
                        continue
                    bar = bar_lookup.get(sym, {}).get(ts)
                    if bar is None:
                        continue
                    bar_open, _, _, _ = bar
                    slip_rate = self._slippage_rate(sym)
                    if lot.qty > 1e-9 and bar_open <= lot.stop_loss_price:
                        fill_px = bar_open * (1.0 - slip_rate)
                        qty = lot.qty
                        slip_usd = qty * (bar_open - fill_px)
                        sec_fee = qty * fill_px * (bt_cfg.sec_sell_fee_bps / 10_000.0)
                        cash += qty * fill_px - sec_fee
                        total_traded_notional += qty * fill_px
                        total_orders_filled += 1
                        stop_loss_exits += 1
                        total_slippage_usd += slip_usd
                        total_sec_fees_usd += sec_fee
                        gross_pnl = (fill_px - lot.avg_entry_price) * qty
                        net_pnl = (
                            gross_pnl - sec_fee - lot.accrued_sec_fee_usd - lot.accrued_borrow_usd
                        )
                        trades.append(
                            SimulatedTrade(
                                symbol=sym,
                                side="long",
                                entry_time=lot.entry_time,
                                exit_time=ts,
                                qty=qty,
                                entry_price=round(lot.avg_entry_price, 6),
                                exit_price=round(fill_px, 6),
                                gross_pnl=round(gross_pnl, 4),
                                net_pnl=round(net_pnl, 4),
                                holding_days=max(
                                    0.0, (ts - lot.entry_time).total_seconds() / 86400.0
                                ),
                                exit_reason="stop_loss_open_gap",
                            )
                        )
                        del positions[sym]
                        stopped_at_open.add(sym)
                    elif lot.qty < -1e-9 and bar_open >= lot.stop_loss_price:
                        fill_px = bar_open * (1.0 + slip_rate)
                        qty = abs(lot.qty)
                        slip_usd = qty * (fill_px - bar_open)
                        cash -= qty * fill_px
                        total_traded_notional += qty * fill_px
                        total_orders_filled += 1
                        stop_loss_exits += 1
                        total_slippage_usd += slip_usd
                        gross_pnl = (lot.avg_entry_price - fill_px) * qty
                        net_pnl = gross_pnl - lot.accrued_sec_fee_usd - lot.accrued_borrow_usd
                        trades.append(
                            SimulatedTrade(
                                symbol=sym,
                                side="short",
                                entry_time=lot.entry_time,
                                exit_time=ts,
                                qty=qty,
                                entry_price=round(lot.avg_entry_price, 6),
                                exit_price=round(fill_px, 6),
                                gross_pnl=round(gross_pnl, 4),
                                net_pnl=round(net_pnl, 4),
                                holding_days=max(
                                    0.0, (ts - lot.entry_time).total_seconds() / 86400.0
                                ),
                                exit_reason="stop_loss_open_gap",
                            )
                        )
                        del positions[sym]
                        stopped_at_open.add(sym)

            # 3. Execute queued approved orders from bar t-1 close at bar t open
            ordered_pending = sorted(
                pending_queue,
                key=lambda item: 0 if item.verdict.order.is_risk_reducing else 1,
            )
            pending_queue = []
            carried_exits: list[_PendingExecution] = []

            # Build open-price mark table for live post-gap exposure checks
            open_marks: dict[str, float] = dict(last_known_close)
            for sym, sym_bars in bar_lookup.items():
                b = sym_bars.get(ts)
                if b is not None and math.isfinite(b[0]) and b[0] > 0.0:
                    open_marks[sym] = b[0]

            if not bankrupt:
                for item in ordered_pending:
                    verdict = item.verdict
                    order = verdict.order
                    sym = order.symbol
                    if sym in stopped_at_open:
                        continue
                    bar = bar_lookup.get(sym, {}).get(ts)
                    if bar is None or not math.isfinite(bar[0]) or bar[0] <= 0.0:
                        # Carry forward Pass 1 risk-reducing exits if bar is temporarily missing!
                        if order.is_risk_reducing and sym in positions:
                            carried_exits.append(item)
                        continue
                    bar_open = bar[0]

                    qty = verdict.adjusted_qty
                    if qty <= 1e-9:
                        continue

                    existing = positions.get(sym)
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
                        # Guard Pass 2 entries against overnight gaps & cash/exposure breaches!
                        is_short_entry = order.side == OrderSide.sell and (
                            existing is None or existing.qty <= 1e-9
                        )

                        # (a) Cancel Pass 2 entry if bar_open already gapped past its planned stop
                        if verdict.stop_loss_price is not None:
                            if not is_short_entry and bar_open <= verdict.stop_loss_price:
                                risk_rejections += 1
                                continue
                            if is_short_entry and bar_open >= verdict.stop_loss_price:
                                risk_rejections += 1
                                continue

                        # (b) Re-check live open equity, cash, gross, net & position headroom
                        open_equity = cash + sum(
                            lot.qty * open_marks.get(s, lot.avg_entry_price)
                            for s, lot in positions.items()
                        )
                        if open_equity <= 0.0:
                            risk_rejections += 1
                            continue

                        est_fill_px = (
                            bar_open * (1.0 + slip_rate)
                            if order.side == OrderSide.buy
                            else bar_open * (1.0 - slip_rate)
                        )
                        cur_gross = sum(
                            abs(lot.qty * open_marks.get(s, lot.avg_entry_price))
                            for s, lot in positions.items()
                        )
                        cur_net = sum(
                            lot.qty * open_marks.get(s, lot.avg_entry_price)
                            for s, lot in positions.items()
                        )
                        eff_gross_cap = (
                            risk_cfg.max_gross_exposure * min(1.0, max(0.0, exposure_multiplier))
                        ) * open_equity
                        gross_headroom = max(0.0, eff_gross_cap - cur_gross)
                        if is_short_entry:
                            net_headroom = max(0.0, cur_net - risk_cfg.min_net_exposure * open_equity)
                            pos_cap = risk_cfg.max_short_position_pct * open_equity
                        else:
                            net_headroom = max(0.0, risk_cfg.max_net_exposure * open_equity - cur_net)
                            pos_cap = risk_cfg.max_long_position_pct * open_equity

                        cur_pos_val = abs(existing.qty) * bar_open if existing is not None else 0.0
                        pos_headroom = max(0.0, pos_cap - cur_pos_val)

                        # Prevent unmodeled free margin loans on longs (cash must stay >= 0)
                        cash_headroom = (
                            max(0.0, cash) if order.side == OrderSide.buy else float("inf")
                        )

                        max_notional = min(
                            qty * est_fill_px,
                            gross_headroom,
                            net_headroom,
                            pos_headroom,
                            cash_headroom,
                        )
                        if max_notional <= 1e-6:
                            risk_rejections += 1
                            continue

                        if is_short_entry or not strat_cfg.allow_fractional_longs:
                            qty = float(math.floor(max_notional / est_fill_px + 1e-12))
                            if qty < 1.0:
                                risk_rejections += 1
                                continue
                        else:
                            qty = math.floor((max_notional / est_fill_px) * 1e6) / 1e6
                            if qty <= 1e-6:
                                risk_rejections += 1
                                continue

                    if order.side == OrderSide.buy:
                        fill_px = bar_open * (1.0 + slip_rate)
                        slip_usd = qty * (fill_px - bar_open)
                        sec_fee = 0.0
                        cash -= qty * fill_px
                    else:
                        fill_px = bar_open * (1.0 - slip_rate)
                        slip_usd = qty * (bar_open - fill_px)
                        sec_fee = qty * fill_px * (bt_cfg.sec_sell_fee_bps / 10_000.0)
                        cash += qty * fill_px - sec_fee

                    total_traded_notional += qty * fill_px
                    total_orders_filled += 1
                    total_slippage_usd += slip_usd
                    total_sec_fees_usd += sec_fee

                    delta_signed = qty if order.side == OrderSide.buy else -qty
                    if existing is None:
                        new_stop = self._reanchor_stop(
                            fill_px=fill_px,
                            is_short=(delta_signed < 0.0),
                            dist_pct=item.stop_loss_dist_pct,
                            fallback_stop=verdict.stop_loss_price,
                        )
                        positions[sym] = _PositionLot(
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
                                    existing.stop_loss_price = max(
                                        existing.stop_loss_price, new_stop
                                    )
                                else:
                                    existing.stop_loss_price = min(
                                        existing.stop_loss_price, new_stop
                                    )
                        else:
                            # Guard against accidental sign crossing
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

                            trades.append(
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
                                    exit_reason=order.reason or (
                                        "rebalance_exit" if is_reducing else "rebalance_reduce"
                                    ),
                                )
                            )
                            new_q = existing.qty + delta_signed
                            if abs(new_q) <= 1e-9 or (
                                existing.qty > 0.0 and new_q < 0.0
                            ) or (existing.qty < 0.0 and new_q > 0.0):
                                del positions[sym]
                            else:
                                existing.qty = new_q

            # 4. Intra-bar stop-loss checks on all open positions (bar t high/low)
            if not bankrupt:
                for sym in list(positions.keys()):
                    lot = positions[sym]
                    if lot.stop_loss_price is None:
                        continue
                    bar = bar_lookup.get(sym, {}).get(ts)
                    if bar is None:
                        continue
                    bar_open, bar_high, bar_low, _ = bar
                    slip_rate = self._slippage_rate(sym)
                    if lot.qty > 1e-9 and bar_low <= lot.stop_loss_price:
                        trigger_px = min(bar_open, lot.stop_loss_price)
                        fill_px = trigger_px * (1.0 - slip_rate)
                        qty = lot.qty
                        slip_usd = qty * (trigger_px - fill_px)
                        sec_fee = qty * fill_px * (bt_cfg.sec_sell_fee_bps / 10_000.0)
                        cash += qty * fill_px - sec_fee
                        total_traded_notional += qty * fill_px
                        total_orders_filled += 1
                        stop_loss_exits += 1
                        total_slippage_usd += slip_usd
                        total_sec_fees_usd += sec_fee
                        gross_pnl = (fill_px - lot.avg_entry_price) * qty
                        net_pnl = (
                            gross_pnl - sec_fee - lot.accrued_sec_fee_usd - lot.accrued_borrow_usd
                        )
                        trades.append(
                            SimulatedTrade(
                                symbol=sym,
                                side="long",
                                entry_time=lot.entry_time,
                                exit_time=ts,
                                qty=qty,
                                entry_price=round(lot.avg_entry_price, 6),
                                exit_price=round(fill_px, 6),
                                gross_pnl=round(gross_pnl, 4),
                                net_pnl=round(net_pnl, 4),
                                holding_days=max(
                                    0.0, (ts - lot.entry_time).total_seconds() / 86400.0
                                ),
                                exit_reason="stop_loss_intraday",
                            )
                        )
                        del positions[sym]
                    elif lot.qty < -1e-9 and bar_high >= lot.stop_loss_price:
                        trigger_px = max(bar_open, lot.stop_loss_price)
                        fill_px = trigger_px * (1.0 + slip_rate)
                        qty = abs(lot.qty)
                        slip_usd = qty * (fill_px - trigger_px)
                        cash -= qty * fill_px
                        total_traded_notional += qty * fill_px
                        total_orders_filled += 1
                        stop_loss_exits += 1
                        total_slippage_usd += slip_usd
                        gross_pnl = (lot.avg_entry_price - fill_px) * qty
                        net_pnl = gross_pnl - lot.accrued_sec_fee_usd - lot.accrued_borrow_usd
                        trades.append(
                            SimulatedTrade(
                                symbol=sym,
                                side="short",
                                entry_time=lot.entry_time,
                                exit_time=ts,
                                qty=qty,
                                entry_price=round(lot.avg_entry_price, 6),
                                exit_price=round(fill_px, 6),
                                gross_pnl=round(gross_pnl, 4),
                                net_pnl=round(net_pnl, 4),
                                holding_days=max(
                                    0.0, (ts - lot.entry_time).total_seconds() / 86400.0
                                ),
                                exit_reason="stop_loss_intraday",
                            )
                        )
                        del positions[sym]

            # 5. Mark-to-market at bar t close (distinguish live bars vs carried marks!)
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

            # Optional end-of-window mark-to-close liquidation for exact trade reconciliation
            if close_open_positions_at_end and ts == sim_timestamps[-1] and positions and not bankrupt:
                for sym in list(positions.keys()):
                    lot = positions[sym]
                    px = mtm_prices.get(sym, lot.avg_entry_price)
                    slip_rate = self._slippage_rate(sym)
                    if lot.qty > 1e-9:
                        fill_px = px * (1.0 - slip_rate)
                        qty = lot.qty
                        slip_usd = qty * (px - fill_px)
                        sec_fee = qty * fill_px * (bt_cfg.sec_sell_fee_bps / 10_000.0)
                        cash += qty * fill_px - sec_fee
                        total_traded_notional += qty * fill_px
                        total_orders_filled += 1
                        total_slippage_usd += slip_usd
                        total_sec_fees_usd += sec_fee
                        gross_pnl = (fill_px - lot.avg_entry_price) * qty
                        net_pnl = (
                            gross_pnl - sec_fee - lot.accrued_sec_fee_usd - lot.accrued_borrow_usd
                        )
                        trades.append(
                            SimulatedTrade(
                                symbol=sym,
                                side="long",
                                entry_time=lot.entry_time,
                                exit_time=ts,
                                qty=qty,
                                entry_price=round(lot.avg_entry_price, 6),
                                exit_price=round(fill_px, 6),
                                gross_pnl=round(gross_pnl, 4),
                                net_pnl=round(net_pnl, 4),
                                holding_days=max(
                                    0.0, (ts - lot.entry_time).total_seconds() / 86400.0
                                ),
                                exit_reason="end_of_window",
                            )
                        )
                    else:
                        fill_px = px * (1.0 + slip_rate)
                        qty = abs(lot.qty)
                        slip_usd = qty * (fill_px - px)
                        cash -= qty * fill_px
                        total_traded_notional += qty * fill_px
                        total_orders_filled += 1
                        total_slippage_usd += slip_usd
                        gross_pnl = (lot.avg_entry_price - fill_px) * qty
                        net_pnl = gross_pnl - lot.accrued_sec_fee_usd - lot.accrued_borrow_usd
                        trades.append(
                            SimulatedTrade(
                                symbol=sym,
                                side="short",
                                entry_time=lot.entry_time,
                                exit_time=ts,
                                qty=qty,
                                entry_price=round(lot.avg_entry_price, 6),
                                exit_price=round(fill_px, 6),
                                gross_pnl=round(gross_pnl, 4),
                                net_pnl=round(net_pnl, 4),
                                holding_days=max(
                                    0.0, (ts - lot.entry_time).total_seconds() / 86400.0
                                ),
                                exit_reason="end_of_window",
                            )
                        )
                    del positions[sym]

            mtm_value = 0.0
            gross_notional = 0.0
            net_notional = 0.0
            for sym, lot in positions.items():
                px = mtm_prices.get(sym, lot.avg_entry_price)
                val = lot.qty * px
                mtm_value += val
                gross_notional += abs(val)
                net_notional += val

            equity = cash + mtm_value
            if equity <= 0.0:
                bankrupt = True
                equity = 0.0
                positions.clear()
                carried_exits.clear()

            high_water_mark = max(high_water_mark, equity)
            drawdown_pct = (
                (equity - high_water_mark) / high_water_mark if high_water_mark > 1e-9 else 0.0
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
                    cash=round(cash, 2),
                    gross_exposure=round(gross_exp, 4),
                    net_exposure=round(net_exp, 4),
                    daily_pnl_pct=round(daily_pnl_pct, 6),
                    drawdown_pct=round(min(0.0, drawdown_pct), 6),
                )
            )
            # Only append genuine live benchmark bars (never carry forward stale SPY closes!)
            if bt_cfg.benchmark_symbol in live_close_prices:
                benchmark_closes.append(live_close_prices[bt_cfg.benchmark_symbol])

            # 6. Compute strictly point-in-time signals & queue approved orders for bar t+1 open
            pending_queue = list(carried_exits)
            carried_symbols = {item.verdict.order.symbol for item in carried_exits}
            is_rebalance_day = (bar_idx % strat_cfg.rebalance_every_n_days) == 0

            if is_rebalance_day and ts != sim_timestamps[-1] and not bankrupt and equity > 0.0:
                ts_ns = int(pd.Timestamp(ts).as_unit("ns").value)
                pit_bars: dict[str, pd.DataFrame] = {}
                fresh_bar_ages: list[float] = []

                for sym, df in sorted_bars.items():
                    # Exclude stale/halted symbols that lack a fresh bar at ts!
                    sym_last_ts = last_bar_ts.get(sym)
                    if sym_last_ts is None:
                        continue
                    age_min = (ts - sym_last_ts).total_seconds() / 60.0
                    if age_min > risk_cfg.max_data_staleness_minutes:
                        continue
                    end_idx = int(np.searchsorted(ts_arrays[sym], ts_ns, side="right"))
                    if end_idx >= MIN_HISTORY_BARS:
                        start_idx = max(0, end_idx - _FEATURE_LOOKBACK_BARS)
                        pit_bars[sym] = df.iloc[start_idx:end_idx]
                        fresh_bar_ages.append(max(0.0, age_min))

                # Real bar-age calculation (or stale sentinel if no fresh bars exist)
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

                signals = feature_engine.build_signals(
                    as_of=ts,
                    bars_by_symbol=pit_bars,
                    news_by_symbol=pit_news if use_news else None,
                    macro_points=macro_points,
                    benchmark_symbol=bt_cfg.benchmark_symbol,
                )

                dials = ControlDials(
                    pause_new_entries=False,
                    exposure_multiplier=exposure_multiplier,
                    allow_shorting=allow_shorting,
                    updated_by="backtest",
                    reason=label,
                    updated_at=ts,
                )
                signed_pos = {sym: lot.qty for sym, lot in positions.items()}
                proposed = portfolio_ctor.build_proposed_orders(
                    signals=signals,
                    current_positions=signed_pos,
                    prices=mtm_prices,
                    equity=equity,
                    dials=dials,
                    as_of=ts,
                    universe_meta=self.universe_meta,
                )
                # Avoid duplicate orders for symbols that already have a carried Pass 1 exit
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

                for v in verdicts:
                    if v.approved and v.adjusted_qty > 0.0:
                        dist_pct: float | None = None
                        if v.stop_loss_price is not None and v.order.reference_price > 0.0:
                            dist_pct = (
                                abs(v.stop_loss_price - v.order.reference_price)
                                / v.order.reference_price
                            )
                        pending_queue.append(
                            _PendingExecution(verdict=v, stop_loss_dist_pct=dist_pct)
                        )
                    else:
                        risk_rejections += 1

            prev_day_equity = equity
            prev_ts = ts

        metrics = compute_backtest_metrics(
            label=label,
            equity_curve=equity_curve,
            trades=trades,
            initial_capital=bt_cfg.initial_capital,
            total_traded_notional=total_traded_notional,
            total_orders_filled=total_orders_filled,
            stop_loss_exits=stop_loss_exits,
            risk_rejections=risk_rejections,
            total_slippage_usd=total_slippage_usd,
            total_borrow_cost_usd=total_borrow_cost_usd,
            total_sec_fees_usd=total_sec_fees_usd,
            benchmark_symbol=bt_cfg.benchmark_symbol,
            benchmark_closes=benchmark_closes,
        )
        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trades,
            config_snapshot={
                "label": label,
                "allow_shorting": allow_shorting,
                "use_news": use_news,
                "exposure_multiplier": exposure_multiplier,
            },
        )

    def run_ablation(
        self,
        bars_by_symbol: Mapping[str, pd.DataFrame],
        *,
        news_by_symbol: Mapping[str, Sequence[NewsArticle]] | None = None,
        macro_points: Sequence[MacroPoint] | None = None,
        earnings_by_symbol: Mapping[str, Sequence[date]] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
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
                start_date=start_date,
                end_date=end_date,
                label=label,
                allow_shorting=allow_short,
                use_news=use_news,
            )
            for label, allow_short, use_news in variants
        ]

    def run_walk_forward(
        self,
        bars_by_symbol: Mapping[str, pd.DataFrame],
        *,
        news_by_symbol: Mapping[str, Sequence[NewsArticle]] | None = None,
        macro_points: Sequence[MacroPoint] | None = None,
        earnings_by_symbol: Mapping[str, Sequence[date]] | None = None,
        n_folds: int = 3,
        allow_shorting: bool = True,
        use_news: bool = True,
    ) -> list[BacktestMetrics]:
        """Evaluate strategy across sequential out-of-sample calendar folds."""
        all_dates: list[date] = sorted(
            {
                ensure_utc(pd.Timestamp(ts).tz_convert("UTC").to_pydatetime()).date()
                if pd.Timestamp(ts).tzinfo is not None
                else ensure_utc(pd.Timestamp(ts, tz="UTC").to_pydatetime()).date()
                for df in bars_by_symbol.values()
                if not df.empty
                for ts in df["timestamp"]
            }
        )
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
                start_date=eval_dates[lo],
                end_date=eval_dates[hi],
                label=f"Fold {f_idx + 1}",
                allow_shorting=allow_shorting,
                use_news=use_news,
            )
            fold_metrics.append(res.metrics)
        return fold_metrics

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
