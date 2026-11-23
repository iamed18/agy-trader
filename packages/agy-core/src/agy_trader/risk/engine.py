from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

from agy_trader.core.calendar import nyse_trading_days_apart
from agy_trader.core.config import RiskConfig
from agy_trader.core.models import (
    AssetRiskMetadata,
    AssetType,
    ControlDials,
    OrderSide,
    ProposedOrder,
    RiskVerdict,
    SignalRow,
    UtcDatetime,
    ensure_utc,
)

NY_TZ = ZoneInfo("America/New_York")
MAX_REF_PRICE_DEVIATION = 0.02  # 2% max divergence between order.reference_price and state.prices
MAX_FALLBACK_LONG_STOP_PCT = 0.25  # Hard 25% max stop width if signal is unavailable on long
_trading_days_apart = nyse_trading_days_apart


class PortfolioRiskState(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, frozen=True)

    as_of: UtcDatetime
    equity: float
    daily_pnl_pct: float
    drawdown_pct: float = Field(le=0.0)  # Must be <= 0.0 (e.g., -0.05 for 5% drawdown)
    positions: dict[str, float] = Field(default_factory=dict)
    prices: dict[str, float] = Field(default_factory=dict)
    latest_bar_age_minutes: float | None = Field(default=None, ge=0.0)


def _signed_delta(side: OrderSide, qty: float) -> float:
    return qty if side == OrderSide.buy else -qty


def _is_truly_risk_reducing(current_qty: float, side: OrderSide, qty: float) -> bool:
    """Independently verify an order strictly reduces absolute exposure without crossing zero."""
    if abs(current_qty) <= 1e-9 or qty <= 0.0:
        return False
    delta = _signed_delta(side, qty)
    if current_qty > 0.0:
        return delta < 0.0 and qty <= current_qty + 1e-9
    return delta > 0.0 and qty <= abs(current_qty) + 1e-9


def _floor_decimals(value: float, decimals: int = 6) -> float:
    scale = 10**decimals
    return math.floor(value * scale + 1e-12) / scale


class HardRiskEngine:
    """Immutable hard risk gatekeeper enforcing all position, short, sector & kill-switch rules.

    Evaluates batches cumulatively against projected post-order portfolio state so multiple
    simultaneous orders can never stack past sector, gross, or net exposure ceilings.
    """

    def __init__(
        self,
        config: RiskConfig,
        allow_fractional_longs: bool = True,
        max_signal_age_hours: float = 96.0,
    ) -> None:
        self.config = config
        self.allow_fractional_longs = allow_fractional_longs
        self.max_signal_age_hours = max_signal_age_hours

    def evaluate_batch(
        self,
        orders: Sequence[ProposedOrder],
        state: PortfolioRiskState,
        dials: ControlDials,
        universe_meta: Mapping[str, AssetRiskMetadata],
        signals_by_symbol: Mapping[str, SignalRow] | None = None,
        earnings_by_symbol: Mapping[str, Sequence[date]] | None = None,
        frozen_symbols: set[str] | frozenset[str] = frozenset(),
        blackout_halted: bool = False,
    ) -> list[RiskVerdict]:
        """Validate a batch of ProposedOrders against immutable hard risk limits."""
        as_of_utc = ensure_utc(state.as_of)
        as_of_et_date = as_of_utc.astimezone(NY_TZ).date()
        signals_map = {k.upper(): v for k, v in (signals_by_symbol or {}).items()}
        earnings_map = {k.upper(): v for k, v in (earnings_by_symbol or {}).items()}
        blacklist = dials.active_blacklist(as_of_utc)
        frozen_set = frozenset(s.strip().upper() for s in frozen_symbols if s.strip())

        projected: dict[str, float] = {k.upper(): float(v) for k, v in state.positions.items()}
        prices: dict[str, float] = {k.upper(): float(v) for k, v in state.prices.items()}

        verdicts_by_idx: dict[int, RiskVerdict] = {}

        # Layer 2 BLACKOUT_PERIOD compliance rail: enforce BEFORE Pass-1/Pass-2 split so
        # BOTH risk-reducing exits/trims AND risk-increasing entries are strictly blocked.
        eligible_orders: list[tuple[int, ProposedOrder]] = []
        for idx, order in enumerate(orders):
            if blackout_halted:
                verdicts_by_idx[idx] = RiskVerdict(
                    approved=False,
                    order=order,
                    adjusted_qty=0.0,
                    violations=["BLACKOUT_PERIOD_HALT"],
                )
            elif order.symbol in frozen_set:
                verdicts_by_idx[idx] = RiskVerdict(
                    approved=False,
                    order=order,
                    adjusted_qty=0.0,
                    violations=["BLACKOUT_PERIOD_FROZEN"],
                )
            else:
                eligible_orders.append((idx, order))

        # Partition into Pass 1 (genuinely risk-reducing) and Pass 2 (risk-increasing)
        # while tracking remaining reducible shares per symbol so side-flip open legs
        # are routed to Pass 2 instead of falsely classified as Pass 1 reductions!
        remaining_reducible: dict[str, float] = dict(projected)
        reducing_indices: list[tuple[int, ProposedOrder]] = []
        increasing_indices: list[tuple[int, ProposedOrder]] = []

        # Prioritize orders explicitly flagged is_risk_reducing=True when claiming reduction capacity
        ordered_candidates = sorted(
            eligible_orders,
            key=lambda pair: (0 if pair[1].is_risk_reducing else 1, pair[0]),
        )
        for idx, order in ordered_candidates:
            rem_q = remaining_reducible.get(order.symbol, 0.0)
            if _is_truly_risk_reducing(rem_q, order.side, order.qty):
                reducing_indices.append((idx, order))
                new_rem = rem_q + _signed_delta(order.side, order.qty)
                remaining_reducible[order.symbol] = 0.0 if abs(new_rem) <= 1e-9 else new_rem
            else:
                increasing_indices.append((idx, order))

        # Preserve caller relative order within each pass
        reducing_indices.sort(key=lambda pair: pair[0])
        increasing_indices.sort(key=lambda pair: pair[0])

        # Pass 1: Risk-reducing exits/trims (always allowed even during kill-switch / pause)
        for idx, order in reducing_indices:
            cur_q = projected.get(order.symbol, 0.0)
            if not _is_truly_risk_reducing(cur_q, order.side, order.qty):
                verdicts_by_idx[idx] = RiskVerdict(
                    approved=False,
                    order=order,
                    adjusted_qty=0.0,
                    violations=["NO_OPEN_POSITION_TO_REDUCE"],
                )
                continue

            max_close_qty = abs(cur_q)
            is_full_close = order.qty >= max_close_qty - 1e-9

            if cur_q < -1e-9:
                # Covering a short: full close allowed verbatim; partial cover must leave
                # an integer remaining short share count (no fractional short positions!)
                if is_full_close:
                    approved_qty = max_close_qty
                else:
                    rem_after = max_close_qty - order.qty
                    rem_int = math.ceil(rem_after - 1e-12)
                    approved_qty = max_close_qty - rem_int
                    if approved_qty < 1.0 - 1e-9:
                        verdicts_by_idx[idx] = RiskVerdict(
                            approved=False,
                            order=order,
                            adjusted_qty=0.0,
                            violations=["FRACTIONAL_SHORT_REMAINDER_REJECTED"],
                        )
                        continue
            else:
                # Reducing a long: never overshoot cur_q via rounding
                if is_full_close:
                    approved_qty = max_close_qty
                else:
                    approved_qty = min(max_close_qty, _floor_decimals(order.qty, 6))
                    if approved_qty <= 1e-6:
                        verdicts_by_idx[idx] = RiskVerdict(
                            approved=False,
                            order=order,
                            adjusted_qty=0.0,
                            violations=["ORDER_QTY_BELOW_MINIMUM"],
                        )
                        continue

            delta = _signed_delta(order.side, approved_qty)
            new_q = cur_q + delta
            if abs(new_q) <= 1e-9:
                new_q = 0.0
            projected[order.symbol] = new_q
            verdicts_by_idx[idx] = RiskVerdict(
                approved=True,
                order=order,
                adjusted_qty=approved_qty,
                stop_loss_price=None,
                violations=[],
            )

        # Pass 2: Risk-increasing entries/adds (evaluated cumulatively against projected portfolio)
        for idx, order in increasing_indices:
            verdict = self._evaluate_risk_increasing(
                order=order,
                projected=projected,
                prices=prices,
                state=state,
                as_of_utc=as_of_utc,
                as_of_et_date=as_of_et_date,
                dials=dials,
                blacklist=blacklist,
                universe_meta=universe_meta,
                signals_map=signals_map,
                earnings_map=earnings_map,
            )
            if verdict.approved and verdict.adjusted_qty > 0.0:
                sym = order.symbol
                projected[sym] = projected.get(sym, 0.0) + _signed_delta(
                    order.side, verdict.adjusted_qty
                )
            verdicts_by_idx[idx] = verdict

        return [verdicts_by_idx[i] for i in range(len(orders))]

    def _evaluate_risk_increasing(
        self,
        order: ProposedOrder,
        projected: Mapping[str, float],
        prices: Mapping[str, float],
        state: PortfolioRiskState,
        as_of_utc: datetime,
        as_of_et_date: date,
        dials: ControlDials,
        blacklist: set[str],
        universe_meta: Mapping[str, AssetRiskMetadata],
        signals_map: Mapping[str, SignalRow],
        earnings_map: Mapping[str, Sequence[date]],
    ) -> RiskVerdict:
        violations: list[str] = []
        sym = order.symbol

        # 0. Equity & numeric sanity
        if not math.isfinite(state.equity) or state.equity <= 0.0:
            violations.append("NON_POSITIVE_EQUITY")
            return RiskVerdict(approved=False, order=order, adjusted_qty=0.0, violations=violations)

        # 1. Hard Kill-Switches
        if not math.isfinite(state.daily_pnl_pct) or state.daily_pnl_pct <= -self.config.daily_loss_halt_pct:
            violations.append("KILL_SWITCH_DAILY_LOSS")
        if not math.isfinite(state.drawdown_pct) or state.drawdown_pct <= -self.config.max_drawdown_halt_pct:
            violations.append("KILL_SWITCH_MAX_DRAWDOWN")

        # 2. Data Staleness Guard (fail closed when bar age is omitted, negative, or stale)
        if (
            state.latest_bar_age_minutes is None
            or not math.isfinite(state.latest_bar_age_minutes)
            or state.latest_bar_age_minutes < 0.0
            or state.latest_bar_age_minutes > self.config.max_data_staleness_minutes
        ):
            violations.append("STALE_MARKET_DATA")

        # 3. Defense-in-depth ControlDials checks
        if dials.pause_new_entries:
            violations.append("PAUSED_BY_OPS")
        if sym in blacklist:
            violations.append("BLACKLISTED_SYMBOL")

        # 4. Price integrity: fail closed if symbol or ANY open position lacks valid finite positive price
        state_px = prices.get(sym)
        if state_px is None or not math.isfinite(state_px) or state_px <= 0.0:
            violations.append("MISSING_OR_INVALID_SYMBOL_PRICE")
        else:
            rel_dev = abs(order.reference_price - state_px) / state_px
            if rel_dev > MAX_REF_PRICE_DEVIATION:
                violations.append("REFERENCE_PRICE_DIVERGENCE")

        for p_sym, p_qty in projected.items():
            if abs(p_qty) <= 1e-9:
                continue
            p_px = prices.get(p_sym)
            if p_px is None or not math.isfinite(p_px) or p_px <= 0.0:
                violations.append("MISSING_OR_INVALID_PORTFOLIO_PRICE")
                break

        # 5. Reject single-order sign flips (caller must split into close + open)
        cur_q = projected.get(sym, 0.0)
        delta_full = _signed_delta(order.side, order.qty)
        if (cur_q > 1e-9 and delta_full < 0.0) or (cur_q < -1e-9 and delta_full > 0.0):
            violations.append("UNSPLIT_SIDE_FLIP_REJECTED")

        # Determine whether this order opens/increases a LONG or SHORT position
        is_short_order = order.side == OrderSide.sell and cur_q <= 1e-9

        # Validate signal timestamp if present (no future-dated or stale signals)
        sig = signals_map.get(sym)
        if sig is not None:
            sig_age_hours = (as_of_utc - sig.as_of).total_seconds() / 3600.0
            if sig_age_hours < -1e-6 or sig_age_hours > self.max_signal_age_hours:
                violations.append("STALE_OR_FUTURE_SIGNAL")
                sig = None

        if (
            not is_short_order
            and sig is not None
            and sig.has_recent_restatement
            and self.config.restatement_blackout_days > 0
        ):
            violations.append("RESTATEMENT_BLACKOUT")

        # 6. Asset Metadata, Earnings Blackout & Shorting Safety Checks
        meta = universe_meta.get(sym)
        if meta is None:
            violations.append("UNKNOWN_ASSET_METADATA")
        else:
            if (
                meta.asset_type == AssetType.equity
                and self.config.earnings_blackout_days is not None
                and self.config.earnings_blackout_days >= 0
            ):
                for earn_dt in earnings_map.get(sym, ()):
                    if (
                        _trading_days_apart(as_of_et_date, earn_dt)
                        <= self.config.earnings_blackout_days
                    ):
                        violations.append("EARNINGS_BLACKOUT")
                        break

            if is_short_order:
                if not dials.allow_shorting:
                    violations.append("SHORTING_DISABLED_BY_OPS")
                if self.config.max_short_position_pct <= 0.0:
                    violations.append("SHORTING_DISABLED_BY_RISK_CONFIG")
                if not meta.shortable:
                    violations.append("ASSET_NOT_SHORTABLE")
                if not meta.easy_to_borrow:
                    violations.append("ASSET_NOT_EASY_TO_BORROW")

                if sig is None:
                    violations.append("SHORT_MISSING_SIGNAL_FOR_SQUEEZE_CHECK")
                else:
                    if sig.ret_5d >= self.config.short_squeeze_5d_return_cap:
                        violations.append("SHORT_SQUEEZE_5D_RETURN_GUARD")
                    if sig.ann_vol >= self.config.short_squeeze_ann_vol_cap:
                        violations.append("SHORT_SQUEEZE_VOLATILITY_GUARD")

        # Conservative pricing anchor for sizing and stops
        sizing_px = (
            max(order.reference_price, state_px)
            if (state_px is not None and math.isfinite(state_px) and state_px > 0.0)
            else order.reference_price
        )
        anchor_px = (
            state_px
            if (state_px is not None and math.isfinite(state_px) and state_px > 0.0)
            else order.reference_price
        )

        # 7. Mandatory directional stop-loss validation / synthesis
        stop_price = self._resolve_stop_loss(
            order=order,
            anchor_px=anchor_px,
            is_short=is_short_order,
            signal=sig,
        )
        if stop_price is None:
            violations.append("MISSING_OR_INVALID_STOP_LOSS")

        if violations:
            return RiskVerdict(approved=False, order=order, adjusted_qty=0.0, violations=violations)

        assert meta is not None
        assert state_px is not None

        # 8. Cumulative Exposure Headroom Calculations (Position, Sector, Gross, Net)
        equity = state.equity
        current_pos_notional = abs(cur_q) * sizing_px
        pos_cap_pct = (
            self.config.max_short_position_pct
            if is_short_order
            else self.config.max_long_position_pct
        )
        pos_headroom = max(0.0, pos_cap_pct * equity - current_pos_notional)

        cur_gross = 0.0
        cur_net = 0.0
        cur_sector_gross = 0.0
        for p_sym, p_qty in projected.items():
            if abs(p_qty) <= 1e-9:
                continue
            p_px = prices[p_sym]
            notional = p_qty * p_px
            cur_gross += abs(notional)
            cur_net += notional
            p_meta = universe_meta.get(p_sym)
            if p_meta is not None and p_meta.sector == meta.sector:
                cur_sector_gross += abs(notional)

        sector_headroom = max(0.0, self.config.max_sector_gross_pct * equity - cur_sector_gross)

        effective_gross_cap = (
            self.config.max_gross_exposure * min(1.0, max(0.0, dials.exposure_multiplier)) * equity
        )
        gross_headroom = max(0.0, effective_gross_cap - cur_gross)

        if is_short_order:
            net_headroom = max(0.0, cur_net - self.config.min_net_exposure * equity)
        else:
            net_headroom = max(0.0, self.config.max_net_exposure * equity - cur_net)

        allowed_notional = min(
            order.qty * sizing_px,
            pos_headroom,
            sector_headroom,
            gross_headroom,
            net_headroom,
        )
        if allowed_notional <= 1e-9:
            cap_reasons: list[str] = []
            if pos_headroom <= 1e-9:
                cap_reasons.append("POSITION_CAP_EXCEEDED")
            if sector_headroom <= 1e-9:
                cap_reasons.append("SECTOR_GROSS_CAP_EXCEEDED")
            if gross_headroom <= 1e-9:
                cap_reasons.append("GROSS_EXPOSURE_CAP_EXCEEDED")
            if net_headroom <= 1e-9:
                cap_reasons.append("NET_EXPOSURE_BOUND_EXCEEDED")
            return RiskVerdict(
                approved=False,
                order=order,
                adjusted_qty=0.0,
                violations=cap_reasons or ["EXPOSURE_CAP_EXCEEDED"],
            )

        raw_allowed_qty = allowed_notional / sizing_px
        if is_short_order:
            adjusted_qty = float(math.floor(raw_allowed_qty + 1e-12))
            if adjusted_qty < 1.0:
                return RiskVerdict(
                    approved=False,
                    order=order,
                    adjusted_qty=0.0,
                    violations=["FRACTIONAL_SHORT_REJECTED"],
                )
        elif self.allow_fractional_longs:
            adjusted_qty = _floor_decimals(raw_allowed_qty, 6)
            if adjusted_qty <= 1e-6:
                return RiskVerdict(
                    approved=False,
                    order=order,
                    adjusted_qty=0.0,
                    violations=["ORDER_QTY_BELOW_MINIMUM"],
                )
        else:
            adjusted_qty = float(math.floor(raw_allowed_qty + 1e-12))
            if adjusted_qty < 1.0:
                return RiskVerdict(
                    approved=False,
                    order=order,
                    adjusted_qty=0.0,
                    violations=["ORDER_QTY_BELOW_ONE_SHARE"],
                )

        # Final post-rounding safety check so rounding never exceeds any hard ceiling
        if adjusted_qty * sizing_px > allowed_notional + 1e-6:
            if is_short_order or not self.allow_fractional_longs:
                adjusted_qty = max(0.0, adjusted_qty - 1.0)
            else:
                adjusted_qty = _floor_decimals(allowed_notional / sizing_px, 6)
            if adjusted_qty <= (0.5 if is_short_order else 1e-6):
                return RiskVerdict(
                    approved=False,
                    order=order,
                    adjusted_qty=0.0,
                    violations=["POST_ROUNDING_EXPOSURE_EXCEEDED"],
                )

        return RiskVerdict(
            approved=True,
            order=order,
            adjusted_qty=adjusted_qty,
            stop_loss_price=stop_price,
            violations=[],
        )

    def _resolve_stop_loss(
        self,
        order: ProposedOrder,
        anchor_px: float,
        is_short: bool,
        signal: SignalRow | None,
    ) -> float | None:
        if not math.isfinite(anchor_px) or anchor_px <= 0.0:
            return None

        has_valid_atr = (
            signal is not None
            and math.isfinite(signal.atr_pct)
            and signal.atr_pct > 0.0
        )

        if order.stop_loss_price is not None:
            sp = order.stop_loss_price
            if not math.isfinite(sp) or sp <= 0.0:
                return None
            if is_short and sp <= anchor_px:
                return None
            if not is_short and sp >= anchor_px:
                return None

            if has_valid_atr:
                assert signal is not None
                if is_short:
                    max_stop = anchor_px * (
                        1.0 + self.config.stop_loss_atr_mult_short * signal.atr_pct
                    )
                    candidate = min(sp, max_stop)
                else:
                    min_stop = anchor_px * max(
                        0.05, 1.0 - self.config.stop_loss_atr_mult_long * signal.atr_pct
                    )
                    candidate = max(sp, min_stop)
            else:
                if is_short:
                    return None
                # Clamp long stop to hard max fallback width when signal is absent
                min_stop = anchor_px * (1.0 - MAX_FALLBACK_LONG_STOP_PCT)
                candidate = max(sp, min_stop)
        else:
            if not has_valid_atr:
                return None
            assert signal is not None
            if is_short:
                candidate = anchor_px * (
                    1.0 + self.config.stop_loss_atr_mult_short * signal.atr_pct
                )
            else:
                candidate = anchor_px * max(
                    0.05, 1.0 - self.config.stop_loss_atr_mult_long * signal.atr_pct
                )

        rounded = round(candidate, 4)
        # Re-verify strict positivity and directional inequality AFTER rounding
        if not math.isfinite(rounded) or rounded <= 0.0:
            return None
        if is_short and rounded <= anchor_px:
            return None
        if not is_short and rounded >= anchor_px:
            return None
        return rounded
