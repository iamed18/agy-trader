from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime

from agy_trader.core.config import RiskConfig, StrategyConfig
from agy_trader.core.models import (
    AssetRiskMetadata,
    ControlDials,
    OrderSide,
    PositionSide,
    ProposedOrder,
    SignalRow,
    ensure_utc,
)


class PortfolioConstructor:
    """Inverse-volatility (ATR) Long/Short target weight builder and order proposer.

    Emits strictly separated close + open orders on Long<->Short flips so risk-reducing
    exits can execute cleanly even if the subsequent risk-increasing entry is blocked.
    """

    def __init__(self, strategy_config: StrategyConfig, risk_config: RiskConfig) -> None:
        self.strategy = strategy_config
        self.risk = risk_config

    def compute_target_weights(
        self,
        signals: Sequence[SignalRow],
        dials: ControlDials,
        as_of: datetime,
        universe_meta: Mapping[str, AssetRiskMetadata],
        current_weights: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute signed target portfolio weights (+long, -short) per symbol."""
        as_of_utc = ensure_utc(as_of)
        blacklist = dials.active_blacklist(as_of_utc)
        cur_w = dict(current_weights or {})

        if not signals:
            # No fresh signals: respect blacklist, allow_shorting=False, and exposure_multiplier
            targets_empty: dict[str, float] = {}
            for sym, w in cur_w.items():
                if sym in blacklist or (w < 0.0 and not dials.allow_shorting):
                    targets_empty[sym] = 0.0
                else:
                    targets_empty[sym] = w
            cur_gross = sum(abs(w) for w in targets_empty.values())
            max_allowed = self.risk.max_gross_exposure * dials.exposure_multiplier
            if cur_gross > max_allowed and cur_gross > 1e-12:
                scale_down = max_allowed / cur_gross
                targets_empty = {sym: w * scale_down for sym, w in targets_empty.items()}
            return targets_empty

        # Deduplicate signals by symbol (keep latest as_of <= as_of_utc)
        deduped_map: dict[str, SignalRow] = {}
        for s in signals:
            if s.as_of > as_of_utc:
                continue
            prev = deduped_map.get(s.symbol)
            if prev is None or s.as_of >= prev.as_of:
                deduped_map[s.symbol] = s
        deduped = list(deduped_map.values())
        if not deduped:
            return self.compute_target_weights(
                signals=(),
                dials=dials,
                as_of=as_of_utc,
                universe_meta=universe_meta,
                current_weights=cur_w,
            )

        macro_scale = min(s.macro_regime_scale for s in deduped)
        effective_gross_budget = (
            self.risk.max_gross_exposure * macro_scale * dials.exposure_multiplier
        )
        effective_gross_budget = max(0.0, min(self.risk.max_gross_exposure, effective_gross_budget))

        # Eligible long candidates
        long_cands = [
            s
            for s in deduped
            if s.symbol not in blacklist
            and s.symbol in universe_meta
            and s.composite_score >= self.strategy.min_signal_threshold
        ]
        long_cands.sort(key=lambda s: (-s.composite_score, s.symbol))
        selected_longs = long_cands[: self.strategy.top_k_long]

        # Eligible short candidates (pre-filter squeeze guards so slots aren't wasted)
        selected_shorts: list[SignalRow] = []
        if (
            dials.allow_shorting
            and self.strategy.top_k_short > 0
            and self.risk.max_short_position_pct > 0.0
        ):
            short_cands = [
                s
                for s in deduped
                if s.symbol not in blacklist
                and s.symbol in universe_meta
                and universe_meta[s.symbol].shortable
                and universe_meta[s.symbol].easy_to_borrow
                and s.ret_5d < self.risk.short_squeeze_5d_return_cap
                and s.ann_vol < self.risk.short_squeeze_ann_vol_cap
                and s.composite_score <= -self.strategy.min_signal_threshold
            ]
            short_cands.sort(key=lambda s: (s.composite_score, s.symbol))
            selected_shorts = short_cands[: self.strategy.top_k_short]

        targets: dict[str, float] = {}
        for sym in set(cur_w.keys()) | {s.symbol for s in deduped}:
            targets[sym] = 0.0

        # Allocate gross budget across selected longs and shorts proportional to inverse ATR
        raw_inv_vol: dict[str, float] = {}
        for s in selected_longs:
            raw_inv_vol[s.symbol] = 1.0 / max(s.atr_pct, 0.005)
        for s in selected_shorts:
            cap_ratio = self.risk.max_short_position_pct / max(self.risk.max_long_position_pct, 1e-6)
            raw_inv_vol[s.symbol] = cap_ratio / max(s.atr_pct, 0.005)

        total_inv_vol = sum(raw_inv_vol.values())
        if total_inv_vol > 1e-12 and effective_gross_budget > 0.0:
            for s in selected_longs:
                w = effective_gross_budget * (raw_inv_vol[s.symbol] / total_inv_vol)
                targets[s.symbol] = min(w, self.risk.max_long_position_pct)
            for s in selected_shorts:
                w = effective_gross_budget * (raw_inv_vol[s.symbol] / total_inv_vol)
                targets[s.symbol] = -min(w, self.risk.max_short_position_pct)

        # Enforce sector gross concentration cap smoothly on target weights
        sector_totals: dict[str, float] = {}
        for sym, w in targets.items():
            if abs(w) <= 1e-12 or sym not in universe_meta:
                continue
            sec = universe_meta[sym].sector
            sector_totals[sec] = sector_totals.get(sec, 0.0) + abs(w)
        for sec, sec_gross in sector_totals.items():
            if sec_gross > self.risk.max_sector_gross_pct and sec_gross > 1e-12:
                sec_scale = self.risk.max_sector_gross_pct / sec_gross
                for sym in list(targets.keys()):
                    if sym in universe_meta and universe_meta[sym].sector == sec:
                        targets[sym] *= sec_scale

        # Enforce net exposure bounds smoothly by scaling the dominant side
        long_sum = sum(w for w in targets.values() if w > 0.0)
        short_sum = sum(-w for w in targets.values() if w < 0.0)
        net_exp = long_sum - short_sum
        if net_exp > self.risk.max_net_exposure and long_sum > 1e-12:
            target_long_sum = max(0.0, short_sum + self.risk.max_net_exposure)
            l_scale = target_long_sum / long_sum
            for sym, w in list(targets.items()):
                if w > 0.0:
                    targets[sym] = w * l_scale
        elif net_exp < self.risk.min_net_exposure and short_sum > 1e-12:
            target_short_sum = max(0.0, long_sum - self.risk.min_net_exposure)
            s_scale = target_short_sum / short_sum
            for sym, w in list(targets.items()):
                if w < 0.0:
                    targets[sym] = w * s_scale

        # If ops paused new entries, clamp target magnitude so we never open or increase risk
        if dials.pause_new_entries:
            for sym, tgt in list(targets.items()):
                cw = cur_w.get(sym, 0.0)
                if sym in blacklist or cw == 0.0:
                    targets[sym] = 0.0
                elif cw > 0.0:
                    targets[sym] = max(0.0, min(cw, tgt))
                else:
                    targets[sym] = min(0.0, max(cw, tgt))

        return targets

    def build_proposed_orders(
        self,
        signals: Sequence[SignalRow],
        current_positions: Mapping[str, float],
        prices: Mapping[str, float],
        equity: float,
        dials: ControlDials,
        as_of: datetime,
        universe_meta: Mapping[str, AssetRiskMetadata],
    ) -> list[ProposedOrder]:
        """Convert signals and current signed share positions into ProposedOrders."""
        if not math.isfinite(equity) or equity <= 0.0:
            return []

        as_of_utc = ensure_utc(as_of)
        signal_map: dict[str, SignalRow] = {}
        for s in signals:
            if s.as_of <= as_of_utc:
                prev = signal_map.get(s.symbol)
                if prev is None or s.as_of >= prev.as_of:
                    signal_map[s.symbol] = s

        current_weights: dict[str, float] = {}
        for sym, qty in current_positions.items():
            px = prices.get(sym, 0.0)
            if px > 0.0 and math.isfinite(px):
                current_weights[sym] = (qty * px) / equity

        target_weights = self.compute_target_weights(
            signals=list(signal_map.values()),
            dials=dials,
            as_of=as_of_utc,
            universe_meta=universe_meta,
            current_weights=current_weights,
        )

        orders: list[ProposedOrder] = []
        all_symbols = sorted(set(current_positions.keys()) | set(target_weights.keys()))
        deadband = self.strategy.rebalance_deadband_pct

        for sym in all_symbols:
            px = prices.get(sym, 0.0)
            if not math.isfinite(px) or px <= 0.0:
                continue

            cur_qty = float(current_positions.get(sym, 0.0))
            cur_w = (cur_qty * px) / equity
            tgt_w = float(target_weights.get(sym, 0.0))

            # Skip small adjustments inside deadband unless closing fully or flipping sides
            is_full_close = abs(tgt_w) <= 1e-9 and abs(cur_qty) > 1e-9
            is_side_flip = (cur_qty > 1e-9 and tgt_w < -1e-9) or (
                cur_qty < -1e-9 and tgt_w > 1e-9
            )
            if not is_full_close and not is_side_flip and abs(tgt_w - cur_w) < deadband:
                continue

            sig = signal_map.get(sym)
            atr_pct = sig.atr_pct if sig is not None else 0.02

            # Case 1: Sign flip -> emit separate risk-reducing close + risk-increasing open
            if is_side_flip:
                close_side = OrderSide.sell if cur_qty > 0.0 else OrderSide.buy
                orders.append(
                    ProposedOrder(
                        symbol=sym,
                        side=close_side,
                        qty=abs(cur_qty),
                        target_side=None,
                        is_risk_reducing=True,
                        reference_price=px,
                        stop_loss_price=None,
                        reason="signal_flip_close",
                    )
                )
                self._append_entry_order(
                    orders=orders,
                    symbol=sym,
                    target_weight=tgt_w,
                    equity=equity,
                    px=px,
                    atr_pct=atr_pct,
                    reason="signal_flip_open",
                )
                continue

            # Case 2: Full close to zero
            if is_full_close:
                close_side = OrderSide.sell if cur_qty > 0.0 else OrderSide.buy
                orders.append(
                    ProposedOrder(
                        symbol=sym,
                        side=close_side,
                        qty=abs(cur_qty),
                        target_side=None,
                        is_risk_reducing=True,
                        reference_price=px,
                        stop_loss_price=None,
                        reason="exit_to_flat",
                    )
                )
                continue

            # Case 3: Same-side adjustment or fresh open from flat
            raw_tgt_qty = (tgt_w * equity) / px
            if tgt_w < 0.0:
                # Alpaca requires whole shares for short positions (epsilon avoids float-floor jitter)
                tgt_qty = -float(math.floor(abs(raw_tgt_qty) + 1e-12))
            elif self.strategy.allow_fractional_longs:
                tgt_qty = round(raw_tgt_qty, 6)
            else:
                tgt_qty = float(math.floor(raw_tgt_qty + 1e-12))

            delta = tgt_qty - cur_qty
            if abs(delta) <= 1e-6:
                continue

            side = OrderSide.buy if delta > 0.0 else OrderSide.sell
            is_reducing = abs(tgt_qty) < abs(cur_qty) and (
                tgt_qty == 0.0 or math.copysign(1.0, tgt_qty) == math.copysign(1.0, cur_qty)
            )

            stop_price: float | None = None
            target_pos_side: PositionSide | None = None
            if not is_reducing:
                if tgt_qty > 0.0:
                    target_pos_side = PositionSide.long
                    stop_price = round(
                        px * max(0.05, 1.0 - self.risk.stop_loss_atr_mult_long * atr_pct), 4
                    )
                elif tgt_qty < 0.0:
                    target_pos_side = PositionSide.short
                    stop_price = round(
                        px * (1.0 + self.risk.stop_loss_atr_mult_short * atr_pct), 4
                    )
            else:
                if tgt_qty > 0.0:
                    target_pos_side = PositionSide.long
                elif tgt_qty < 0.0:
                    target_pos_side = PositionSide.short

            orders.append(
                ProposedOrder(
                    symbol=sym,
                    side=side,
                    qty=abs(delta),
                    target_side=target_pos_side,
                    is_risk_reducing=is_reducing,
                    reference_price=px,
                    stop_loss_price=stop_price,
                    reason="rebalance_reduce" if is_reducing else "rebalance_open_or_add",
                )
            )

        return orders

    def _append_entry_order(
        self,
        orders: list[ProposedOrder],
        symbol: str,
        target_weight: float,
        equity: float,
        px: float,
        atr_pct: float,
        reason: str,
    ) -> None:
        raw_qty = abs(target_weight * equity) / px
        if target_weight < 0.0:
            qty = float(math.floor(raw_qty + 1e-12))
            if qty < 1.0:
                return
            stop_price = round(px * (1.0 + self.risk.stop_loss_atr_mult_short * atr_pct), 4)
            orders.append(
                ProposedOrder(
                    symbol=symbol,
                    side=OrderSide.sell,
                    qty=qty,
                    target_side=PositionSide.short,
                    is_risk_reducing=False,
                    reference_price=px,
                    stop_loss_price=stop_price,
                    reason=reason,
                )
            )
        elif target_weight > 0.0:
            qty = (
                round(raw_qty, 6)
                if self.strategy.allow_fractional_longs
                else float(math.floor(raw_qty + 1e-12))
            )
            if qty <= 1e-6:
                return
            stop_price = round(px * max(0.05, 1.0 - self.risk.stop_loss_atr_mult_long * atr_pct), 4)
            orders.append(
                ProposedOrder(
                    symbol=symbol,
                    side=OrderSide.buy,
                    qty=qty,
                    target_side=PositionSide.long,
                    is_risk_reducing=False,
                    reference_price=px,
                    stop_loss_price=stop_price,
                    reason=reason,
                )
            )
