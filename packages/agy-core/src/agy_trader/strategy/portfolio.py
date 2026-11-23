from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime

import numpy as np

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
from agy_trader.strategy.features import SAFE_HAVEN_SECTORS, SAFE_HAVEN_SYMBOLS


def _is_safe_haven(sym: str, universe_meta: Mapping[str, AssetRiskMetadata]) -> bool:
    meta = universe_meta.get(sym)
    return (meta is not None and meta.sector in SAFE_HAVEN_SECTORS) or (sym in SAFE_HAVEN_SYMBOLS)


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
        frozen_symbols: set[str] | frozenset[str] = frozenset(),
    ) -> dict[str, float]:
        """Compute signed target portfolio weights (+long, -short) per symbol."""
        as_of_utc = ensure_utc(as_of)
        blacklist = dials.active_blacklist(as_of_utc)
        frozen_set = frozenset(s.strip().upper() for s in frozen_symbols if s.strip())
        cur_w = dict(current_weights or {})

        if not signals:
            # No fresh signals: lock frozen positions, respect blacklist, allow_shorting, and multiplier
            targets_empty: dict[str, float] = {}
            for sym, w in cur_w.items():
                if sym in frozen_set:
                    targets_empty[sym] = w
                elif sym in blacklist or (w < 0.0 and not dials.allow_shorting):
                    targets_empty[sym] = 0.0
                else:
                    targets_empty[sym] = w
            frozen_gross = sum(
                abs(w) for sym, w in targets_empty.items() if sym in frozen_set
            )
            non_frozen_gross = sum(
                abs(w) for sym, w in targets_empty.items() if sym not in frozen_set
            )
            max_allowed = self.risk.max_gross_exposure * dials.exposure_multiplier
            if frozen_gross + non_frozen_gross > max_allowed and non_frozen_gross > 1e-12:
                remaining_budget = max(0.0, max_allowed - frozen_gross)
                scale_down = remaining_budget / non_frozen_gross
                targets_empty = {
                    sym: (w if sym in frozen_set else w * scale_down)
                    for sym, w in targets_empty.items()
                }
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
                frozen_symbols=frozen_set,
            )

        macro_scale = min(s.macro_regime_scale for s in deduped)
        risk_scale = max(0.0, min(1.0, macro_scale * dials.exposure_multiplier))
        effective_gross_budget = self.risk.max_gross_exposure * risk_scale

        exit_thresh = (
            self.strategy.exit_signal_threshold
            if self.strategy.exit_signal_threshold is not None
            else self.strategy.min_signal_threshold
        )

        def _long_abs_ok(s: SignalRow) -> bool:
            if not self.strategy.abs_momentum_filter or s.abs_trend_raw is None:
                return True
            return s.abs_trend_raw > 0.0

        def _short_abs_ok(s: SignalRow) -> bool:
            if not self.strategy.abs_momentum_filter or s.abs_trend_raw is None:
                return True
            return s.abs_trend_raw < 0.0

        def _is_held_long(s: SignalRow) -> bool:
            return cur_w.get(s.symbol, 0.0) > 1e-9

        def _is_held_short(s: SignalRow) -> bool:
            return cur_w.get(s.symbol, 0.0) < -1e-9

        def _corr_ok(cand: SignalRow, chosen: list[SignalRow]) -> bool:
            max_corr = self.strategy.max_pair_corr_60d
            if max_corr is None or not chosen:
                return True
            return not any(
                cand.peer_corr_60d.get(q.symbol, q.peer_corr_60d.get(cand.symbol, 0.0)) > max_corr
                for q in chosen
            )

        def _select_with_hysteresis(
            cands: list[SignalRow],
            top_k: int,
            is_held: Callable[[SignalRow], bool],
            sort_key: Callable[[SignalRow], tuple[float, str]],
        ) -> list[SignalRow]:
            cands.sort(key=sort_key)
            buffered = cands[: top_k + self.strategy.hold_rank_buffer]
            selected = [s for s in buffered if is_held(s)][:top_k]
            selected_syms = {s.symbol for s in selected}
            for s in cands:
                if len(selected) >= top_k:
                    break
                if s.symbol not in selected_syms and _corr_ok(s, selected):
                    selected.append(s)
                    selected_syms.add(s.symbol)
            selected.sort(key=sort_key)
            return selected

        # Eligible long candidates with hysteresis, Stage-1 finalist gate, & optional dual-momentum filter
        long_cands = [
            s
            for s in deduped
            if s.symbol not in frozen_set
            and s.symbol not in blacklist
            and s.symbol in universe_meta
            and s.stage1_finalist
            and _long_abs_ok(s)
            and s.composite_score
            >= (exit_thresh if _is_held_long(s) else self.strategy.min_signal_threshold)
        ]
        selected_longs = _select_with_hysteresis(
            long_cands,
            top_k=self.strategy.top_k_long,
            is_held=_is_held_long,
            sort_key=lambda s: (-s.composite_score, s.symbol),
        )
        selected_long_syms = {s.symbol for s in selected_longs}

        # Eligible short candidates (pre-filter squeeze guards + optional bear-regime gate)
        selected_shorts: list[SignalRow] = []
        shorts_regime_ok = (macro_scale < 1.0) if self.strategy.short_only_in_bear_regime else True
        if (
            dials.allow_shorting
            and self.strategy.top_k_short > 0
            and self.risk.max_short_position_pct > 0.0
            and shorts_regime_ok
        ):
            short_cands = [
                s
                for s in deduped
                if s.symbol not in frozen_set
                and s.symbol not in blacklist
                and s.symbol not in selected_long_syms
                and s.symbol in universe_meta
                and universe_meta[s.symbol].shortable
                and universe_meta[s.symbol].easy_to_borrow
                and s.ret_5d < self.risk.short_squeeze_5d_return_cap
                and s.ann_vol < self.risk.short_squeeze_ann_vol_cap
                and _short_abs_ok(s)
                and s.composite_score
                <= (-exit_thresh if _is_held_short(s) else -self.strategy.min_signal_threshold)
            ]
            selected_shorts = _select_with_hysteresis(
                short_cands,
                top_k=self.strategy.top_k_short,
                is_held=_is_held_short,
                sort_key=lambda s: (s.composite_score, s.symbol),
            )

        targets: dict[str, float] = {}
        for sym in set(cur_w.keys()) | {s.symbol for s in deduped}:
            targets[sym] = cur_w.get(sym, 0.0) if sym in frozen_set else 0.0

        use_multi_sleeve = (
            self.strategy.waterfill_weights
            or self.strategy.exempt_safe_havens_from_macro_scale
            or self.strategy.safe_haven_rotation_fraction > 0.0
        )

        if use_multi_sleeve:
            self._allocate_multi_sleeve_weights(
                targets=targets,
                selected_longs=selected_longs,
                selected_shorts=selected_shorts,
                deduped_map=deduped_map,
                blacklist=blacklist,
                frozen_set=frozen_set,
                dials=dials,
                macro_scale=macro_scale,
                risk_scale=risk_scale,
                universe_meta=universe_meta,
            )
        else:
            # Allocate remaining gross budget across non-frozen selected longs and shorts
            frozen_gross = sum(abs(w) for sym, w in targets.items() if sym in frozen_set)
            avail_gross_budget = max(0.0, effective_gross_budget - frozen_gross)
            raw_inv_vol: dict[str, float] = {}
            for s in selected_longs:
                raw_inv_vol[s.symbol] = self._raw_long_base_weight(s, selected_longs)
            for s in selected_shorts:
                cap_ratio = self.risk.max_short_position_pct / max(
                    self.risk.max_long_position_pct, 1e-6
                )
                raw_inv_vol[s.symbol] = cap_ratio / max(s.atr_pct, 0.005)

            total_inv_vol = sum(raw_inv_vol.values())
            if total_inv_vol > 1e-12 and avail_gross_budget > 0.0:
                for s in selected_longs:
                    w = avail_gross_budget * (raw_inv_vol[s.symbol] / total_inv_vol)
                    targets[s.symbol] = min(w, self.risk.max_long_position_pct)
                for s in selected_shorts:
                    w = avail_gross_budget * (raw_inv_vol[s.symbol] / total_inv_vol)
                    targets[s.symbol] = -min(w, self.risk.max_short_position_pct)

        return self.apply_portfolio_overlays(
            targets,
            signals=deduped,
            dials=dials,
            universe_meta=universe_meta,
            current_weights=cur_w,
            frozen_set=frozen_set,
            as_of=as_of_utc,
        )

    def apply_portfolio_overlays(
        self,
        targets: dict[str, float],
        *,
        signals: Sequence[SignalRow],
        dials: ControlDials,
        universe_meta: Mapping[str, AssetRiskMetadata],
        current_weights: Mapping[str, float] = {},
        frozen_set: frozenset[str] = frozenset(),
        as_of: datetime | None = None,
    ) -> dict[str, float]:
        """Enforce sector cap, macro/gross/net exposure ceilings, PSD vol-target, and pause_new_entries."""
        normalized: dict[str, float] = {}
        for raw_sym, raw_w in targets.items():
            sym = str(raw_sym).strip().upper()
            if not sym:
                continue
            val = float(raw_w)
            if not math.isfinite(val):
                raise ValueError(f"Non-finite target weight for {sym}: {raw_w!r}")
            normalized[sym] = val
        targets = normalized

        cur_w: dict[str, float] = {}
        for raw_sym, raw_val in current_weights.items():
            sym = str(raw_sym).strip().upper()
            if sym and math.isfinite(float(raw_val)):
                cur_w[sym] = float(raw_val)

        as_of_utc = (
            ensure_utc(as_of)
            if as_of is not None
            else (
                max((ensure_utc(s.as_of) for s in signals), default=ensure_utc(dials.updated_at))
            )
        )
        blacklist = dials.active_blacklist(as_of_utc)
        deduped_map: dict[str, SignalRow] = {}
        for s in signals:
            if s.as_of <= as_of_utc:
                prev = deduped_map.get(s.symbol)
                if prev is None or s.as_of >= prev.as_of:
                    deduped_map[s.symbol] = s

        for sym in list(targets.keys()):
            if sym not in frozen_set and (
                sym in blacklist or (targets[sym] < 0.0 and not dials.allow_shorting)
            ):
                targets[sym] = 0.0

        use_multi_sleeve = (
            self.strategy.waterfill_weights
            or self.strategy.exempt_safe_havens_from_macro_scale
            or self.strategy.safe_haven_rotation_fraction > 0.0
        )
        macro_scale = min((s.macro_regime_scale for s in deduped_map.values()), default=1.0)
        risk_scale = max(0.0, min(1.0, macro_scale * dials.exposure_multiplier))

        # 1. Sector gross exposure cap
        self._apply_sector_gross_cap(targets, universe_meta, frozen_set=frozen_set)

        # 2. Portfolio gross exposure cap & net exposure bounds
        gross_cap = (
            self.risk.max_gross_exposure * dials.exposure_multiplier
            if use_multi_sleeve
            else self.risk.max_gross_exposure * risk_scale
        )
        frozen_gross = sum(abs(w) for sym, w in targets.items() if sym in frozen_set)
        non_frozen_gross = sum(abs(w) for sym, w in targets.items() if sym not in frozen_set)
        if frozen_gross + non_frozen_gross > gross_cap and non_frozen_gross > 1e-12:
            gs = max(0.0, gross_cap - frozen_gross) / non_frozen_gross
            for sym in list(targets.keys()):
                if sym not in frozen_set:
                    targets[sym] *= gs

        net_scale = dials.exposure_multiplier if use_multi_sleeve else risk_scale
        effective_max_net = min(
            self.risk.max_net_exposure,
            self.risk.max_net_exposure * net_scale,
        )
        effective_min_net = max(
            self.risk.min_net_exposure,
            self.risk.min_net_exposure * net_scale,
        )
        self._apply_net_exposure_bounds(
            targets, effective_max_net, effective_min_net, frozen_set=frozen_set
        )

        # 3. Portfolio-level covariance volatility targeting (with PSD eigenvalue floor)
        if self.strategy.portfolio_vol_target_ann is not None:
            active_syms = [
                sym
                for sym, w in targets.items()
                if sym not in frozen_set and abs(w) > 1e-9
            ]
            if active_syms:
                n_act = len(active_syms)
                w_vol_vec = np.zeros(n_act, dtype=np.float64)
                for i, sym_i in enumerate(active_syms):
                    sig_i = deduped_map.get(sym_i)
                    raw_vol = sig_i.ann_vol if sig_i is not None else 0.20
                    vol_i = (
                        max(float(raw_vol), 0.05)
                        if math.isfinite(float(raw_vol))
                        else 0.20
                    )
                    w_vol_vec[i] = targets[sym_i] * vol_i

                diag_var = float(np.dot(w_vol_vec, w_vol_vec))
                corr_mat = np.eye(n_act, dtype=np.float64)
                for i in range(n_act):
                    sym_i = active_syms[i]
                    sig_i = deduped_map.get(sym_i)
                    for j in range(i + 1, n_act):
                        sym_j = active_syms[j]
                        sig_j = deduped_map.get(sym_j)
                        c_ij = (
                            sig_i.peer_corr_60d.get(sym_j)
                            if sig_i is not None
                            else None
                        )
                        c_ji = (
                            sig_j.peer_corr_60d.get(sym_i)
                            if sig_j is not None
                            else None
                        )
                        if c_ij is not None and math.isfinite(float(c_ij)):
                            raw_c = float(c_ij)
                        elif c_ji is not None and math.isfinite(float(c_ji)):
                            raw_c = float(c_ji)
                        else:
                            raw_c = 0.25
                        clamped_c = max(-0.99, min(0.99, raw_c))
                        corr_mat[i, j] = clamped_c
                        corr_mat[j, i] = clamped_c

                try:
                    eigvals, eigvecs = np.linalg.eigh(corr_mat)
                    psd_corr = (
                        eigvecs
                        @ np.diag(np.maximum(eigvals, 0.05))
                        @ eigvecs.T
                    )
                    raw_port_var = float(w_vol_vec @ psd_corr @ w_vol_vec)
                except np.linalg.LinAlgError:
                    raw_port_var = diag_var

                min_var_floor = diag_var * 0.25 if n_act > 1 else diag_var
                port_var = (
                    max(raw_port_var, min_var_floor, 1e-8)
                    if math.isfinite(raw_port_var)
                    else max(diag_var, 1e-8)
                )
                port_vol = math.sqrt(port_var)
                vol_scale = min(
                    1.0, self.strategy.portfolio_vol_target_ann / max(port_vol, 0.05)
                )
                if vol_scale < 1.0:
                    for sym in list(targets.keys()):
                        if sym not in frozen_set:
                            targets[sym] *= vol_scale

        # 4. If ops paused new entries, clamp target magnitude so we never open or increase risk
        if dials.pause_new_entries:
            for sym, tgt in list(targets.items()):
                if sym in frozen_set:
                    continue
                cw = cur_w.get(sym, 0.0)
                if sym in blacklist or cw == 0.0:
                    targets[sym] = 0.0
                elif cw > 0.0:
                    targets[sym] = max(0.0, min(cw, tgt))
                else:
                    targets[sym] = min(0.0, max(cw, tgt))

        return targets

    def _raw_long_base_weight(
        self,
        s: SignalRow,
        selected_longs: Sequence[SignalRow],
    ) -> float:
        mode = self.strategy.vol_sizing_mode
        if mode == "inv_atr":
            base = 1.0 / max(s.atr_pct, 0.005)
        elif mode == "capped_inv_vol":
            clipped = max(
                self.strategy.vol_sizing_floor,
                min(self.strategy.vol_sizing_cap, max(s.ann_vol, 0.05)),
            )
            base = 1.0 / clipped
        else:
            base = 1.0

        if self.strategy.score_tilt_power > 0.0:
            base *= max(0.25, s.composite_score + 0.5) ** self.strategy.score_tilt_power
        if self.strategy.corr_penalty_eta > 0.0 and len(selected_longs) >= 2:
            others = [o.symbol for o in selected_longs if o.symbol != s.symbol]
            mc = sum(max(0.0, s.peer_corr_60d.get(o, 0.0)) for o in others) / len(others)
            base /= 1.0 + self.strategy.corr_penalty_eta * mc
        return base

    def _allocate_multi_sleeve_weights(
        self,
        *,
        targets: dict[str, float],
        selected_longs: list[SignalRow],
        selected_shorts: list[SignalRow],
        deduped_map: Mapping[str, SignalRow],
        blacklist: set[str],
        frozen_set: frozenset[str],
        dials: ControlDials,
        macro_scale: float,
        risk_scale: float,
        universe_meta: Mapping[str, AssetRiskMetadata],
    ) -> None:
        frozen_long_used = sum(
            w for sym, w in targets.items() if sym in frozen_set and w > 0.0
        )
        frozen_gross_used = sum(
            abs(w) for sym, w in targets.items() if sym in frozen_set
        )
        max_total_long = max(
            0.0,
            min(self.risk.max_net_exposure, self.risk.max_gross_exposure)
            * dials.exposure_multiplier
            - frozen_long_used,
        )
        short_syms = {s.symbol for s in selected_shorts}
        sh_candidates = [
            sym
            for sym in ("GLD", "TLT")
            if sym not in frozen_set
            and sym not in blacklist
            and sym not in short_syms
            and sym in universe_meta
            and sym in deduped_map
            and deduped_map[sym].safe_haven_trend_ok
        ]

        if selected_longs:
            raw_w: dict[str, float] = {
                s.symbol: self._raw_long_base_weight(s, selected_longs)
                for s in selected_longs
            }

            rem = max_total_long
            unalloc = [s.symbol for s in selected_longs]
            assigned = {sym: 0.0 for sym in unalloc}
            for _ in range(len(unalloc)):
                if not unalloc or rem <= 1e-9:
                    break
                sub_sum = sum(raw_w[u] for u in unalloc)
                if sub_sum <= 1e-12:
                    break
                capped = False
                for u in list(unalloc):
                    cw = rem * (raw_w[u] / sub_sum)
                    if cw >= self.risk.max_long_position_pct:
                        assigned[u] = self.risk.max_long_position_pct
                        rem -= self.risk.max_long_position_pct
                        unalloc.remove(u)
                        capped = True
                if not capped:
                    for u in unalloc:
                        assigned[u] = rem * (raw_w[u] / sub_sum)
                    break

            for sym, w_val in assigned.items():
                eff_s = (
                    1.0
                    if (
                        self.strategy.exempt_safe_havens_from_macro_scale
                        and _is_safe_haven(sym, universe_meta)
                    )
                    else macro_scale
                )
                targets[sym] = min(w_val * eff_s, self.risk.max_long_position_pct)

        used_non_frozen_long = sum(
            w for sym, w in targets.items() if sym not in frozen_set and w > 0.0
        )
        freed_budget = max(0.0, max_total_long - used_non_frozen_long)
        if sh_candidates and freed_budget > 0.02 and self.strategy.safe_haven_rotation_fraction > 0.0:
            rotate_budget = min(
                freed_budget * self.strategy.safe_haven_rotation_fraction,
                freed_budget,
            )
            if rotate_budget > 0.01:
                sh_raw = {
                    sym: (1.0 / max(deduped_map[sym].ann_vol, 0.05))
                    * max(0.3, deduped_map[sym].safe_haven_strength)
                    for sym in sh_candidates
                }
                tot_sh = sum(sh_raw.values())
                if tot_sh > 1e-12:
                    for sym in sh_candidates:
                        targets[sym] = min(
                            self.risk.max_long_position_pct,
                            targets.get(sym, 0.0) + rotate_budget * (sh_raw[sym] / tot_sh),
                        )

        if selected_shorts:
            short_inv_vol = {s.symbol: 1.0 / max(s.atr_pct, 0.005) for s in selected_shorts}
            tot_short_iv = sum(short_inv_vol.values())
            non_frozen_long_used = sum(
                w for sym, w in targets.items() if sym not in frozen_set and w > 0.0
            )
            gross_avail = max(
                0.0,
                self.risk.max_gross_exposure * risk_scale
                - non_frozen_long_used
                - frozen_gross_used,
            )
            short_budget = min(
                gross_avail,
                len(selected_shorts) * self.risk.max_short_position_pct,
            )
            if tot_short_iv > 1e-12 and short_budget > 0.0:
                for s in selected_shorts:
                    w = short_budget * (short_inv_vol[s.symbol] / tot_short_iv)
                    targets[s.symbol] = -min(w, self.risk.max_short_position_pct)

    def _apply_sector_gross_cap(
        self,
        targets: dict[str, float],
        universe_meta: Mapping[str, AssetRiskMetadata],
        *,
        frozen_set: frozenset[str] = frozenset(),
    ) -> None:
        sector_totals: dict[str, float] = {}
        sector_frozen_totals: dict[str, float] = {}
        for sym, w in targets.items():
            if abs(w) <= 1e-12 or sym not in universe_meta:
                continue
            sec = universe_meta[sym].sector
            sector_totals[sec] = sector_totals.get(sec, 0.0) + abs(w)
            if sym in frozen_set:
                sector_frozen_totals[sec] = sector_frozen_totals.get(sec, 0.0) + abs(w)
        for sec, sec_gross in sector_totals.items():
            if sec_gross > self.risk.max_sector_gross_pct and sec_gross > 1e-12:
                frozen_sec = sector_frozen_totals.get(sec, 0.0)
                non_frozen_sec = sec_gross - frozen_sec
                if non_frozen_sec <= 1e-12:
                    continue
                sec_scale = max(0.0, self.risk.max_sector_gross_pct - frozen_sec) / non_frozen_sec
                for sym in list(targets.keys()):
                    if (
                        sym not in frozen_set
                        and sym in universe_meta
                        and universe_meta[sym].sector == sec
                    ):
                        targets[sym] *= sec_scale

    @staticmethod
    def _apply_net_exposure_bounds(
        targets: dict[str, float],
        effective_max_net: float,
        effective_min_net: float,
        *,
        frozen_set: frozenset[str] = frozenset(),
    ) -> None:
        long_sum = sum(w for w in targets.values() if w > 0.0)
        short_sum = sum(-w for w in targets.values() if w < 0.0)
        frozen_long = sum(w for sym, w in targets.items() if sym in frozen_set and w > 0.0)
        frozen_short = sum(-w for sym, w in targets.items() if sym in frozen_set and w < 0.0)
        non_frozen_long = long_sum - frozen_long
        non_frozen_short = short_sum - frozen_short
        net_exp = long_sum - short_sum
        if net_exp > effective_max_net and non_frozen_long > 1e-12:
            target_non_frozen_long = max(0.0, short_sum + effective_max_net - frozen_long)
            l_scale = min(1.0, target_non_frozen_long / non_frozen_long)
            for sym, w in list(targets.items()):
                if sym not in frozen_set and w > 0.0:
                    targets[sym] = w * l_scale
        elif net_exp < effective_min_net and non_frozen_short > 1e-12:
            target_non_frozen_short = max(0.0, long_sum - effective_min_net - frozen_short)
            s_scale = min(1.0, target_non_frozen_short / non_frozen_short)
            for sym, w in list(targets.items()):
                if sym not in frozen_set and w < 0.0:
                    targets[sym] = w * s_scale

    def build_proposed_orders(
        self,
        signals: Sequence[SignalRow],
        current_positions: Mapping[str, float],
        prices: Mapping[str, float],
        equity: float,
        dials: ControlDials,
        as_of: datetime,
        universe_meta: Mapping[str, AssetRiskMetadata],
        frozen_symbols: set[str] | frozenset[str] = frozenset(),
        *,
        target_weights_override: Mapping[str, float] | None = None,
    ) -> list[ProposedOrder]:
        """Convert signals and current signed share positions into ProposedOrders."""
        if not math.isfinite(equity) or equity <= 0.0:
            return []

        as_of_utc = ensure_utc(as_of)
        blacklist = dials.active_blacklist(as_of_utc)
        frozen_set = frozenset(s.strip().upper() for s in frozen_symbols if s.strip())
        signal_map: dict[str, SignalRow] = {}
        for s in signals:
            if s.as_of <= as_of_utc:
                prev = signal_map.get(s.symbol)
                if prev is None or s.as_of >= prev.as_of:
                    signal_map[s.symbol] = s

        norm_positions: dict[str, float] = {}
        for raw_sym, raw_qty in current_positions.items():
            sym = str(raw_sym).strip().upper()
            if sym and math.isfinite(float(raw_qty)):
                norm_positions[sym] = float(raw_qty)

        norm_prices: dict[str, float] = {}
        for raw_sym, raw_px in prices.items():
            sym = str(raw_sym).strip().upper()
            if sym and math.isfinite(float(raw_px)):
                norm_prices[sym] = float(raw_px)

        current_weights: dict[str, float] = {}
        for sym, qty in norm_positions.items():
            px = norm_prices.get(sym, 0.0)
            if px > 0.0 and math.isfinite(px):
                current_weights[sym] = (qty * px) / equity

        if target_weights_override is not None:
            target_weights: dict[str, float] = {}
            for raw_k, raw_w in target_weights_override.items():
                sym = str(raw_k).strip().upper()
                if not sym:
                    continue
                w_val = float(raw_w)
                if not math.isfinite(w_val):
                    raise ValueError(
                        f"target_weights_override contains non-finite weight for {sym}: {raw_w!r}"
                    )
                target_weights[sym] = w_val
            if dials.pause_new_entries:
                for sym, tgt in list(target_weights.items()):
                    if sym in frozen_set:
                        continue
                    cw = current_weights.get(sym, 0.0)
                    if sym in blacklist or cw == 0.0:
                        target_weights[sym] = 0.0
                    elif cw > 0.0:
                        target_weights[sym] = max(0.0, min(cw, tgt))
                    else:
                        target_weights[sym] = min(0.0, max(cw, tgt))
        else:
            target_weights = self.compute_target_weights(
                signals=list(signal_map.values()),
                dials=dials,
                as_of=as_of_utc,
                universe_meta=universe_meta,
                current_weights=current_weights,
                frozen_symbols=frozen_set,
            )

        macro_scale = min((s.macro_regime_scale for s in signal_map.values()), default=1.0)
        risk_scale = max(0.0, min(1.0, macro_scale * dials.exposure_multiplier))
        effective_gross_cap = self.risk.max_gross_exposure * risk_scale
        effective_max_net = min(self.risk.max_net_exposure, self.risk.max_net_exposure * risk_scale)
        effective_min_net = max(self.risk.min_net_exposure, self.risk.min_net_exposure * risk_scale)
        cur_gross = sum(abs(w) for w in current_weights.values())
        cur_net = sum(current_weights.values())
        portfolio_over_budget = (
            (cur_gross > effective_gross_cap + 1e-6)
            or (cur_net > effective_max_net + 1e-6)
            or (cur_net < effective_min_net - 1e-6)
        )

        orders: list[ProposedOrder] = []
        all_symbols = sorted(set(norm_positions.keys()) | set(target_weights.keys()))
        deadband = self.strategy.rebalance_deadband_pct

        for sym in all_symbols:
            if sym in frozen_set:
                continue
            px = norm_prices.get(sym, 0.0)
            if not math.isfinite(px) or px <= 0.0:
                continue

            cur_qty = float(norm_positions.get(sym, 0.0))
            cur_w = (cur_qty * px) / equity
            tgt_w = float(target_weights.get(sym, 0.0))

            # Skip small adjustments inside deadband unless closing fully, flipping sides,
            # or performing a risk-reducing trim while portfolio/position exceeds effective limits
            is_full_close = abs(tgt_w) <= 1e-9 and abs(cur_qty) > 1e-9
            is_side_flip = (cur_qty > 1e-9 and tgt_w < -1e-9) or (
                cur_qty < -1e-9 and tgt_w > 1e-9
            )
            is_trim = (cur_w > 1e-9 and 0.0 <= tgt_w < cur_w - 1e-9) or (
                cur_w < -1e-9 and cur_w + 1e-9 < tgt_w <= 0.0
            )
            pos_cap = (
                self.risk.max_long_position_pct
                if cur_w > 0.0
                else self.risk.max_short_position_pct
            )
            pos_over_cap = abs(cur_w) > pos_cap + 1e-6
            bypass_deadband = (
                is_full_close
                or is_side_flip
                or (is_trim and (portfolio_over_budget or pos_over_cap))
            )
            if not bypass_deadband and abs(tgt_w - cur_w) < deadband:
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
