from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from agy_trader.core.config import (
    LIVE_CONFIRMATION_PHRASE,
    EnvSettings,
    RiskConfig,
    validate_trading_mode,
)
from agy_trader.core.models import (
    AssetRiskMetadata,
    AssetType,
    ControlDials,
    OrderSide,
    PositionSide,
    ProposedOrder,
    SignalRow,
)
from agy_trader.risk.engine import HardRiskEngine, PortfolioRiskState

NY_TZ = ZoneInfo("America/New_York")


def _as_of_et(d: date, hour: int = 13, minute: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=NY_TZ).astimezone(UTC)


def _signal(
    symbol: str,
    as_of: datetime,
    *,
    atr_pct: float = 0.02,
    ann_vol: float = 0.30,
    ret_5d: float = 0.01,
    composite_score: float = 0.8,
) -> SignalRow:
    return SignalRow(
        as_of=as_of,
        symbol=symbol,
        trend_score=0.5,
        pullback_score=0.2,
        news_score=0.3,
        news_velocity_z=0.5,
        composite_score=composite_score,
        macro_regime_scale=1.0,
        atr_pct=atr_pct,
        ann_vol=ann_vol,
        ret_5d=ret_5d,
    )


def _meta(
    symbol: str,
    *,
    asset_type: AssetType = AssetType.equity,
    sector: str = "Technology",
    shortable: bool = True,
    easy_to_borrow: bool = True,
) -> AssetRiskMetadata:
    return AssetRiskMetadata(
        symbol=symbol,
        asset_type=asset_type,
        sector=sector,
        shortable=shortable,
        easy_to_borrow=easy_to_borrow,
    )


def _base_state(
    as_of: datetime,
    *,
    equity: float = 100_000.0,
    daily_pnl_pct: float = 0.0,
    drawdown_pct: float = 0.0,
    positions: dict[str, float] | None = None,
    prices: dict[str, float] | None = None,
    latest_bar_age_minutes: float | None = 1.0,
) -> PortfolioRiskState:
    return PortfolioRiskState(
        as_of=as_of,
        equity=equity,
        daily_pnl_pct=daily_pnl_pct,
        drawdown_pct=drawdown_pct,
        positions=positions or {},
        prices=prices or {"AAPL": 100.0},
        latest_bar_age_minutes=latest_bar_age_minutes,
    )


def test_kill_switches_block_pass2_entries_but_allow_pass1_exits() -> None:
    as_of = _as_of_et(date(2025, 3, 5))
    engine = HardRiskEngine(RiskConfig())
    dials = ControlDials()
    meta = {"AAPL": _meta("AAPL"), "MSFT": _meta("MSFT")}
    signals = {"AAPL": _signal("AAPL", as_of), "MSFT": _signal("MSFT", as_of)}

    orders = [
        ProposedOrder(
            symbol="AAPL",
            side=OrderSide.sell,
            qty=50.0,
            is_risk_reducing=True,
            reference_price=100.0,
        ),
        ProposedOrder(
            symbol="MSFT",
            side=OrderSide.buy,
            qty=20.0,
            is_risk_reducing=False,
            reference_price=200.0,
            stop_loss_price=190.0,
        ),
    ]

    # Daily -2% kill-switch
    daily_state = _base_state(
        as_of,
        daily_pnl_pct=-0.02,
        drawdown_pct=-0.02,
        positions={"AAPL": 50.0},
        prices={"AAPL": 100.0, "MSFT": 200.0},
    )
    verdicts_daily = engine.evaluate_batch(orders, daily_state, dials, meta, signals)
    assert verdicts_daily[0].approved is True
    assert verdicts_daily[0].adjusted_qty == 50.0
    assert verdicts_daily[1].approved is False
    assert "KILL_SWITCH_DAILY_LOSS" in verdicts_daily[1].violations

    # Max drawdown -8% kill-switch
    dd_state = _base_state(
        as_of,
        daily_pnl_pct=-0.005,
        drawdown_pct=-0.08,
        positions={"AAPL": 50.0},
        prices={"AAPL": 100.0, "MSFT": 200.0},
    )
    verdicts_dd = engine.evaluate_batch(orders, dd_state, dials, meta, signals)
    assert verdicts_dd[0].approved is True
    assert verdicts_dd[1].approved is False
    assert "KILL_SWITCH_MAX_DRAWDOWN" in verdicts_dd[1].violations


def test_portfolio_risk_state_validation_rejects_invalid_drawdown_and_bar_age() -> None:
    as_of = _as_of_et(date(2025, 3, 5))

    with pytest.raises(ValidationError):
        PortfolioRiskState(
            as_of=as_of,
            equity=100_000.0,
            daily_pnl_pct=0.0,
            drawdown_pct=0.05,
        )

    with pytest.raises(ValidationError):
        PortfolioRiskState(
            as_of=as_of,
            equity=100_000.0,
            daily_pnl_pct=0.0,
            drawdown_pct=-0.01,
            latest_bar_age_minutes=-0.1,
        )


def test_stale_market_data_and_stale_or_future_signals_rejected() -> None:
    as_of = _as_of_et(date(2025, 3, 5))
    engine = HardRiskEngine(RiskConfig(max_data_staleness_minutes=15), max_signal_age_hours=24.0)
    dials = ControlDials()
    meta = {"AAPL": _meta("AAPL")}
    buy_order = ProposedOrder(
        symbol="AAPL",
        side=OrderSide.buy,
        qty=10.0,
        reference_price=100.0,
        stop_loss_price=95.0,
    )

    # Stale market data (> 15 min)
    stale_data_state = _base_state(as_of, latest_bar_age_minutes=15.1)
    v_stale_data = engine.evaluate_batch(
        [buy_order], stale_data_state, dials, meta, {"AAPL": _signal("AAPL", as_of)}
    )
    assert v_stale_data[0].approved is False
    assert "STALE_MARKET_DATA" in v_stale_data[0].violations

    # Future signal (> as_of)
    healthy_state = _base_state(as_of, latest_bar_age_minutes=1.0)
    v_future_sig = engine.evaluate_batch(
        [buy_order],
        healthy_state,
        dials,
        meta,
        {"AAPL": _signal("AAPL", as_of + timedelta(minutes=5))},
    )
    assert v_future_sig[0].approved is False
    assert "STALE_OR_FUTURE_SIGNAL" in v_future_sig[0].violations

    # Stale signal (> max_signal_age_hours)
    v_old_sig = engine.evaluate_batch(
        [buy_order],
        healthy_state,
        dials,
        meta,
        {"AAPL": _signal("AAPL", as_of - timedelta(hours=25))},
    )
    assert v_old_sig[0].approved is False
    assert "STALE_OR_FUTURE_SIGNAL" in v_old_sig[0].violations


def test_short_selling_safety_and_partial_cover_fractional_guard() -> None:
    as_of = _as_of_et(date(2025, 3, 5))
    engine = HardRiskEngine(RiskConfig())
    state = _base_state(as_of, equity=100_000.0, prices={"GME": 100.0, "AMC": 100.0})
    short_order = ProposedOrder(
        symbol="GME",
        side=OrderSide.sell,
        qty=10.0,
        target_side=PositionSide.short,
        reference_price=100.0,
        stop_loss_price=104.0,
    )

    # 1. shortable=False
    v1 = engine.evaluate_batch(
        [short_order],
        state,
        ControlDials(),
        {"GME": _meta("GME", shortable=False)},
        {"GME": _signal("GME", as_of)},
    )
    assert v1[0].approved is False and "ASSET_NOT_SHORTABLE" in v1[0].violations

    # 2. easy_to_borrow=False
    v2 = engine.evaluate_batch(
        [short_order],
        state,
        ControlDials(),
        {"GME": _meta("GME", easy_to_borrow=False)},
        {"GME": _signal("GME", as_of)},
    )
    assert v2[0].approved is False and "ASSET_NOT_EASY_TO_BORROW" in v2[0].violations

    # 3. allow_shorting=False
    v3 = engine.evaluate_batch(
        [short_order],
        state,
        ControlDials(allow_shorting=False),
        {"GME": _meta("GME")},
        {"GME": _signal("GME", as_of)},
    )
    assert v3[0].approved is False and "SHORTING_DISABLED_BY_OPS" in v3[0].violations

    # 4. Squeeze ret_5d >= 0.20
    v4 = engine.evaluate_batch(
        [short_order],
        state,
        ControlDials(),
        {"GME": _meta("GME")},
        {"GME": _signal("GME", as_of, ret_5d=0.20)},
    )
    assert v4[0].approved is False and "SHORT_SQUEEZE_5D_RETURN_GUARD" in v4[0].violations

    # 5. Squeeze ann_vol >= 0.90
    v5 = engine.evaluate_batch(
        [short_order],
        state,
        ControlDials(),
        {"GME": _meta("GME")},
        {"GME": _signal("GME", as_of, ann_vol=0.90)},
    )
    assert v5[0].approved is False and "SHORT_SQUEEZE_VOLATILITY_GUARD" in v5[0].violations

    # 6. Fractional short qty (< 1 share after sizing cap): price $10,000, max short 6% of $100k = $6,000 -> 0.6 shares -> rejected
    expensive_state = _base_state(as_of, equity=100_000.0, prices={"GME": 10_000.0})
    expensive_short = ProposedOrder(
        symbol="GME",
        side=OrderSide.sell,
        qty=1.0,
        target_side=PositionSide.short,
        reference_price=10_000.0,
        stop_loss_price=10_400.0,
    )
    v6 = engine.evaluate_batch(
        [expensive_short],
        expensive_state,
        ControlDials(),
        {"GME": _meta("GME")},
        {"GME": _signal("GME", as_of)},
    )
    assert v6[0].approved is False and "FRACTIONAL_SHORT_REJECTED" in v6[0].violations

    # 7. Partial short cover leaving fractional short is rejected; full cover succeeds
    short_pos_state = _base_state(
        as_of,
        equity=100_000.0,
        positions={"AMC": -5.0},
        prices={"AMC": 100.0},
    )
    partial_frac_cover = ProposedOrder(
        symbol="AMC",
        side=OrderSide.buy,
        qty=0.4,
        is_risk_reducing=True,
        reference_price=100.0,
    )
    v_partial = engine.evaluate_batch(
        [partial_frac_cover],
        short_pos_state,
        ControlDials(),
        {"AMC": _meta("AMC")},
    )
    assert v_partial[0].approved is False
    assert "FRACTIONAL_SHORT_REMAINDER_REJECTED" in v_partial[0].violations

    full_cover = ProposedOrder(
        symbol="AMC",
        side=OrderSide.buy,
        qty=5.0,
        is_risk_reducing=True,
        reference_price=100.0,
    )
    v_full = engine.evaluate_batch(
        [full_cover],
        short_pos_state,
        ControlDials(),
        {"AMC": _meta("AMC")},
    )
    assert v_full[0].approved is True
    assert v_full[0].adjusted_qty == 5.0


def test_earnings_blackout_across_presidents_day_and_etf_exempt() -> None:
    engine = HardRiskEngine(RiskConfig(earnings_blackout_days=2))
    dials = ControlDials()
    meta = {
        "AAPL": _meta("AAPL", asset_type=AssetType.equity),
        "XLK": _meta("XLK", asset_type=AssetType.etf),
    }

    # Earnings on Wednesday 2025-02-19; Monday 2025-02-17 is Presidents' Day (NYSE closed).
    # Trading days from Friday 2025-02-14 to Wednesday 2025-02-19:
    # Fri -> Tue (1 trading day), Tue -> Wed (2 trading days) -> distance == 2 -> BLOCKED!
    # Thursday 2025-02-13 -> distance == 3 -> ALLOWED!
    earn_date = date(2025, 2, 19)
    earnings = {"AAPL": [earn_date], "XLK": [earn_date]}

    fri_as_of = _as_of_et(date(2025, 2, 14))
    aapl_order = ProposedOrder(
        symbol="AAPL", side=OrderSide.buy, qty=10.0, reference_price=100.0, stop_loss_price=95.0
    )
    etf_order = ProposedOrder(
        symbol="XLK", side=OrderSide.buy, qty=10.0, reference_price=100.0, stop_loss_price=95.0
    )
    fri_state = _base_state(fri_as_of, prices={"AAPL": 100.0, "XLK": 100.0})
    fri_sigs = {"AAPL": _signal("AAPL", fri_as_of), "XLK": _signal("XLK", fri_as_of)}

    v_fri = engine.evaluate_batch(
        [aapl_order, etf_order], fri_state, dials, meta, fri_sigs, earnings
    )
    assert v_fri[0].approved is False and "EARNINGS_BLACKOUT" in v_fri[0].violations
    assert v_fri[1].approved is True  # ETFs NOT blocked

    thu_as_of = _as_of_et(date(2025, 2, 13))
    thu_state = _base_state(thu_as_of, prices={"AAPL": 100.0})
    thu_sigs = {"AAPL": _signal("AAPL", thu_as_of)}
    v_thu = engine.evaluate_batch([aapl_order], thu_state, dials, meta, thu_sigs, earnings)
    assert v_thu[0].approved is True


def test_earnings_blackout_none_disables_entry_block_while_preserving_other_guards() -> None:
    assert RiskConfig().earnings_blackout_days == 2
    earn_day = date(2025, 2, 19)
    as_of = _as_of_et(earn_day)
    meta = {"AAPL": _meta("AAPL", asset_type=AssetType.equity)}
    order = ProposedOrder(
        symbol="AAPL", side=OrderSide.buy, qty=10.0, reference_price=100.0, stop_loss_price=95.0
    )
    earnings = {"AAPL": [earn_day]}

    engine_disabled = HardRiskEngine(RiskConfig(earnings_blackout_days=None))
    state = _base_state(as_of, prices={"AAPL": 100.0})
    sig_clean = {"AAPL": _signal("AAPL", as_of)}
    v_ok = engine_disabled.evaluate_batch([order], state, ControlDials(), meta, sig_clean, earnings)
    assert v_ok[0].approved is True
    assert v_ok[0].adjusted_qty == 10.0
    assert v_ok[0].stop_loss_price == 95.0

    # With stop_loss_price=None, ATR stop synthesis still runs normally (default mult 2.5, atr_pct=0.02 -> 95.0)
    order_no_stop = ProposedOrder(
        symbol="AAPL", side=OrderSide.buy, qty=10.0, reference_price=100.0, stop_loss_price=None
    )
    v_synth = engine_disabled.evaluate_batch(
        [order_no_stop], state, ControlDials(), meta, sig_clean, earnings
    )
    assert v_synth[0].approved is True
    assert v_synth[0].stop_loss_price == pytest.approx(95.0)

    # Exact-day boundary earnings_blackout_days=0 still blocks earn_day, unlike None
    engine_zero = HardRiskEngine(RiskConfig(earnings_blackout_days=0))
    v_zero = engine_zero.evaluate_batch([order], state, ControlDials(), meta, sig_clean, earnings)
    assert v_zero[0].approved is False
    assert "EARNINGS_BLACKOUT" in v_zero[0].violations

    # RESTATEMENT_BLACKOUT and BLACKOUT_PERIOD still block even when earnings_blackout_days=None
    sig_restate = {
        "AAPL": _signal("AAPL", as_of).model_copy(update={"has_recent_restatement": True})
    }
    v_restate = engine_disabled.evaluate_batch(
        [order], state, ControlDials(), meta, sig_restate, earnings
    )
    assert v_restate[0].approved is False
    assert "RESTATEMENT_BLACKOUT" in v_restate[0].violations

    v_frozen = engine_disabled.evaluate_batch(
        [order], state, ControlDials(), meta, sig_clean, earnings, frozen_symbols={"AAPL"}
    )
    assert v_frozen[0].approved is False
    assert "BLACKOUT_PERIOD_FROZEN" in v_frozen[0].violations

    v_halted = engine_disabled.evaluate_batch(
        [order], state, ControlDials(), meta, sig_clean, earnings, blackout_halted=True
    )
    assert v_halted[0].approved is False
    assert "BLACKOUT_PERIOD_HALT" in v_halted[0].violations


def test_cumulative_batch_caps_portfolio_price_integrity_and_ref_divergence() -> None:
    as_of = _as_of_et(date(2025, 3, 5))
    cfg = RiskConfig(
        max_long_position_pct=0.10,
        max_sector_gross_pct=0.25,
        max_gross_exposure=1.00,
        max_net_exposure=0.60,
    )
    engine = HardRiskEngine(cfg)
    dials = ControlDials()
    tech_syms = ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL"]
    meta = {s: _meta(s, sector="Technology") for s in tech_syms}
    prices = {s: 100.0 for s in tech_syms}
    signals = {s: _signal(s, as_of) for s in tech_syms}

    # 5 simultaneous $10k Tech longs = $50k, must be capped at max_sector_gross_pct=0.25 ($25k)
    orders = [
        ProposedOrder(
            symbol=s,
            side=OrderSide.buy,
            qty=100.0,  # $10k each
            reference_price=100.0,
            stop_loss_price=95.0,
        )
        for s in tech_syms
    ]
    state = _base_state(as_of, equity=100_000.0, prices=prices)
    verdicts = engine.evaluate_batch(orders, state, dials, meta, signals)
    total_approved_notional = sum(v.adjusted_qty * 100.0 for v in verdicts if v.approved)
    assert total_approved_notional <= 25_000.0 + 1e-6
    assert total_approved_notional == pytest.approx(25_000.0)
    assert verdicts[0].approved and verdicts[1].approved
    assert verdicts[2].approved and verdicts[2].adjusted_qty == pytest.approx(50.0)
    assert verdicts[3].approved is False and "SECTOR_GROSS_CAP_EXCEEDED" in verdicts[3].violations
    assert verdicts[4].approved is False and "SECTOR_GROSS_CAP_EXCEEDED" in verdicts[4].violations

    # Unpriced existing open position fails closed
    unpriced_state = _base_state(
        as_of,
        equity=100_000.0,
        positions={"IBM": 10.0},
        prices={"AAPL": 100.0},
    )
    v_unpriced = engine.evaluate_batch(
        [orders[0]],
        unpriced_state,
        dials,
        {"AAPL": _meta("AAPL"), "IBM": _meta("IBM", sector="Industrials")},
        {"AAPL": _signal("AAPL", as_of)},
    )
    assert v_unpriced[0].approved is False
    assert "MISSING_OR_INVALID_PORTFOLIO_PRICE" in v_unpriced[0].violations

    # Reference price diverging > 2% from state.prices rejected
    divergent_order = ProposedOrder(
        symbol="AAPL",
        side=OrderSide.buy,
        qty=10.0,
        reference_price=103.0,  # 3% above state price 100.0
        stop_loss_price=95.0,
    )
    v_div = engine.evaluate_batch(
        [divergent_order],
        _base_state(as_of, prices={"AAPL": 100.0}),
        dials,
        {"AAPL": _meta("AAPL")},
        {"AAPL": _signal("AAPL", as_of)},
    )
    assert v_div[0].approved is False
    assert "REFERENCE_PRICE_DIVERGENCE" in v_div[0].violations

    # Gross and net exposure cumulative bounds with multiple sectors
    broad_syms = ["A1", "A2", "A3", "A4", "B1", "B2", "B3"]
    broad_meta = {
        "A1": _meta("A1", sector="Tech"),
        "A2": _meta("A2", sector="Tech"),
        "A3": _meta("A3", sector="Health"),
        "A4": _meta("A4", sector="Health"),
        "B1": _meta("B1", sector="Finance"),
        "B2": _meta("B2", sector="Finance"),
        "B3": _meta("B3", sector="Energy"),
    }
    broad_prices = {s: 100.0 for s in broad_syms}
    broad_sigs = {s: _signal(s, as_of) for s in broad_syms}
    broad_orders = [
        ProposedOrder(
            symbol=s, side=OrderSide.buy, qty=100.0, reference_price=100.0, stop_loss_price=95.0
        )
        for s in broad_syms
    ]
    v_net = engine.evaluate_batch(
        broad_orders,
        _base_state(as_of, equity=100_000.0, prices=broad_prices),
        dials,
        broad_meta,
        broad_sigs,
    )
    total_long_notional = sum(v.adjusted_qty * 100.0 for v in v_net if v.approved)
    assert total_long_notional == pytest.approx(60_000.0)  # max_net_exposure=0.60
    assert v_net[-1].approved is False and "NET_EXPOSURE_BOUND_EXCEEDED" in v_net[-1].violations

    # Gross cap with exposure_multiplier=0.4 (effective gross = 40% = $40k)
    v_gross = engine.evaluate_batch(
        broad_orders,
        _base_state(as_of, equity=100_000.0, prices=broad_prices),
        ControlDials(exposure_multiplier=0.40),
        broad_meta,
        broad_sigs,
    )
    assert sum(v.adjusted_qty * 100.0 for v in v_gross if v.approved) == pytest.approx(40_000.0)


def test_side_flip_batch_and_unsplit_side_flip_rejection() -> None:
    as_of = _as_of_et(date(2025, 3, 5))
    engine = HardRiskEngine(RiskConfig())
    dials = ControlDials()
    meta = {"AAPL": _meta("AAPL")}
    signals = {"AAPL": _signal("AAPL", as_of)}
    state = _base_state(as_of, equity=100_000.0, positions={"AAPL": 100.0}, prices={"AAPL": 100.0})

    # Split side-flip: SELL 100 close + SELL 60 open short -> BOTH approved!
    split_orders = [
        ProposedOrder(
            symbol="AAPL",
            side=OrderSide.sell,
            qty=100.0,
            is_risk_reducing=True,
            reference_price=100.0,
        ),
        ProposedOrder(
            symbol="AAPL",
            side=OrderSide.sell,
            qty=60.0,
            target_side=PositionSide.short,
            is_risk_reducing=False,
            reference_price=100.0,
            stop_loss_price=104.0,
        ),
    ]
    verdicts_split = engine.evaluate_batch(split_orders, state, dials, meta, signals)
    assert verdicts_split[0].approved is True and verdicts_split[0].adjusted_qty == 100.0
    assert verdicts_split[1].approved is True and verdicts_split[1].adjusted_qty == 60.0

    # Untrusted is_risk_reducing=True on unsplit SELL 200 while holding +100 -> rejected!
    unsplit_order = ProposedOrder(
        symbol="AAPL",
        side=OrderSide.sell,
        qty=200.0,
        is_risk_reducing=True,
        reference_price=100.0,
        stop_loss_price=104.0,
    )
    verdicts_unsplit = engine.evaluate_batch([unsplit_order], state, dials, meta, signals)
    assert verdicts_unsplit[0].approved is False
    assert "UNSPLIT_SIDE_FLIP_REJECTED" in verdicts_unsplit[0].violations


def test_stop_loss_enforcement_clamping_fallback_and_post_rounding() -> None:
    as_of = _as_of_et(date(2025, 3, 5))
    engine = HardRiskEngine(RiskConfig(stop_loss_atr_mult_long=2.5, stop_loss_atr_mult_short=2.0))
    dials = ControlDials()
    meta = {"AAPL": _meta("AAPL")}

    # 1. Wrong-direction long stop (>= anchor) rejected
    state = _base_state(as_of, prices={"AAPL": 100.0})
    sig = {"AAPL": _signal("AAPL", as_of, atr_pct=0.02)}
    bad_dir = ProposedOrder(
        symbol="AAPL", side=OrderSide.buy, qty=10.0, reference_price=100.0, stop_loss_price=101.0
    )
    v_bad_dir = engine.evaluate_batch([bad_dir], state, dials, meta, sig)
    assert v_bad_dir[0].approved is False and "MISSING_OR_INVALID_STOP_LOSS" in v_bad_dir[0].violations

    # 2. Wide long stop clamped by ATR multiplier (100 * (1 - 2.5*0.02) = 95.0)
    wide_long = ProposedOrder(
        symbol="AAPL", side=OrderSide.buy, qty=10.0, reference_price=100.0, stop_loss_price=80.0
    )
    v_clamp_long = engine.evaluate_batch([wide_long], state, dials, meta, sig)
    assert v_clamp_long[0].approved is True
    assert v_clamp_long[0].stop_loss_price == pytest.approx(95.0)

    # 3. Wide short stop clamped by ATR multiplier (100 * (1 + 2.0*0.02) = 104.0)
    wide_short = ProposedOrder(
        symbol="AAPL",
        side=OrderSide.sell,
        qty=10.0,
        target_side=PositionSide.short,
        reference_price=100.0,
        stop_loss_price=130.0,
    )
    v_clamp_short = engine.evaluate_batch([wide_short], state, dials, meta, sig)
    assert v_clamp_short[0].approved is True
    assert v_clamp_short[0].stop_loss_price == pytest.approx(104.0)

    # 4. Fallback max long stop width (25%) when signal is absent
    very_wide_long = ProposedOrder(
        symbol="AAPL", side=OrderSide.buy, qty=10.0, reference_price=100.0, stop_loss_price=60.0
    )
    v_fallback = engine.evaluate_batch([very_wide_long], state, dials, meta, signals_by_symbol=None)
    assert v_fallback[0].approved is True
    assert v_fallback[0].stop_loss_price == pytest.approx(75.0)

    # 5. Post-rounding guard: tiny ATR rounds stop_loss_price right onto anchor_px -> rejected
    tiny_px_state = _base_state(as_of, prices={"AAPL": 0.01})
    tiny_sig = {"AAPL": _signal("AAPL", as_of, atr_pct=0.001)}
    tiny_order = ProposedOrder(
        symbol="AAPL", side=OrderSide.buy, qty=10.0, reference_price=0.01, stop_loss_price=0.00999
    )
    v_round = engine.evaluate_batch([tiny_order], tiny_px_state, dials, meta, tiny_sig)
    assert v_round[0].approved is False
    assert "MISSING_OR_INVALID_STOP_LOSS" in v_round[0].violations


def test_live_trading_double_lock_validate_trading_mode() -> None:
    # Paper mode passes even without keys or phrase
    validate_trading_mode(EnvSettings(alpaca_paper=True))

    # Non-paper missing keys -> fails
    with pytest.raises(ValueError, match="double-lock"):
        validate_trading_mode(
            EnvSettings(
                alpaca_paper=False,
                alpaca_api_key="",
                alpaca_secret_key="secret",
                live_trading_confirmation=LIVE_CONFIRMATION_PHRASE,
            )
        )

    # Non-paper missing/wrong confirmation phrase -> fails
    with pytest.raises(ValueError, match="double-lock"):
        validate_trading_mode(
            EnvSettings(
                alpaca_paper=False,
                alpaca_api_key="key",
                alpaca_secret_key="secret",
                live_trading_confirmation="YES",
            )
        )

    # All 3 conditions met -> passes
    validate_trading_mode(
        EnvSettings(
            alpaca_paper=False,
            alpaca_api_key="key",
            alpaca_secret_key="secret",
            live_trading_confirmation=LIVE_CONFIRMATION_PHRASE,
        )
    )


def test_risk_engine_additional_hard_guards_and_min_net_exposure() -> None:
    as_of = _as_of_et(date(2025, 3, 5))
    cfg = RiskConfig(
        max_short_position_pct=0.10,
        max_sector_gross_pct=0.50,
        max_gross_exposure=1.00,
        min_net_exposure=-0.25,
    )
    engine = HardRiskEngine(cfg)
    meta = {
        "AAPL": _meta("AAPL", sector="Tech"),
        "MSFT": _meta("MSFT", sector="Finance"),
        "XOM": _meta("XOM", sector="Energy"),
    }
    buy_aapl = ProposedOrder(
        symbol="AAPL", side=OrderSide.buy, qty=10.0, reference_price=100.0, stop_loss_price=95.0
    )

    # 1. latest_bar_age_minutes=None must fail closed with STALE_MARKET_DATA
    none_age_state = _base_state(as_of, latest_bar_age_minutes=None, prices={"AAPL": 100.0})
    v_none_age = engine.evaluate_batch(
        [buy_aapl], none_age_state, ControlDials(), meta, {"AAPL": _signal("AAPL", as_of)}
    )
    assert v_none_age[0].approved is False
    assert "STALE_MARKET_DATA" in v_none_age[0].violations

    # 2. ControlDials.pause_new_entries=True blocks Pass 2 entries while allowing Pass 1 exits
    healthy_state = _base_state(
        as_of,
        positions={"MSFT": 10.0},
        prices={"AAPL": 100.0, "MSFT": 100.0},
    )
    exit_msft = ProposedOrder(
        symbol="MSFT", side=OrderSide.sell, qty=10.0, is_risk_reducing=True, reference_price=100.0
    )
    v_paused = engine.evaluate_batch(
        [exit_msft, buy_aapl],
        healthy_state,
        ControlDials(pause_new_entries=True),
        meta,
        {"AAPL": _signal("AAPL", as_of)},
    )
    assert v_paused[0].approved is True
    assert v_paused[1].approved is False
    assert "PAUSED_BY_OPS" in v_paused[1].violations

    # 3. blacklisted_symbols blocks entries for active blacklist
    v_bl = engine.evaluate_batch(
        [buy_aapl],
        _base_state(as_of, prices={"AAPL": 100.0}),
        ControlDials(blacklisted_symbols={"AAPL": as_of + timedelta(hours=4)}),
        meta,
        {"AAPL": _signal("AAPL", as_of)},
    )
    assert v_bl[0].approved is False
    assert "BLACKLISTED_SYMBOL" in v_bl[0].violations

    # 4. UNKNOWN_ASSET_METADATA when symbol is missing from universe_meta
    v_unknown = engine.evaluate_batch(
        [
            ProposedOrder(
                symbol="PLTR",
                side=OrderSide.buy,
                qty=10.0,
                reference_price=100.0,
                stop_loss_price=95.0,
            )
        ],
        _base_state(as_of, prices={"PLTR": 100.0}),
        ControlDials(),
        meta,
        {"PLTR": _signal("PLTR", as_of)},
    )
    assert v_unknown[0].approved is False
    assert "UNKNOWN_ASSET_METADATA" in v_unknown[0].violations

    # 5. Short without signal rejected with SHORT_MISSING_SIGNAL_FOR_SQUEEZE_CHECK
    short_aapl = ProposedOrder(
        symbol="AAPL",
        side=OrderSide.sell,
        qty=10.0,
        target_side=PositionSide.short,
        reference_price=100.0,
        stop_loss_price=104.0,
    )
    v_no_sig_short = engine.evaluate_batch(
        [short_aapl],
        _base_state(as_of, prices={"AAPL": 100.0}),
        ControlDials(),
        meta,
        signals_by_symbol=None,
    )
    assert v_no_sig_short[0].approved is False
    assert "SHORT_MISSING_SIGNAL_FOR_SQUEEZE_CHECK" in v_no_sig_short[0].violations

    # 6. min_net_exposure (-0.25) breach on cumulative shorts:
    # 3 short orders of $10k each = -$30k (-0.30 net) -> 3rd clipped to $5k (50 shares),
    # and a 4th short order is rejected with NET_EXPOSURE_BOUND_EXCEEDED!
    short_meta = {
        "S1": _meta("S1", sector="Tech"),
        "S2": _meta("S2", sector="Finance"),
        "S3": _meta("S3", sector="Energy"),
        "S4": _meta("S4", sector="Healthcare"),
    }
    short_prices = {s: 100.0 for s in short_meta}
    short_sigs = {s: _signal(s, as_of) for s in short_meta}
    short_orders = [
        ProposedOrder(
            symbol=s,
            side=OrderSide.sell,
            qty=100.0,
            target_side=PositionSide.short,
            reference_price=100.0,
            stop_loss_price=104.0,
        )
        for s in ("S1", "S2", "S3", "S4")
    ]
    v_min_net = engine.evaluate_batch(
        short_orders,
        _base_state(as_of, equity=100_000.0, prices=short_prices),
        ControlDials(),
        short_meta,
        short_sigs,
    )
    assert v_min_net[0].adjusted_qty == 100.0
    assert v_min_net[1].adjusted_qty == 100.0
    assert v_min_net[2].adjusted_qty == 50.0  # hits -0.25 * $100k = -$25k net floor
    assert v_min_net[3].approved is False
    assert "NET_EXPOSURE_BOUND_EXCEEDED" in v_min_net[3].violations


def test_restatement_blackout_blocks_new_longs_allows_exits_and_respects_zero_config() -> None:
    as_of = _as_of_et(date(2025, 3, 5))
    meta = {"AAPL": _meta("AAPL"), "MSFT": _meta("MSFT")}
    dials = ControlDials()
    sig_restated = _signal("AAPL", as_of).model_copy(update={"has_recent_restatement": True})
    sig_clean = _signal("MSFT", as_of).model_copy(update={"has_recent_restatement": False})
    signals = {"AAPL": sig_restated, "MSFT": sig_clean}

    buy_aapl = ProposedOrder(
        symbol="AAPL",
        side=OrderSide.buy,
        qty=20.0,
        target_side=PositionSide.long,
        reference_price=100.0,
        stop_loss_price=95.0,
    )
    sell_exit_aapl = ProposedOrder(
        symbol="AAPL",
        side=OrderSide.sell,
        qty=15.0,
        is_risk_reducing=True,
        reference_price=100.0,
    )
    short_aapl = ProposedOrder(
        symbol="AAPL",
        side=OrderSide.sell,
        qty=10.0,
        target_side=PositionSide.short,
        reference_price=100.0,
        stop_loss_price=104.0,
    )

    engine = HardRiskEngine(RiskConfig(restatement_blackout_days=10))
    # Pass 1 exit allowed; Pass 2 long entry blocked with RESTATEMENT_BLACKOUT
    verdicts = engine.evaluate_batch(
        [sell_exit_aapl, buy_aapl],
        _base_state(as_of, positions={"AAPL": 15.0}, prices={"AAPL": 100.0, "MSFT": 100.0}),
        dials,
        meta,
        signals,
    )
    assert verdicts[0].approved is True
    assert verdicts[0].adjusted_qty == 15.0
    assert verdicts[1].approved is False
    assert "RESTATEMENT_BLACKOUT" in verdicts[1].violations

    # Short entry is NOT blocked by RESTATEMENT_BLACKOUT
    verdict_short = engine.evaluate_batch(
        [short_aapl],
        _base_state(as_of, prices={"AAPL": 100.0}),
        dials,
        meta,
        signals,
    )
    assert verdict_short[0].approved is True

    # restatement_blackout_days=0 disables the blackout
    engine_off = HardRiskEngine(RiskConfig(restatement_blackout_days=0))
    verdict_off = engine_off.evaluate_batch(
        [buy_aapl],
        _base_state(as_of, prices={"AAPL": 100.0}),
        dials,
        meta,
        signals,
    )
    assert verdict_off[0].approved is True


