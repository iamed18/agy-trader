from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from agy_trader.backtest.metrics import (
    EquityPoint,
    SimulatedTrade,
    compute_backtest_metrics,
)
from agy_trader.core.config import RiskConfig, StrategyConfig
from agy_trader.core.models import (
    AssetRiskMetadata,
    AssetType,
    ControlDials,
    OrderSide,
    PositionSide,
    SignalRow,
)
from agy_trader.data.alpaca_market import AlpacaMarketDataProvider
from agy_trader.strategy.portfolio import PortfolioConstructor

NY_TZ = ZoneInfo("America/New_York")


def _eq_pt(day_offset: int, equity: float) -> EquityPoint:
    return EquityPoint(
        timestamp=datetime(2025, 1, 2, 21, 0, tzinfo=UTC) + timedelta(days=day_offset),
        equity=equity,
        cash=equity,
        gross_exposure=0.0,
        net_exposure=0.0,
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
    )


def _trade(symbol: str, net_pnl: float, holding_days: float = 2.0) -> SimulatedTrade:
    t0 = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    return SimulatedTrade(
        symbol=symbol,
        side="long",
        entry_time=t0,
        exit_time=t0 + timedelta(days=holding_days),
        qty=10.0,
        entry_price=100.0,
        exit_price=100.0 + net_pnl / 10.0,
        gross_pnl=net_pnl,
        net_pnl=net_pnl,
        holding_days=holding_days,
        exit_reason="test",
    )


def _sig(
    symbol: str,
    as_of: datetime,
    *,
    composite_score: float = 0.8,
    atr_pct: float = 0.02,
    macro_regime_scale: float = 1.0,
    ret_5d: float = 0.0,
    ann_vol: float = 0.25,
) -> SignalRow:
    return SignalRow(
        as_of=as_of,
        symbol=symbol,
        trend_score=composite_score,
        pullback_score=0.0,
        news_score=0.0,
        news_velocity_z=0.0,
        composite_score=composite_score,
        macro_regime_scale=macro_regime_scale,
        atr_pct=atr_pct,
        ann_vol=ann_vol,
        ret_5d=ret_5d,
    )


def _meta(
    symbol: str,
    *,
    sector: str = "Technology",
    asset_type: AssetType = AssetType.equity,
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


# ---------------------------------------------------------------------------
# Metrics direct formula & bankruptcy tests
# ---------------------------------------------------------------------------


def test_compute_backtest_metrics_exact_formulas() -> None:
    initial_capital = 100_000.0
    # Period returns: +2%, -2%, +4%
    equities = [100_000.0, 102_000.0, 99_960.0, 103_958.4]
    curve = [_eq_pt(i, eq) for i, eq in enumerate(equities)]

    trades = [
        _trade("AAPL", 300.0, holding_days=1.5),
        _trade("MSFT", -100.0, holding_days=3.0),
        _trade("NVDA", 200.0, holding_days=4.5),
    ]
    benchmark_closes = [200.0, 204.0, 199.92, 203.9184]  # +2%, -2%, +2%
    traded_notional = 50_000.0

    metrics = compute_backtest_metrics(
        label="ExactCheck",
        equity_curve=curve,
        trades=trades,
        initial_capital=initial_capital,
        total_traded_notional=traded_notional,
        total_orders_filled=6,
        stop_loss_exits=1,
        risk_rejections=2,
        total_slippage_usd=12.345,
        total_borrow_cost_usd=3.456,
        total_sec_fees_usd=0.789,
        benchmark_symbol="SPY",
        benchmark_closes=benchmark_closes,
    )

    rets = np.array([0.02, -0.02, 0.04], dtype=float)
    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets, ddof=1))
    exp_cagr = (103_958.4 / 100_000.0) ** (252.0 / 3.0) - 1.0
    exp_ann_vol = std_r * math.sqrt(252.0)
    exp_sharpe = (mean_r / std_r) * math.sqrt(252.0)
    downside_dev = math.sqrt((0.02**2) / 3.0)
    exp_sortino = (mean_r / downside_dev) * math.sqrt(252.0)
    exp_max_dd = (99_960.0 - 102_000.0) / 102_000.0  # -0.02
    exp_calmar = exp_cagr / abs(exp_max_dd)

    avg_equity = float(np.mean(equities))
    years = 4.0 / 252.0
    exp_turnover = (traded_notional / 2.0) / (avg_equity * years)

    bm_rets = np.array([0.02, -0.02, 0.02], dtype=float)
    bm_sharpe = (float(np.mean(bm_rets)) / float(np.std(bm_rets, ddof=1))) * math.sqrt(252.0)
    bm_ret = (203.9184 / 200.0) - 1.0
    bm_max_dd = (199.92 - 204.0) / 204.0

    assert metrics.total_return_pct == round(((103_958.4 / 100_000.0) - 1.0) * 100.0, 2)
    assert metrics.cagr_pct == round(exp_cagr * 100.0, 2)
    assert metrics.ann_volatility_pct == round(exp_ann_vol * 100.0, 2)
    assert metrics.sharpe_ratio == round(exp_sharpe, 3)
    assert metrics.sortino_ratio == round(exp_sortino, 3)
    assert metrics.calmar_ratio == round(exp_calmar, 3)
    assert metrics.max_drawdown_pct == round(exp_max_dd * 100.0, 2)
    assert metrics.win_rate_pct == round((2.0 / 3.0) * 100.0, 2)
    assert metrics.profit_factor == 5.0
    assert metrics.avg_holding_days == 3.0
    assert metrics.annual_turnover == round(exp_turnover, 2)
    assert metrics.benchmark_return_pct == round(bm_ret * 100.0, 2)
    assert metrics.benchmark_sharpe == round(bm_sharpe, 3)
    assert metrics.benchmark_max_drawdown_pct == round(bm_max_dd * 100.0, 2)

    # Zero-loss profit factor sentinel cap 999.0
    zero_loss_metrics = compute_backtest_metrics(
        label="ZeroLoss",
        equity_curve=curve,
        trades=[_trade("AAPL", 250.0), _trade("MSFT", 150.0)],
        initial_capital=initial_capital,
        total_traded_notional=10_000.0,
        total_orders_filled=4,
        stop_loss_exits=0,
        risk_rejections=0,
        total_slippage_usd=0.0,
        total_borrow_cost_usd=0.0,
        total_sec_fees_usd=0.0,
        benchmark_symbol="SPY",
        benchmark_closes=benchmark_closes,
    )
    assert zero_loss_metrics.win_rate_pct == 100.0
    assert zero_loss_metrics.profit_factor == 999.0


def test_compute_backtest_metrics_bankruptcy_and_non_positive_equity() -> None:
    for equities in (
        [100_000.0, 50_000.0, 0.0, 0.0],
        [100_000.0, -5_000.0, 10_000.0],
    ):
        curve = [_eq_pt(i, eq) for i, eq in enumerate(equities)]
        metrics = compute_backtest_metrics(
            label="Bankrupt",
            equity_curve=curve,
            trades=[],
            initial_capital=100_000.0,
            total_traded_notional=0.0,
            total_orders_filled=0,
            stop_loss_exits=0,
            risk_rejections=0,
            total_slippage_usd=0.0,
            total_borrow_cost_usd=0.0,
            total_sec_fees_usd=0.0,
            benchmark_symbol="SPY",
            benchmark_closes=[100.0, 100.0],
        )
        assert math.isfinite(metrics.sharpe_ratio)
        assert math.isfinite(metrics.sortino_ratio)
        assert math.isfinite(metrics.calmar_ratio)
        assert metrics.ann_volatility_pct < 1e6
        assert metrics.max_drawdown_pct == -100.0
        assert metrics.cagr_pct == -100.0


# ---------------------------------------------------------------------------
# PortfolioConstructor direct unit tests
# ---------------------------------------------------------------------------


def test_portfolio_inverse_atr_weighting_and_deadband_and_side_flips() -> None:
    as_of = datetime(2025, 4, 1, 20, 0, tzinfo=UTC)
    strat = StrategyConfig(
        min_signal_threshold=0.25,
        rebalance_deadband_pct=0.02,
        top_k_long=4,
        top_k_short=2,
    )
    risk = RiskConfig(
        max_long_position_pct=0.25,
        max_short_position_pct=0.15,
        max_sector_gross_pct=1.0,
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
    )
    ctor = PortfolioConstructor(strat, risk)
    dials = ControlDials()
    meta = {
        "LOW_ATR": _meta("LOW_ATR", sector="Tech"),
        "HIGH_ATR": _meta("HIGH_ATR", sector="Finance"),
    }

    # Equal composite score, but LOW_ATR has 1% ATR vs HIGH_ATR 4% ATR -> LOW_ATR gets 4x raw weight
    sigs = [
        _sig("LOW_ATR", as_of, composite_score=0.7, atr_pct=0.01),
        _sig("HIGH_ATR", as_of, composite_score=0.7, atr_pct=0.04),
    ]
    weights = ctor.compute_target_weights(sigs, dials, as_of, meta)
    assert weights["LOW_ATR"] > weights["HIGH_ATR"] > 0.0

    # Deadband suppresses small rebalance tweaks (< 2%)
    # Set target weight for LOW_ATR to ~0.25 and current weight to 0.24 (diff 0.01 < deadband 0.02)
    equity = 100_000.0
    prices = {"LOW_ATR": 100.0, "HIGH_ATR": 100.0}
    cur_positions = {
        "LOW_ATR": (weights["LOW_ATR"] - 0.005) * equity / 100.0,
        "HIGH_ATR": (weights["HIGH_ATR"] + 0.005) * equity / 100.0,
    }
    suppressed = ctor.build_proposed_orders(
        signals=sigs,
        current_positions=cur_positions,
        prices=prices,
        equity=equity,
        dials=dials,
        as_of=as_of,
        universe_meta=meta,
    )
    assert suppressed == []

    # Full close allowed even when current weight is smaller than deadband (e.g. 0.5% < 2%)
    small_exit_orders = ctor.build_proposed_orders(
        signals=[_sig("LOW_ATR", as_of, composite_score=0.0)],
        current_positions={"LOW_ATR": 5.0},  # $500 / $100k = 0.005 < deadband 0.02
        prices={"LOW_ATR": 100.0},
        equity=equity,
        dials=dials,
        as_of=as_of,
        universe_meta=meta,
    )
    assert len(small_exit_orders) == 1
    assert small_exit_orders[0].is_risk_reducing is True
    assert small_exit_orders[0].reason == "exit_to_flat"
    assert small_exit_orders[0].side == OrderSide.sell

    # Side flips: Long -> Short and Short -> Long MUST emit two separate orders each
    flip_meta = {
        "FLIP_L2S": _meta("FLIP_L2S", sector="Tech"),
        "FLIP_S2L": _meta("FLIP_S2L", sector="Energy"),
    }
    flip_sigs = [
        _sig("FLIP_L2S", as_of, composite_score=-0.8, atr_pct=0.02),
        _sig("FLIP_S2L", as_of, composite_score=0.8, atr_pct=0.02),
    ]
    flip_orders = ctor.build_proposed_orders(
        signals=flip_sigs,
        current_positions={"FLIP_L2S": 10.0, "FLIP_S2L": -10.0},
        prices={"FLIP_L2S": 100.0, "FLIP_S2L": 100.0},
        equity=equity,
        dials=dials,
        as_of=as_of,
        universe_meta=flip_meta,
    )
    by_sym: dict[str, list] = {"FLIP_L2S": [], "FLIP_S2L": []}
    for o in flip_orders:
        by_sym[o.symbol].append(o)

    # FLIP_L2S: sell 10 close (reducing) + sell open short (not reducing)
    assert len(by_sym["FLIP_L2S"]) == 2
    c1, o1 = by_sym["FLIP_L2S"]
    assert c1.reason == "signal_flip_close"
    assert c1.is_risk_reducing is True
    assert c1.side == OrderSide.sell
    assert c1.qty == 10.0
    assert o1.reason == "signal_flip_open"
    assert o1.is_risk_reducing is False
    assert o1.side == OrderSide.sell
    assert o1.target_side == PositionSide.short

    # FLIP_S2L: buy 10 cover (reducing) + buy open long (not reducing)
    assert len(by_sym["FLIP_S2L"]) == 2
    c2, o2 = by_sym["FLIP_S2L"]
    assert c2.reason == "signal_flip_close"
    assert c2.is_risk_reducing is True
    assert c2.side == OrderSide.buy
    assert c2.qty == 10.0
    assert o2.reason == "signal_flip_open"
    assert o2.is_risk_reducing is False
    assert o2.side == OrderSide.buy
    assert o2.target_side == PositionSide.long


def test_portfolio_empty_signals_caps_squeeze_filter_and_pause() -> None:
    as_of = datetime(2025, 4, 1, 20, 0, tzinfo=UTC)
    strat = StrategyConfig(min_signal_threshold=0.3, top_k_long=5, top_k_short=2)
    risk = RiskConfig(
        max_long_position_pct=0.20,
        max_short_position_pct=0.12,
        max_sector_gross_pct=0.25,
        max_gross_exposure=1.0,
        max_net_exposure=0.30,
        min_net_exposure=-0.20,
        short_squeeze_5d_return_cap=0.20,
        short_squeeze_ann_vol_cap=0.90,
    )
    ctor = PortfolioConstructor(strat, risk)

    # 1. Empty signals branch respects blacklist, allow_shorting=False, and exposure_multiplier < 1.0
    dials_empty = ControlDials(
        allow_shorting=False,
        exposure_multiplier=0.5,  # max_allowed gross = 0.50
        blacklisted_symbols={"AAPL": as_of + timedelta(hours=12)},
    )
    empty_targets = ctor.compute_target_weights(
        signals=[],
        dials=dials_empty,
        as_of=as_of,
        universe_meta={"AAPL": _meta("AAPL"), "MSFT": _meta("MSFT"), "NVDA": _meta("NVDA")},
        current_weights={"AAPL": 0.20, "MSFT": -0.10, "NVDA": 0.80},
    )
    assert empty_targets["AAPL"] == 0.0  # blacklisted -> zeroed
    assert empty_targets["MSFT"] == 0.0  # short & allow_shorting=False -> zeroed
    assert empty_targets["NVDA"] == pytest.approx(0.50)  # scaled down to max_gross * 0.50

    # 2. Sector cap scaling & max_net_exposure scaling
    sec_meta = {
        "T1": _meta("T1", sector="Tech"),
        "T2": _meta("T2", sector="Tech"),
        "F1": _meta("F1", sector="Finance"),
    }
    sec_sigs = [
        _sig("T1", as_of, composite_score=0.9, atr_pct=0.02),
        _sig("T2", as_of, composite_score=0.8, atr_pct=0.02),
        _sig("F1", as_of, composite_score=0.7, atr_pct=0.02),
    ]
    w_sec = ctor.compute_target_weights(sec_sigs, ControlDials(), as_of, sec_meta)
    assert w_sec["T1"] + w_sec["T2"] <= risk.max_sector_gross_pct + 1e-9
    assert sum(w_sec.values()) <= risk.max_net_exposure + 1e-9

    # 3. Min net exposure scaling when shorts dominate
    short_meta = {
        "S1": _meta("S1", sector="Tech"),
        "S2": _meta("S2", sector="Finance"),
        "S3": _meta("S3", sector="Energy"),
    }
    short_sigs = [
        _sig("S1", as_of, composite_score=-0.9, atr_pct=0.02),
        _sig("S2", as_of, composite_score=-0.8, atr_pct=0.02),
    ]
    w_short_net = ctor.compute_target_weights(short_sigs, ControlDials(), as_of, short_meta)
    assert sum(w_short_net.values()) >= risk.min_net_exposure - 1e-9

    # 4. Squeeze candidate pre-filtering: S_RET (ret_5d >= 0.20) & S_VOL (ann_vol >= 0.90) skipped so S_OK takes slot
    sq_meta = {
        "S_RET": _meta("S_RET", sector="Tech"),
        "S_VOL": _meta("S_VOL", sector="Finance"),
        "S_HTB": _meta("S_HTB", sector="Energy", easy_to_borrow=False),
        "S_OK": _meta("S_OK", sector="Healthcare"),
    }
    sq_sigs = [
        _sig("S_RET", as_of, composite_score=-0.95, ret_5d=0.22),
        _sig("S_VOL", as_of, composite_score=-0.90, ann_vol=0.95),
        _sig("S_HTB", as_of, composite_score=-0.85),
        _sig("S_OK", as_of, composite_score=-0.50),
    ]
    w_sq = ctor.compute_target_weights(sq_sigs, ControlDials(), as_of, sq_meta)
    assert w_sq["S_RET"] == 0.0
    assert w_sq["S_VOL"] == 0.0
    assert w_sq["S_HTB"] == 0.0
    assert w_sq["S_OK"] < 0.0

    # 5. pause_new_entries=True prevents new opens and adds, while allowing trims/exits
    pause_dials = ControlDials(pause_new_entries=True)
    pause_meta = {
        "NEW_LONG": _meta("NEW_LONG", sector="Tech"),
        "CUR_LONG": _meta("CUR_LONG", sector="Finance"),
        "CUR_SHORT": _meta("CUR_SHORT", sector="Energy"),
    }
    pause_sigs = [
        _sig("NEW_LONG", as_of, composite_score=0.9),
        _sig("CUR_LONG", as_of, composite_score=0.9),  # wants larger weight (add)
        _sig("CUR_SHORT", as_of, composite_score=-0.9),  # wants larger short (add)
    ]
    w_paused = ctor.compute_target_weights(
        pause_sigs,
        pause_dials,
        as_of,
        pause_meta,
        current_weights={"CUR_LONG": 0.05, "CUR_SHORT": -0.04},
    )
    assert w_paused["NEW_LONG"] == 0.0
    assert 0.0 <= w_paused["CUR_LONG"] <= 0.05 + 1e-12
    assert -0.04 - 1e-12 <= w_paused["CUR_SHORT"] <= 0.0


# ---------------------------------------------------------------------------
# AlpacaMarketDataProvider direct unit tests
# ---------------------------------------------------------------------------


class _FakeBarClient:
    def __init__(self, payload: dict[str, list[dict[str, object]]]) -> None:
        self.payload = payload
        self.requests: list[object] = []

    def get_stock_bars(self, request: object) -> object:
        self.requests.append(request)
        return SimpleNamespace(data=self.payload)


def test_alpaca_market_normalizes_1d_and_1h_before_filtering() -> None:
    # 1d: Alpaca returns 00:00 ET period-start timestamps.
    # Verify normalizing to 16:00 ET happens BEFORE filtering [start, end]!
    d_prev = datetime(2025, 3, 4, 0, 0, tzinfo=NY_TZ).astimezone(UTC)
    d_in_1 = datetime(2025, 3, 5, 0, 0, tzinfo=NY_TZ).astimezone(UTC)
    d_in_2 = datetime(2025, 3, 6, 0, 0, tzinfo=NY_TZ).astimezone(UTC)
    d_next = datetime(2025, 3, 7, 0, 0, tzinfo=NY_TZ).astimezone(UTC)

    fake_1d = _FakeBarClient(
        {
            "AAPL": [
                {"timestamp": d_prev, "open": 99, "high": 100, "low": 98, "close": 99.5, "volume": 10, "vwap": 99.2},
                {"timestamp": d_in_1, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 20, "vwap": 100.5},
                {"timestamp": d_in_2, "open": 101, "high": 103, "low": 100, "close": 102, "volume": 30, "vwap": -5.0},
                {"timestamp": d_next, "open": 102, "high": 104, "low": 101, "close": 103, "volume": 40, "vwap": 102.5},
            ]
        }
    )
    provider_1d = AlpacaMarketDataProvider("k", "s", client=fake_1d)  # type: ignore[arg-type]
    # start is 2025-03-05 16:00 ET (> 00:00 ET raw bar!). Only post-normalization does d_in_1 match start!
    start_1d = datetime(2025, 3, 5, 16, 0, tzinfo=NY_TZ).astimezone(UTC)
    end_1d = datetime(2025, 3, 6, 16, 0, tzinfo=NY_TZ).astimezone(UTC)

    out_1d = provider_1d.fetch_bars(["aapl"], start=start_1d, end=end_1d, timeframe="1d")
    df_1d = out_1d["AAPL"]
    assert len(df_1d) == 2
    assert df_1d["timestamp"].iloc[0].to_pydatetime() == datetime(2025, 3, 5, 16, 0, tzinfo=NY_TZ).astimezone(UTC)
    assert df_1d["timestamp"].iloc[1].to_pydatetime() == datetime(2025, 3, 6, 16, 0, tzinfo=NY_TZ).astimezone(UTC)
    assert list(df_1d["close"]) == [101.0, 102.0]
    assert math.isnan(float(df_1d["vwap"].iloc[1]))  # negative vwap coerced to NaN cleanly

    # 1h: Alpaca returns 10:00 ET period-start -> normalized to 11:00 ET (+1h) BEFORE filtering
    h_10 = datetime(2025, 3, 5, 10, 0, tzinfo=NY_TZ).astimezone(UTC)
    h_11 = datetime(2025, 3, 5, 11, 0, tzinfo=NY_TZ).astimezone(UTC)
    fake_1h = _FakeBarClient(
        {
            "MSFT": [
                {"timestamp": h_10, "open": 200, "high": 202, "low": 199, "close": 201, "volume": 100, "vwap": 200.5},
                {"timestamp": h_11, "open": 201, "high": 203, "low": 200, "close": 202, "volume": 150, "vwap": 201.5},
            ]
        }
    )
    provider_1h = AlpacaMarketDataProvider("k", "s", client=fake_1h)  # type: ignore[arg-type]
    start_1h = datetime(2025, 3, 5, 11, 0, tzinfo=NY_TZ).astimezone(UTC)
    end_1h = datetime(2025, 3, 5, 11, 30, tzinfo=NY_TZ).astimezone(UTC)

    out_1h = provider_1h.fetch_bars(["MSFT"], start=start_1h, end=end_1h, timeframe="1h")
    df_1h = out_1h["MSFT"]
    assert len(df_1h) == 1
    assert df_1h["timestamp"].iloc[0].to_pydatetime() == datetime(2025, 3, 5, 11, 0, tzinfo=NY_TZ).astimezone(UTC)
    assert float(df_1h["close"].iloc[0]) == 201.0


def test_alpaca_market_rejects_corrupt_ohlcv() -> None:
    ts = datetime(2025, 3, 5, 0, 0, tzinfo=NY_TZ).astimezone(UTC)
    start = datetime(2025, 3, 1, 0, 0, tzinfo=UTC)
    end = datetime(2025, 3, 10, 0, 0, tzinfo=UTC)

    corrupt_cases = [
        {"timestamp": ts, "open": float("nan"), "high": 101, "low": 99, "close": 100, "volume": 10, "vwap": 100},
        {"timestamp": ts, "open": float("inf"), "high": 101, "low": 99, "close": 100, "volume": 10, "vwap": 100},
        {"timestamp": ts, "open": -100, "high": 101, "low": 99, "close": 100, "volume": 10, "vwap": 100},
        {"timestamp": ts, "open": 100, "high": 101, "low": 99, "close": 100, "volume": -1, "vwap": 100},
        {"timestamp": ts, "open": 100, "high": 98, "low": 99, "close": 100, "volume": 10, "vwap": 100},
        {"timestamp": ts, "open": 105, "high": 101, "low": 99, "close": 100, "volume": 10, "vwap": 100},
        {"timestamp": ts, "open": 100, "high": 105, "low": 101, "close": 102, "volume": 10, "vwap": 100},
    ]

    for bad_bar in corrupt_cases:
        fake = _FakeBarClient({"AAPL": [bad_bar]})
        provider = AlpacaMarketDataProvider("k", "s", client=fake)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            provider.fetch_bars(["AAPL"], start=start, end=end, timeframe="1d")
