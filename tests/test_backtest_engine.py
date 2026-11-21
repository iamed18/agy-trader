from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from agy_trader.backtest.engine import BacktestEngine
from agy_trader.core.config import AppConfig, BacktestConfig, RiskConfig, StrategyConfig
from agy_trader.core.models import (
    AssetType,
    OrderSide,
    PositionSide,
    ProposedOrder,
    RiskVerdict,
    SignalRow,
    UniverseSymbol,
)
from agy_trader.data.synthetic import generate_synthetic_dataset
from agy_trader.nlp.sentiment import LexiconSentimentScorer
from agy_trader.storage.db import StateDB
from agy_trader.storage.parquet_store import ParquetBarStore

NY_TZ = ZoneInfo("America/New_York")


def _build_ohlc_df(
    dates: list[date],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
) -> pd.DataFrame:
    timestamps = [
        datetime(d.year, d.month, d.day, 16, 0, tzinfo=NY_TZ).astimezone(UTC) for d in dates
    ]
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(timestamps, tz="UTC"),
            "open": np.asarray(opens, dtype=float),
            "high": np.asarray(highs, dtype=float),
            "low": np.asarray(lows, dtype=float),
            "close": np.asarray(closes, dtype=float),
            "volume": np.full(len(dates), 5_000_000.0),
            "vwap": np.asarray(closes, dtype=float),
        }
    )


def _dummy_signal(symbol: str, as_of: datetime, atr_pct: float = 0.02) -> SignalRow:
    return SignalRow(
        as_of=as_of,
        symbol=symbol,
        trend_score=0.0,
        pullback_score=0.0,
        news_score=0.0,
        news_velocity_z=0.0,
        composite_score=0.0,
        macro_regime_scale=1.0,
        atr_pct=atr_pct,
        ann_vol=0.25,
        ret_5d=0.0,
    )


# ---------------------------------------------------------------------------
# End-to-end synthetic backtest, 4-way ablation, and walk-forward reproducibility
# ---------------------------------------------------------------------------


def test_end_to_end_synthetic_ablation_and_walk_forward_deterministic(tmp_path) -> None:
    universe = [
        UniverseSymbol(symbol="SPY", asset_type=AssetType.etf, sector="Broad Market"),
        UniverseSymbol(symbol="AAPL", asset_type=AssetType.equity, sector="Technology"),
        UniverseSymbol(symbol="MSFT", asset_type=AssetType.equity, sector="Technology"),
        UniverseSymbol(symbol="JPM", asset_type=AssetType.equity, sector="Financials"),
        UniverseSymbol(symbol="XOM", asset_type=AssetType.equity, sector="Energy"),
        UniverseSymbol(symbol="TLT", asset_type=AssetType.etf, sector="Fixed Income"),
    ]
    store = ParquetBarStore(tmp_path / "bars")
    db = StateDB(tmp_path / "state.db")
    generate_synthetic_dataset(
        universe, store, db, days=180, end_date=date(2025, 10, 31), seed=2025
    )

    # Score synthetic news deterministically
    scorer = LexiconSentimentScorer()
    unscored = db.get_unscored_news(limit=5000)
    preds = scorer.score_articles(unscored)
    db.update_news_scores(
        [
            (art.article_id, pred.score, pred.label, pred.confidence, pred.scorer_model)
            for art, pred in zip(unscored, preds, strict=True)
        ]
    )

    bars = store.load_universe_bars([u.symbol for u in universe])
    start_utc = datetime(2024, 1, 1, tzinfo=UTC)
    end_utc = datetime(2026, 1, 1, tzinfo=UTC)
    news_by_symbol = {
        u.symbol: db.get_news_for_symbol_window(u.symbol, start_utc, end_utc) for u in universe
    }
    macro_points = [
        pt
        for sid in ("VIXCLS", "BAMLH0A0HYM2", "T10Y2Y")
        for pt in db.get_macro_series(sid)
    ]
    earnings_map = db.get_all_earnings_map()

    cfg = AppConfig(
        strategy=StrategyConfig(min_signal_threshold=0.25),
        risk=RiskConfig(),
        backtest=BacktestConfig(),
    )
    engine = BacktestEngine(cfg, universe)

    ablation_1 = engine.run_ablation(
        bars,
        news_by_symbol=news_by_symbol,
        macro_points=macro_points,
        earnings_by_symbol=earnings_map,
    )
    ablation_2 = engine.run_ablation(
        bars,
        news_by_symbol=news_by_symbol,
        macro_points=macro_points,
        earnings_by_symbol=earnings_map,
    )

    assert len(ablation_1) == 4
    assert [r.metrics.label for r in ablation_1] == [
        "Price+Vol Only (Long-Only)",
        "Price+Vol + News (Long-Only)",
        "Price+Vol Only (Long/Short)",
        "Price+Vol + News (Long/Short)",
    ]

    for r1, r2 in zip(ablation_1, ablation_2, strict=True):
        assert r1.metrics == r2.metrics
        assert r1.equity_curve == r2.equity_curve
        assert r1.trades == r2.trades
        assert r1.metrics.trading_days > 100
        assert r1.metrics.total_orders_filled > 0

        # Unmocked integration invariants on every EquityPoint
        for pt in r1.equity_curve:
            assert pt.gross_exposure <= cfg.risk.max_gross_exposure + 1e-3
            assert cfg.risk.min_net_exposure - 1e-3 <= pt.net_exposure <= cfg.risk.max_net_exposure + 1e-3
            assert pt.cash >= -1e-2

    # Long-Only variants have ZERO short trades and net_exposure >= 0
    for lo_res in ablation_1[:2]:
        assert all(t.side == "long" for t in lo_res.trades)
        assert all(pt.net_exposure >= -1e-6 for pt in lo_res.equity_curve)

    # Long/Short + News executes both long & short trades or differs measurably from Price-Only
    ls_price, ls_news = ablation_1[2], ablation_1[3]
    has_long_and_short = any(t.side == "long" for t in ls_news.trades) and any(
        t.side == "short" for t in ls_news.trades
    )
    differs_from_price_only = (
        ls_news.metrics.total_return_pct != ls_price.metrics.total_return_pct
        or ls_news.trades != ls_price.trades
    )
    assert has_long_and_short or differs_from_price_only

    wf_1 = engine.run_walk_forward(
        bars,
        news_by_symbol=news_by_symbol,
        macro_points=macro_points,
        earnings_by_symbol=earnings_map,
        n_folds=3,
    )
    wf_2 = engine.run_walk_forward(
        bars,
        news_by_symbol=news_by_symbol,
        macro_points=macro_points,
        earnings_by_symbol=earnings_map,
        n_folds=3,
    )
    assert len(wf_1) == 3
    assert wf_1 == wf_2
    for i in range(len(wf_1) - 1):
        assert wf_1[i].trading_days > 0
        assert wf_1[i].end_date < wf_1[i + 1].start_date
    assert wf_1[-1].trading_days > 0


# ---------------------------------------------------------------------------
# Exact accounting & execution realism verification tests
# ---------------------------------------------------------------------------


def _craft_constant_warmup_bars(n_warmup: int = 65) -> tuple[list[date], list[float], list[float], list[float], list[float]]:
    # 65 warmup days + 4 active days = 69 business days
    dates = [ts.date() for ts in pd.bdate_range(start=date(2025, 1, 2), periods=n_warmup + 4)]
    opens = [100.0] * len(dates)
    highs = [101.0] * len(dates)
    lows = [99.0] * len(dates)
    closes = [100.0] * len(dates)
    return dates, opens, highs, lows, closes


def test_long_trade_exact_slippage_sec_fee_and_equity_conservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates, opens, highs, lows, closes = _craft_constant_warmup_bars()
    # Active bars indices: 65 (t0), 66 (t1), 67 (t2), 68 (t3)
    # Signal at t0 close -> Buy at t1 open=100.0
    # Signal at t1 close -> Sell at t2 open=110.0
    opens[66], highs[66], lows[66], closes[66] = 100.0, 105.0, 99.0, 104.0
    opens[67], highs[67], lows[67], closes[67] = 110.0, 112.0, 109.0, 111.0
    opens[68], highs[68], lows[68], closes[68] = 111.0, 112.0, 110.0, 111.0

    bars = {"AAPL": _build_ohlc_df(dates, opens, highs, lows, closes)}
    universe = [UniverseSymbol(symbol="AAPL", asset_type=AssetType.equity, sector="Technology")]
    cfg = AppConfig(
        risk=RiskConfig(max_long_position_pct=0.25, max_short_position_pct=0.25),
        backtest=BacktestConfig(
            initial_capital=10_000.0,
            equity_slippage_bps=10.0,  # 0.001
            sec_sell_fee_bps=0.28,  # 0.000028
            short_borrow_apr=0.0,
            benchmark_symbol="AAPL",
        ),
    )
    engine = BacktestEngine(cfg, universe)

    t0_ts = datetime(dates[65].year, dates[65].month, dates[65].day, 16, 0, tzinfo=NY_TZ).astimezone(UTC)
    t1_ts = datetime(dates[66].year, dates[66].month, dates[66].day, 16, 0, tzinfo=NY_TZ).astimezone(UTC)

    monkeypatch.setattr(
        "agy_trader.backtest.engine.FeatureEngine.build_signals",
        lambda self, as_of, **kwargs: [_dummy_signal("AAPL", as_of)],
    )

    def fake_propose(self, signals, current_positions, prices, equity, dials, as_of, universe_meta):
        if as_of == t0_ts:
            return [
                ProposedOrder(
                    symbol="AAPL",
                    side=OrderSide.buy,
                    qty=10.0,
                    target_side=PositionSide.long,
                    is_risk_reducing=False,
                    reference_price=prices["AAPL"],
                    stop_loss_price=50.0,
                    reason="test_long_open",
                )
            ]
        if as_of == t1_ts:
            return [
                ProposedOrder(
                    symbol="AAPL",
                    side=OrderSide.sell,
                    qty=10.0,
                    is_risk_reducing=True,
                    reference_price=prices["AAPL"],
                    reason="test_long_close",
                )
            ]
        return []

    def fake_eval(self, orders, state, dials, universe_meta, signals_by_symbol=None, earnings_by_symbol=None):
        return [
            RiskVerdict(
                approved=True,
                order=o,
                adjusted_qty=o.qty,
                stop_loss_price=o.stop_loss_price,
            )
            for o in orders
        ]

    monkeypatch.setattr("agy_trader.backtest.engine.PortfolioConstructor.build_proposed_orders", fake_propose)
    monkeypatch.setattr("agy_trader.backtest.engine.HardRiskEngine.evaluate_batch", fake_eval)

    res = engine.run(bars, use_news=False)
    assert len(res.trades) == 1
    trade = res.trades[0]

    slip = 0.001
    buy_fill = 100.0 * (1.0 + slip)  # 100.10
    sell_fill = 110.0 * (1.0 - slip)  # 109.89
    sec_fee = 10.0 * sell_fill * (0.28 / 10_000.0)
    gross_pnl = (sell_fill - buy_fill) * 10.0
    net_pnl = gross_pnl - sec_fee

    assert trade.side == "long"
    assert trade.entry_price == pytest.approx(buy_fill, rel=1e-6)
    assert trade.exit_price == pytest.approx(sell_fill, rel=1e-6)
    assert trade.gross_pnl == pytest.approx(gross_pnl, abs=1e-3)
    assert trade.net_pnl == pytest.approx(net_pnl, abs=1e-3)

    expected_equity = 10_000.0 + net_pnl
    last_pt = res.equity_curve[-1]
    assert last_pt.equity == pytest.approx(last_pt.cash, abs=0.02)  # flat at end -> mtm == 0
    assert res.metrics.final_equity == pytest.approx(expected_equity, abs=0.02)
    assert res.metrics.total_sec_fees_usd == pytest.approx(sec_fee, abs=0.02)


def test_short_trade_exact_slippage_and_daily_borrow_cost_accrual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates, opens, highs, lows, closes = _craft_constant_warmup_bars()
    # Signal at t0 close -> Short 10 at t1 open=100.0, t1 close=102.0
    # Signal at t1 close -> Cover 10 at t2 open=95.0 (overnight borrow accrues on t1->t2!)
    opens[66], highs[66], lows[66], closes[66] = 100.0, 103.0, 99.0, 102.0
    opens[67], highs[67], lows[67], closes[67] = 95.0, 96.0, 94.0, 95.0
    opens[68], highs[68], lows[68], closes[68] = 95.0, 96.0, 94.0, 95.0

    bars = {"AAPL": _build_ohlc_df(dates, opens, highs, lows, closes)}
    universe = [UniverseSymbol(symbol="AAPL", asset_type=AssetType.equity, sector="Technology")]
    apr = 0.0365  # 3.65% APR -> exact clean daily rate 0.0001/day
    cfg = AppConfig(
        risk=RiskConfig(max_long_position_pct=0.25, max_short_position_pct=0.25),
        backtest=BacktestConfig(
            initial_capital=10_000.0,
            equity_slippage_bps=10.0,  # 0.001
            sec_sell_fee_bps=0.28,
            short_borrow_apr=apr,
            benchmark_symbol="AAPL",
        ),
    )
    engine = BacktestEngine(cfg, universe)

    t0_ts = datetime(dates[65].year, dates[65].month, dates[65].day, 16, 0, tzinfo=NY_TZ).astimezone(UTC)
    t1_ts = datetime(dates[66].year, dates[66].month, dates[66].day, 16, 0, tzinfo=NY_TZ).astimezone(UTC)

    monkeypatch.setattr(
        "agy_trader.backtest.engine.FeatureEngine.build_signals",
        lambda self, as_of, **kwargs: [_dummy_signal("AAPL", as_of)],
    )

    def fake_propose(self, signals, current_positions, prices, equity, dials, as_of, universe_meta):
        if as_of == t0_ts:
            return [
                ProposedOrder(
                    symbol="AAPL",
                    side=OrderSide.sell,
                    qty=10.0,
                    target_side=PositionSide.short,
                    is_risk_reducing=False,
                    reference_price=prices["AAPL"],
                    stop_loss_price=150.0,
                    reason="test_short_open",
                )
            ]
        if as_of == t1_ts:
            return [
                ProposedOrder(
                    symbol="AAPL",
                    side=OrderSide.buy,
                    qty=10.0,
                    is_risk_reducing=True,
                    reference_price=prices["AAPL"],
                    reason="test_short_cover",
                )
            ]
        return []

    def fake_eval(self, orders, state, dials, universe_meta, signals_by_symbol=None, earnings_by_symbol=None):
        return [
            RiskVerdict(
                approved=True,
                order=o,
                adjusted_qty=o.qty,
                stop_loss_price=o.stop_loss_price,
            )
            for o in orders
        ]

    monkeypatch.setattr("agy_trader.backtest.engine.PortfolioConstructor.build_proposed_orders", fake_propose)
    monkeypatch.setattr("agy_trader.backtest.engine.HardRiskEngine.evaluate_batch", fake_eval)

    res = engine.run(bars, use_news=False)
    assert len(res.trades) == 1
    trade = res.trades[0]

    slip = 0.001
    short_entry = 100.0 * (1.0 - slip)  # 99.90
    cover_fill = 95.0 * (1.0 + slip)  # 95.095
    entry_sec_fee = 10.0 * short_entry * (0.28 / 10_000.0)
    cal_days = max(1, (dates[67] - dates[66]).days)
    borrow_cost = 10.0 * closes[66] * apr * (cal_days / 365.0)
    gross_pnl = (short_entry - cover_fill) * 10.0
    net_pnl = gross_pnl - entry_sec_fee - borrow_cost

    assert trade.side == "short"
    assert trade.entry_price == pytest.approx(short_entry, rel=1e-6)
    assert trade.exit_price == pytest.approx(cover_fill, rel=1e-6)
    assert trade.gross_pnl == pytest.approx(gross_pnl, abs=1e-3)
    assert trade.net_pnl == pytest.approx(net_pnl, abs=1e-3)
    assert res.metrics.total_borrow_cost_usd > 0.0
    assert res.metrics.total_borrow_cost_usd == pytest.approx(borrow_cost, abs=0.01)
    assert res.metrics.final_equity == pytest.approx(10_000.0 + net_pnl, abs=0.02)


def test_gap_through_stop_loss_long_and_short(monkeypatch: pytest.MonkeyPatch) -> None:
    dates, opens, highs, lows, closes = _craft_constant_warmup_bars()
    # Long AAPL: open at t1 open=100 with stop=95 (5% distance), t2 gaps down to open=88 -> fills at 88*(1-slip)
    # Short MSFT: open at t1 open=100 with stop=105 (5% distance), t2 gaps up to open=112 -> fills at 112*(1+slip)
    aapl_o, aapl_h, aapl_l, aapl_c = list(opens), list(highs), list(lows), list(closes)
    msft_o, msft_h, msft_l, msft_c = list(opens), list(highs), list(lows), list(closes)

    aapl_o[66], aapl_h[66], aapl_l[66], aapl_c[66] = 100.0, 101.0, 99.0, 100.0
    aapl_o[67], aapl_h[67], aapl_l[67], aapl_c[67] = 88.0, 89.0, 87.0, 88.5

    msft_o[66], msft_h[66], msft_l[66], msft_c[66] = 100.0, 101.0, 99.0, 100.0
    msft_o[67], msft_h[67], msft_l[67], msft_c[67] = 112.0, 113.0, 111.0, 111.5

    bars = {
        "AAPL": _build_ohlc_df(dates, aapl_o, aapl_h, aapl_l, aapl_c),
        "MSFT": _build_ohlc_df(dates, msft_o, msft_h, msft_l, msft_c),
    }
    universe = [
        UniverseSymbol(symbol="AAPL", asset_type=AssetType.equity, sector="Technology"),
        UniverseSymbol(symbol="MSFT", asset_type=AssetType.equity, sector="Technology"),
    ]
    cfg = AppConfig(
        backtest=BacktestConfig(
            initial_capital=50_000.0,
            equity_slippage_bps=10.0,  # 0.001
            sec_sell_fee_bps=0.28,
            short_borrow_apr=0.0,
            benchmark_symbol="AAPL",
        )
    )
    engine = BacktestEngine(cfg, universe)
    t0_ts = datetime(dates[65].year, dates[65].month, dates[65].day, 16, 0, tzinfo=NY_TZ).astimezone(UTC)

    monkeypatch.setattr(
        "agy_trader.backtest.engine.FeatureEngine.build_signals",
        lambda self, as_of, **kwargs: [_dummy_signal("AAPL", as_of), _dummy_signal("MSFT", as_of)],
    )

    def fake_propose(self, signals, current_positions, prices, equity, dials, as_of, universe_meta):
        if as_of == t0_ts:
            return [
                ProposedOrder(
                    symbol="AAPL",
                    side=OrderSide.buy,
                    qty=10.0,
                    target_side=PositionSide.long,
                    is_risk_reducing=False,
                    reference_price=100.0,
                    stop_loss_price=95.0,
                    reason="open_long",
                ),
                ProposedOrder(
                    symbol="MSFT",
                    side=OrderSide.sell,
                    qty=10.0,
                    target_side=PositionSide.short,
                    is_risk_reducing=False,
                    reference_price=100.0,
                    stop_loss_price=105.0,
                    reason="open_short",
                ),
            ]
        return []

    def fake_eval(self, orders, state, dials, universe_meta, signals_by_symbol=None, earnings_by_symbol=None):
        return [
            RiskVerdict(
                approved=True,
                order=o,
                adjusted_qty=o.qty,
                stop_loss_price=o.stop_loss_price,
            )
            for o in orders
        ]

    monkeypatch.setattr("agy_trader.backtest.engine.PortfolioConstructor.build_proposed_orders", fake_propose)
    monkeypatch.setattr("agy_trader.backtest.engine.HardRiskEngine.evaluate_batch", fake_eval)

    res = engine.run(bars, use_news=False)
    by_sym = {t.symbol: t for t in res.trades}
    assert set(by_sym.keys()) == {"AAPL", "MSFT"}

    slip = 0.001
    assert by_sym["AAPL"].exit_reason == "stop_loss_open_gap"
    assert by_sym["AAPL"].exit_price == pytest.approx(88.0 * (1.0 - slip), rel=1e-6)
    assert by_sym["MSFT"].exit_reason == "stop_loss_open_gap"
    assert by_sym["MSFT"].exit_price == pytest.approx(112.0 * (1.0 + slip), rel=1e-6)


def test_intraday_stop_loss_fill_and_zero_same_bar_lookahead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates, opens, highs, lows, closes = _craft_constant_warmup_bars()
    # t0 (65): signal generated at t0 close (close=100.0)
    # t1 (66): buy fills at t1 open=102.0 (proves bar t0 signal executes at t1 open, NOT t0 close!),
    #          stop re-anchored to fill_px=102.102 * (1 - 0.05) = 96.9969
    # t2 (67): opens at 99.0 (> stop), dips intraday to low=92.0 (<= stop) -> fills at stop * (1 - slip)!
    opens[65], highs[65], lows[65], closes[65] = 100.0, 100.5, 99.5, 100.0
    opens[66], highs[66], lows[66], closes[66] = 102.0, 103.0, 101.0, 102.5
    opens[67], highs[67], lows[67], closes[67] = 99.0, 100.0, 92.0, 94.0

    bars = {"AAPL": _build_ohlc_df(dates, opens, highs, lows, closes)}
    universe = [UniverseSymbol(symbol="AAPL", asset_type=AssetType.equity, sector="Technology")]
    cfg = AppConfig(
        backtest=BacktestConfig(
            initial_capital=20_000.0,
            equity_slippage_bps=10.0,  # 0.001
            sec_sell_fee_bps=0.28,
            short_borrow_apr=0.0,
            benchmark_symbol="AAPL",
        )
    )
    engine = BacktestEngine(cfg, universe)

    t0_ts = datetime(dates[65].year, dates[65].month, dates[65].day, 16, 0, tzinfo=NY_TZ).astimezone(UTC)
    t1_ts = datetime(dates[66].year, dates[66].month, dates[66].day, 16, 0, tzinfo=NY_TZ).astimezone(UTC)
    t2_ts = datetime(dates[67].year, dates[67].month, dates[67].day, 16, 0, tzinfo=NY_TZ).astimezone(UTC)

    observed_signal_bars: list[tuple[datetime, datetime]] = []

    def spy_build_signals(self, as_of, bars_by_symbol, **kwargs):
        # Verify FeatureEngine never receives any bar with timestamp > as_of
        max_bar_ts = bars_by_symbol["AAPL"]["timestamp"].max().to_pydatetime()
        observed_signal_bars.append((as_of, max_bar_ts))
        assert max_bar_ts <= as_of
        return [_dummy_signal("AAPL", as_of)]

    monkeypatch.setattr("agy_trader.backtest.engine.FeatureEngine.build_signals", spy_build_signals)

    def fake_propose(self, signals, current_positions, prices, equity, dials, as_of, universe_meta):
        if as_of == t0_ts:
            return [
                ProposedOrder(
                    symbol="AAPL",
                    side=OrderSide.buy,
                    qty=10.0,
                    target_side=PositionSide.long,
                    is_risk_reducing=False,
                    reference_price=100.0,
                    stop_loss_price=95.0,  # 5% distance from t0 close 100.0
                    reason="open_long",
                )
            ]
        return []

    def fake_eval(self, orders, state, dials, universe_meta, signals_by_symbol=None, earnings_by_symbol=None):
        return [
            RiskVerdict(
                approved=True,
                order=o,
                adjusted_qty=o.qty,
                stop_loss_price=o.stop_loss_price,
            )
            for o in orders
        ]

    monkeypatch.setattr("agy_trader.backtest.engine.PortfolioConstructor.build_proposed_orders", fake_propose)
    monkeypatch.setattr("agy_trader.backtest.engine.HardRiskEngine.evaluate_batch", fake_eval)

    res = engine.run(bars, use_news=False)
    assert len(observed_signal_bars) >= 3
    assert len(res.trades) == 1
    trade = res.trades[0]

    slip = 0.001
    expected_entry = 102.0 * (1.0 + slip)  # t1 open, NOT t0 close (100.0)
    reanchored_stop = round(expected_entry * (1.0 - 0.05), 4)
    expected_exit = reanchored_stop * (1.0 - slip)

    assert trade.entry_time == t1_ts
    assert trade.exit_time == t2_ts
    assert trade.entry_price == pytest.approx(expected_entry, rel=1e-6)
    assert trade.exit_reason == "stop_loss_intraday"
    assert trade.exit_price == pytest.approx(expected_exit, rel=1e-6)

    # At t0 close, no position was filled yet -> cash == initial_capital, gross_exposure == 0
    assert res.equity_curve[0].timestamp == t0_ts
    assert res.equity_curve[0].gross_exposure == 0.0
    assert res.equity_curve[0].cash == 20_000.0


def test_position_add_partial_close_pro_rata_borrow_sec_and_close_at_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 65 warmup + 6 active days (indices 65..70)
    dates, opens, highs, lows, closes = _craft_constant_warmup_bars(n_warmup=65)
    for _ in range(2):
        dates.append((pd.Timestamp(dates[-1]) + pd.offsets.BDay(1)).date())
        opens.append(100.0)
        highs.append(101.0)
        lows.append(99.0)
        closes.append(100.0)

    # Short AAPL:
    # t0 (65): signal -> open short 10 at t1 open=100.0, t1 close=104.0
    # t1 (66): signal -> add short 10 at t2 open=108.0, t2 close=106.0 (same_sign!)
    # t2 (67): signal -> partial cover 8 at t3 open=96.0 (40% of 20), t3 close=96.0
    # t4 (69): end of window -> close remaining 12 at t4 close=92.0 via close_open_positions_at_end=True
    aapl_o, aapl_h, aapl_l, aapl_c = list(opens), list(highs), list(lows), list(closes)
    aapl_o[66], aapl_h[66], aapl_l[66], aapl_c[66] = 100.0, 105.0, 99.0, 104.0
    aapl_o[67], aapl_h[67], aapl_l[67], aapl_c[67] = 108.0, 109.0, 105.0, 106.0
    aapl_o[68], aapl_h[68], aapl_l[68], aapl_c[68] = 96.0, 98.0, 95.0, 96.0
    aapl_o[69], aapl_h[69], aapl_l[69], aapl_c[69] = 96.0, 97.0, 94.0, 95.0
    aapl_o[70], aapl_h[70], aapl_l[70], aapl_c[70] = 94.0, 95.0, 91.0, 92.0

    # Long MSFT:
    # t0 (65): open long 10 at t1 open=100.0
    # t1 (66): add long 10 at t2 open=110.0 (same_sign!)
    # t2 (67): partial close 5 at t3 open=115.0 (25% of 20)
    # t4 (70): end of window -> close remaining 15 at t4 close=120.0
    msft_o, msft_h, msft_l, msft_c = list(opens), list(highs), list(lows), list(closes)
    msft_o[66], msft_h[66], msft_l[66], msft_c[66] = 100.0, 102.0, 99.0, 101.0
    msft_o[67], msft_h[67], msft_l[67], msft_c[67] = 110.0, 112.0, 109.0, 111.0
    msft_o[68], msft_h[68], msft_l[68], msft_c[68] = 115.0, 117.0, 114.0, 116.0
    msft_o[69], msft_h[69], msft_l[69], msft_c[69] = 116.0, 118.0, 115.0, 117.0
    msft_o[70], msft_h[70], msft_l[70], msft_c[70] = 118.0, 121.0, 117.0, 120.0

    bars = {
        "AAPL": _build_ohlc_df(dates, aapl_o, aapl_h, aapl_l, aapl_c),
        "MSFT": _build_ohlc_df(dates, msft_o, msft_h, msft_l, msft_c),
    }
    universe = [
        UniverseSymbol(symbol="AAPL", asset_type=AssetType.equity, sector="Technology"),
        UniverseSymbol(symbol="MSFT", asset_type=AssetType.equity, sector="Technology"),
    ]
    initial_capital = 50_000.0
    apr = 0.0365
    sec_bps = 10.0  # non-trivial 10 bps SEC fee to verify exact pro-rata attribution
    cfg = AppConfig(
        risk=RiskConfig(max_long_position_pct=0.25, max_short_position_pct=0.25),
        backtest=BacktestConfig(
            initial_capital=initial_capital,
            equity_slippage_bps=10.0,
            sec_sell_fee_bps=sec_bps,
            short_borrow_apr=apr,
            benchmark_symbol="MSFT",
        ),
    )
    engine = BacktestEngine(cfg, universe)

    def _ts(idx: int) -> datetime:
        d = dates[idx]
        return datetime(d.year, d.month, d.day, 16, 0, tzinfo=NY_TZ).astimezone(UTC)

    t0, t1, t2 = _ts(65), _ts(66), _ts(67)

    monkeypatch.setattr(
        "agy_trader.backtest.engine.FeatureEngine.build_signals",
        lambda self, as_of, **kwargs: [_dummy_signal("AAPL", as_of), _dummy_signal("MSFT", as_of)],
    )

    def fake_propose(self, signals, current_positions, prices, equity, dials, as_of, universe_meta):
        if as_of == t0:
            return [
                ProposedOrder(
                    symbol="AAPL",
                    side=OrderSide.sell,
                    qty=10.0,
                    target_side=PositionSide.short,
                    is_risk_reducing=False,
                    reference_price=prices["AAPL"],
                    stop_loss_price=200.0,
                    reason="short_open",
                ),
                ProposedOrder(
                    symbol="MSFT",
                    side=OrderSide.buy,
                    qty=10.0,
                    target_side=PositionSide.long,
                    is_risk_reducing=False,
                    reference_price=prices["MSFT"],
                    stop_loss_price=50.0,
                    reason="long_open",
                ),
            ]
        if as_of == t1:
            return [
                ProposedOrder(
                    symbol="AAPL",
                    side=OrderSide.sell,
                    qty=10.0,
                    target_side=PositionSide.short,
                    is_risk_reducing=False,
                    reference_price=prices["AAPL"],
                    stop_loss_price=200.0,
                    reason="short_add",
                ),
                ProposedOrder(
                    symbol="MSFT",
                    side=OrderSide.buy,
                    qty=10.0,
                    target_side=PositionSide.long,
                    is_risk_reducing=False,
                    reference_price=prices["MSFT"],
                    stop_loss_price=50.0,
                    reason="long_add",
                ),
            ]
        if as_of == t2:
            return [
                ProposedOrder(
                    symbol="AAPL",
                    side=OrderSide.buy,
                    qty=8.0,
                    is_risk_reducing=True,
                    reference_price=prices["AAPL"],
                    reason="short_partial_cover",
                ),
                ProposedOrder(
                    symbol="MSFT",
                    side=OrderSide.sell,
                    qty=5.0,
                    is_risk_reducing=True,
                    reference_price=prices["MSFT"],
                    reason="long_partial_trim",
                ),
            ]
        return []

    def fake_eval(self, orders, state, dials, universe_meta, signals_by_symbol=None, earnings_by_symbol=None):
        return [
            RiskVerdict(
                approved=True,
                order=o,
                adjusted_qty=o.qty,
                stop_loss_price=o.stop_loss_price,
            )
            for o in orders
        ]

    monkeypatch.setattr("agy_trader.backtest.engine.PortfolioConstructor.build_proposed_orders", fake_propose)
    monkeypatch.setattr("agy_trader.backtest.engine.HardRiskEngine.evaluate_batch", fake_eval)

    res = engine.run(bars, use_news=False, close_open_positions_at_end=True)

    aapl_trades = [t for t in res.trades if t.symbol == "AAPL"]
    msft_trades = [t for t in res.trades if t.symbol == "MSFT"]
    assert len(aapl_trades) == 2
    assert len(msft_trades) == 2

    # Hand-verify pro-rata borrow & SEC attribution on AAPL short partial cover (8 of 20 shares = 40%)
    slip = 0.001
    sec_rate = sec_bps / 10_000.0
    s1_fill = 100.0 * (1.0 - slip)
    s2_fill = 108.0 * (1.0 - slip)
    avg_short_entry = (10.0 * s1_fill + 10.0 * s2_fill) / 20.0
    sec1 = 10.0 * s1_fill * sec_rate
    sec2 = 10.0 * s2_fill * sec_rate
    b1 = 10.0 * aapl_c[66] * apr * ((dates[67] - dates[66]).days / 365.0)
    b2 = 20.0 * aapl_c[67] * apr * ((dates[68] - dates[67]).days / 365.0)

    partial_cover_fill = 96.0 * (1.0 + slip)
    exp_partial_gross = (avg_short_entry - partial_cover_fill) * 8.0
    exp_partial_net = exp_partial_gross - 0.4 * (sec1 + sec2) - 0.4 * (b1 + b2)

    t_aapl_part = aapl_trades[0]
    assert t_aapl_part.exit_reason == "short_partial_cover"
    assert t_aapl_part.qty == 8.0
    assert t_aapl_part.entry_price == pytest.approx(avg_short_entry, rel=1e-6)
    assert t_aapl_part.gross_pnl == pytest.approx(exp_partial_gross, abs=1e-3)
    assert t_aapl_part.net_pnl == pytest.approx(exp_partial_net, abs=1e-3)

    # Verify end-of-window liquidation reconciled exact net PnL vs equity change
    assert aapl_trades[1].exit_reason == "end_of_window"
    assert aapl_trades[1].qty == 12.0
    assert msft_trades[0].qty == 5.0
    assert msft_trades[1].exit_reason == "end_of_window"
    assert msft_trades[1].qty == 15.0

    total_net_pnl = sum(t.net_pnl for t in res.trades)
    assert abs(total_net_pnl - (res.metrics.final_equity - initial_capital)) < 0.05


def test_post_gap_entry_abort_carried_exit_and_cash_exposure_conservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 65 warmup + 5 active days (indices 65..69)
    dates, opens, highs, lows, closes = _craft_constant_warmup_bars(n_warmup=65)
    dates.append((pd.Timestamp(dates[-1]) + pd.offsets.BDay(1)).date())
    opens.append(100.0)
    highs.append(101.0)
    lows.append(99.0)
    closes.append(100.0)

    # AAPL: held long from t0? No, open long 5 shares at t1 open=100.0, t1 close=100.0.
    # At t1 close, Pass 1 exit queued for t2 open, EXCEPT AAPL has NO bar on t2 (dates[67])!
    # So Pass 1 exit must carry forward to t3 (dates[68]) open=105.0!
    aapl_dates = [d for i, d in enumerate(dates) if i != 67]
    aapl_o = [v for i, v in enumerate(opens) if i != 67]
    aapl_h = [v for i, v in enumerate(highs) if i != 67]
    aapl_l = [v for i, v in enumerate(lows) if i != 67]
    aapl_c = [v for i, v in enumerate(closes) if i != 67]
    # Note: index 67 in aapl_* corresponds to dates[68] (t3)!
    aapl_o[67], aapl_h[67], aapl_l[67], aapl_c[67] = 105.0, 106.0, 104.0, 105.0

    # MSFT: queued Pass 2 long at t0 close with stop=95.0; t1 (66) gaps down to open=90.0 <= 95.0 -> CANCELLED!
    msft_o, msft_h, msft_l, msft_c = list(opens), list(highs), list(lows), list(closes)
    msft_o[66], msft_h[66], msft_l[66], msft_c[66] = 90.0, 91.0, 89.0, 90.5

    # NVDA: existing long 45 shares at $100 (worth $4,500 out of $10,000 initial equity) drops at t2 open to $60
    # (reducing open_equity to $8,200, cash=$5,500) while a large Pass 2 buy on ORCL ($6,000) is queued at t1 close!
    # Live open sizing must shrink ORCL so cash never goes < 0 and gross_exposure <= max_gross_exposure!
    nvda_o, nvda_h, nvda_l, nvda_c = list(opens), list(highs), list(lows), list(closes)
    nvda_o[67], nvda_h[67], nvda_l[67], nvda_c[67] = 60.0, 62.0, 59.0, 60.0
    nvda_o[68], nvda_h[68], nvda_l[68], nvda_c[68] = 60.0, 62.0, 59.0, 60.0
    nvda_o[69], nvda_h[69], nvda_l[69], nvda_c[69] = 60.0, 62.0, 59.0, 60.0

    orcl_o, orcl_h, orcl_l, orcl_c = list(opens), list(highs), list(lows), list(closes)

    bars = {
        "AAPL": _build_ohlc_df(aapl_dates, aapl_o, aapl_h, aapl_l, aapl_c),
        "MSFT": _build_ohlc_df(dates, msft_o, msft_h, msft_l, msft_c),
        "NVDA": _build_ohlc_df(dates, nvda_o, nvda_h, nvda_l, nvda_c),
        "ORCL": _build_ohlc_df(dates, orcl_o, orcl_h, orcl_l, orcl_c),
    }
    universe = [
        UniverseSymbol(symbol="AAPL", asset_type=AssetType.equity, sector="Technology"),
        UniverseSymbol(symbol="MSFT", asset_type=AssetType.equity, sector="Finance"),
        UniverseSymbol(symbol="NVDA", asset_type=AssetType.equity, sector="Energy"),
        UniverseSymbol(symbol="ORCL", asset_type=AssetType.equity, sector="Healthcare"),
    ]
    cfg = AppConfig(
        risk=RiskConfig(
            max_long_position_pct=0.60,
            max_sector_gross_pct=0.80,
            max_gross_exposure=0.80,
            max_net_exposure=0.80,
        ),
        backtest=BacktestConfig(
            initial_capital=10_000.0,
            equity_slippage_bps=10.0,
            short_borrow_apr=0.0,
            benchmark_symbol="ORCL",
        ),
    )
    engine = BacktestEngine(cfg, universe)

    def _ts(idx: int) -> datetime:
        d = dates[idx]
        return datetime(d.year, d.month, d.day, 16, 0, tzinfo=NY_TZ).astimezone(UTC)

    t0, t1, t3 = _ts(65), _ts(66), _ts(68)

    monkeypatch.setattr(
        "agy_trader.backtest.engine.FeatureEngine.build_signals",
        lambda self, as_of, **kwargs: [_dummy_signal(s, as_of) for s in ("AAPL", "MSFT", "NVDA", "ORCL")],
    )

    def fake_propose(self, signals, current_positions, prices, equity, dials, as_of, universe_meta):
        if as_of == t0:
            return [
                ProposedOrder(
                    symbol="AAPL",
                    side=OrderSide.buy,
                    qty=5.0,
                    target_side=PositionSide.long,
                    is_risk_reducing=False,
                    reference_price=100.0,
                    stop_loss_price=80.0,
                    reason="open_aapl",
                ),
                ProposedOrder(
                    symbol="MSFT",
                    side=OrderSide.buy,
                    qty=10.0,
                    target_side=PositionSide.long,
                    is_risk_reducing=False,
                    reference_price=100.0,
                    stop_loss_price=95.0,  # t1 gaps to 90 <= 95 -> MUST abort!
                    reason="abort_msft_gap",
                ),
                ProposedOrder(
                    symbol="NVDA",
                    side=OrderSide.buy,
                    qty=45.0,
                    target_side=PositionSide.long,
                    is_risk_reducing=False,
                    reference_price=100.0,
                    stop_loss_price=10.0,  # wide stop so NVDA stays open across gap
                    reason="open_nvda",
                ),
            ]
        if as_of == t1:
            return [
                ProposedOrder(
                    symbol="AAPL",
                    side=OrderSide.sell,
                    qty=5.0,
                    is_risk_reducing=True,
                    reference_price=100.0,
                    reason="carried_exit_aapl",
                ),
                ProposedOrder(
                    symbol="ORCL",
                    side=OrderSide.buy,
                    qty=55.0,  # $5,500 requested, will exceed cash / gross cap after NVDA gaps down
                    target_side=PositionSide.long,
                    is_risk_reducing=False,
                    reference_price=100.0,
                    stop_loss_price=80.0,
                    reason="big_orcl_entry",
                ),
            ]
        return []

    def fake_eval(self, orders, state, dials, universe_meta, signals_by_symbol=None, earnings_by_symbol=None):
        return [
            RiskVerdict(
                approved=True,
                order=o,
                adjusted_qty=o.qty,
                stop_loss_price=o.stop_loss_price,
            )
            for o in orders
        ]

    monkeypatch.setattr("agy_trader.backtest.engine.PortfolioConstructor.build_proposed_orders", fake_propose)
    monkeypatch.setattr("agy_trader.backtest.engine.HardRiskEngine.evaluate_batch", fake_eval)

    res = engine.run(bars, use_news=False)

    # (a) MSFT gap-down entry was aborted at t1 open
    assert res.metrics.risk_rejections >= 1
    assert not any(t.symbol == "MSFT" for t in res.trades)

    # (b) AAPL Pass 1 exit missed t2 bar and carried forward cleanly to t3 open (105.0)
    aapl_trades = [t for t in res.trades if t.symbol == "AAPL"]
    assert len(aapl_trades) == 1
    assert aapl_trades[0].exit_time == t3
    assert aapl_trades[0].exit_reason == "carried_exit_aapl"
    assert aapl_trades[0].exit_price == pytest.approx(105.0 * (1.0 - 0.001), rel=1e-6)

    # (c) Cash and gross exposure conserved across adverse gap
    for pt in res.equity_curve:
        assert pt.cash >= -1e-2
        assert pt.gross_exposure <= cfg.risk.max_gross_exposure + 1e-3


def test_stale_halted_symbol_excluded_and_live_benchmark_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 75 business days total (65 warmup + 10 eval days)
    dates = [ts.date() for ts in pd.bdate_range(start=date(2025, 1, 2), periods=75)]
    n = len(dates)
    aapl_df = _build_ohlc_df(dates, [100.0] * n, [101.0] * n, [99.0] * n, [100.0] * n)

    # HALT stops after index 67 (last 7 dates missing -> must NOT appear in pit_bars on days 68..74!)
    halt_dates = dates[:68]
    nh = len(halt_dates)
    halt_df = _build_ohlc_df(halt_dates, [100.0] * nh, [101.0] * nh, [99.0] * nh, [100.0] * nh)

    # SPY misses index 68 and 70, and closes monotonically at distinct values on live days:
    # We will verify benchmark_return_pct and benchmark_sharpe match live SPY closes ONLY
    # (never injecting duplicated 0%-return closes for missing days).
    spy_indices = [i for i in range(n) if i not in (68, 70)]
    spy_dates = [dates[i] for i in spy_indices]
    # Give SPY varying daily returns on live dates so duplicating a 0% return would change Sharpe!
    spy_closes = [100.0 + (idx * 1.5 if idx % 2 == 0 else idx * 0.5) for idx in range(len(spy_dates))]
    spy_df = _build_ohlc_df(spy_dates, spy_closes, [c + 1.0 for c in spy_closes], [c - 1.0 for c in spy_closes], spy_closes)

    bars = {"AAPL": aapl_df, "HALT": halt_df, "SPY": spy_df}
    universe = [
        UniverseSymbol(symbol="SPY", asset_type=AssetType.etf, sector="Broad Market"),
        UniverseSymbol(symbol="AAPL", asset_type=AssetType.equity, sector="Technology"),
        UniverseSymbol(symbol="HALT", asset_type=AssetType.equity, sector="Energy"),
    ]
    engine = BacktestEngine(AppConfig(), universe)

    seen_pit_symbols: dict[date, set[str]] = {}

    def spy_build_signals(self, as_of, bars_by_symbol, **kwargs):
        seen_pit_symbols[as_of.date()] = set(bars_by_symbol.keys())
        return []

    monkeypatch.setattr("agy_trader.backtest.engine.FeatureEngine.build_signals", spy_build_signals)

    res = engine.run(bars, use_news=False)
    # On dates[:68], HALT is fresh; on dates[68..73], HALT has >1440 min bar age > max_data_staleness_minutes=15 -> excluded!
    assert "HALT" in seen_pit_symbols[dates[67]]
    for stale_dt in dates[68:-1]:
        assert "HALT" not in seen_pit_symbols[stale_dt]

    # Compare benchmark_sharpe against true live SPY closes (indices 65..74 excluding 68 and 70)
    from agy_trader.backtest.metrics import _compute_return_stats

    live_eval_spy = [
        spy_closes[spy_dates.index(d)] for d in dates[65:] if d in spy_dates
    ]
    _, _, exp_bm_sharpe, _, exp_bm_dd = _compute_return_stats(live_eval_spy)
    assert res.metrics.benchmark_sharpe == round(exp_bm_sharpe, 3)
    assert res.metrics.benchmark_max_drawdown_pct == round(exp_bm_dd * 100.0, 2)


def test_etf_vs_equity_slippage_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    dates, opens, highs, lows, closes = _craft_constant_warmup_bars()
    opens[66], closes[66] = 100.0, 100.0
    opens[67], closes[67] = 110.0, 110.0

    bars = {
        "SPY": _build_ohlc_df(dates, opens, highs, lows, closes),
        "AAPL": _build_ohlc_df(dates, opens, highs, lows, closes),
    }
    universe = [
        UniverseSymbol(symbol="SPY", asset_type=AssetType.etf, sector="Broad Market"),
        UniverseSymbol(symbol="AAPL", asset_type=AssetType.equity, sector="Technology"),
    ]
    cfg = AppConfig(
        risk=RiskConfig(max_long_position_pct=0.25, max_net_exposure=0.80),
        backtest=BacktestConfig(
            initial_capital=50_000.0,
            etf_slippage_bps=5.0,
            equity_slippage_bps=10.0,
            short_borrow_apr=0.0,
            benchmark_symbol="SPY",
        ),
    )
    engine = BacktestEngine(cfg, universe)

    t0_ts = datetime(dates[65].year, dates[65].month, dates[65].day, 16, 0, tzinfo=NY_TZ).astimezone(UTC)
    t1_ts = datetime(dates[66].year, dates[66].month, dates[66].day, 16, 0, tzinfo=NY_TZ).astimezone(UTC)

    monkeypatch.setattr(
        "agy_trader.backtest.engine.FeatureEngine.build_signals",
        lambda self, as_of, **kwargs: [_dummy_signal("SPY", as_of), _dummy_signal("AAPL", as_of)],
    )

    def fake_propose(self, signals, current_positions, prices, equity, dials, as_of, universe_meta):
        if as_of == t0_ts:
            return [
                ProposedOrder(
                    symbol="SPY",
                    side=OrderSide.buy,
                    qty=10.0,
                    target_side=PositionSide.long,
                    is_risk_reducing=False,
                    reference_price=100.0,
                    stop_loss_price=80.0,
                ),
                ProposedOrder(
                    symbol="AAPL",
                    side=OrderSide.buy,
                    qty=10.0,
                    target_side=PositionSide.long,
                    is_risk_reducing=False,
                    reference_price=100.0,
                    stop_loss_price=80.0,
                ),
            ]
        if as_of == t1_ts:
            return [
                ProposedOrder(
                    symbol="SPY", side=OrderSide.sell, qty=10.0, is_risk_reducing=True, reference_price=100.0
                ),
                ProposedOrder(
                    symbol="AAPL", side=OrderSide.sell, qty=10.0, is_risk_reducing=True, reference_price=100.0
                ),
            ]
        return []

    def fake_eval(self, orders, state, dials, universe_meta, signals_by_symbol=None, earnings_by_symbol=None):
        return [
            RiskVerdict(approved=True, order=o, adjusted_qty=o.qty, stop_loss_price=o.stop_loss_price)
            for o in orders
        ]

    monkeypatch.setattr("agy_trader.backtest.engine.PortfolioConstructor.build_proposed_orders", fake_propose)
    monkeypatch.setattr("agy_trader.backtest.engine.HardRiskEngine.evaluate_batch", fake_eval)

    res = engine.run(bars, use_news=False)
    by_sym = {t.symbol: t for t in res.trades}
    assert by_sym["SPY"].entry_price == pytest.approx(100.0 * 1.0005, rel=1e-6)
    assert by_sym["SPY"].exit_price == pytest.approx(110.0 * 0.9995, rel=1e-6)
    assert by_sym["AAPL"].entry_price == pytest.approx(100.0 * 1.0010, rel=1e-6)
    assert by_sym["AAPL"].exit_price == pytest.approx(110.0 * 0.9990, rel=1e-6)


def test_ema50_window_parity_with_full_history_feature_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agy_trader.strategy.features import FeatureEngine

    rng = np.random.default_rng(77)
    n_bars = 220
    dates = [ts.date() for ts in pd.bdate_range(start=date(2024, 6, 3), periods=n_bars)]
    bars: dict[str, pd.DataFrame] = {}
    for idx, sym in enumerate(("SPY", "AAPL", "MSFT")):
        rets = rng.normal(0.0004 * (idx + 1), 0.014, size=n_bars)
        closes = 100.0 * np.exp(np.cumsum(rets))
        opens = closes * 0.998
        highs = np.maximum(opens, closes) * 1.01
        lows = np.minimum(opens, closes) * 0.99
        bars[sym] = _build_ohlc_df(dates, opens.tolist(), highs.tolist(), lows.tolist(), closes.tolist())

    universe = [
        UniverseSymbol(symbol="SPY", asset_type=AssetType.etf, sector="Broad Market"),
        UniverseSymbol(symbol="AAPL", asset_type=AssetType.equity, sector="Technology"),
        UniverseSymbol(symbol="MSFT", asset_type=AssetType.equity, sector="Technology"),
    ]
    cfg = AppConfig()
    engine = BacktestEngine(cfg, universe)
    full_fe = FeatureEngine(cfg.strategy)

    captured_signals: dict[datetime, list[SignalRow]] = {}
    orig_build = FeatureEngine.build_signals

    def capture_build(self, as_of, bars_by_symbol, **kwargs):
        sigs = orig_build(self, as_of, bars_by_symbol, **kwargs)
        captured_signals[as_of] = sigs
        return sigs

    monkeypatch.setattr("agy_trader.backtest.engine.FeatureEngine.build_signals", capture_build)

    # Run backtest only over final 5 days with matching StrategyConfig (use_news=True, no news)
    engine.run(bars, start_date=dates[-5], end_date=dates[-1], use_news=True)
    assert len(captured_signals) == 4  # signals emitted for dates[-5..-2] (last day skips queueing)

    for as_of, engine_sigs in captured_signals.items():
        full_sigs = full_fe.build_signals(as_of=as_of, bars_by_symbol=bars, benchmark_symbol="SPY")
        assert engine_sigs == full_sigs

