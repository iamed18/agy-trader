from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from agy_trader.backtest.metrics import BacktestMetrics
from agy_trader.backtest.sp500_cohorts import validate_research_dir
from agy_trader.backtest.v3_benchmark import (
    TrackComparisonRow,
    TrackSpec,
    WindowMetrics,
    _calc_excess_sharpe,
    build_default_track_specs,
    render_v3_benchmark_markdown,
    run_v3_benchmark,
    select_is_winning_track,
)
from agy_trader.backtest.v3_universe import (
    V3_CORE_ETFS,
    V3EquityCandidate,
)
from agy_trader.cli.main import app
from agy_trader.core.config import AppConfig

NY_TZ = ZoneInfo("America/New_York")


def _trading_dates(n_days: int, start: date = date(2023, 1, 3)) -> list[date]:
    """Generate n_days of weekday trading dates."""
    out: list[date] = []
    cur = start
    while len(out) < n_days:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def _make_bars_df(
    dates: list[date],
    closes: np.ndarray | list[float],
    *,
    volume: float = 2_000_000.0,
    daily_spread: float = 0.01,
) -> pd.DataFrame:
    """Create canonical daily OHLCV DataFrame for testing."""
    c = np.asarray(closes, dtype=float)
    o = c * (1.0 - 0.002)
    h = np.maximum(o, c) * (1.0 + daily_spread)
    lows = np.minimum(o, c) * (1.0 - daily_spread)
    timestamps = [
        datetime(d.year, d.month, d.day, 16, 0, tzinfo=NY_TZ).astimezone(UTC) for d in dates
    ]
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(timestamps, tz="UTC"),
            "open": o,
            "high": h,
            "low": lows,
            "close": c,
            "volume": np.full(len(dates), volume, dtype=float),
            "vwap": c,
        }
    )


def _dummy_metrics(
    label: str,
    *,
    cagr_pct: float = 15.0,
    excess_sharpe: float = 0.5,
    sharpe: float = 1.2,
    trading_days: int = 100,
) -> BacktestMetrics:
    return BacktestMetrics(
        label=label,
        start_date="2023-01-03",
        end_date="2023-12-29",
        trading_days=trading_days,
        initial_capital=100000.0,
        final_equity=115000.0,
        total_return_pct=15.0,
        cagr_pct=cagr_pct,
        ann_volatility_pct=12.0,
        sharpe_ratio=sharpe,
        excess_sharpe_ratio=excess_sharpe,
        sortino_ratio=1.5,
        calmar_ratio=1.1,
        max_drawdown_pct=-10.0,
        win_rate_pct=55.0,
        profit_factor=1.4,
        avg_holding_days=5.0,
        annual_turnover=4.0,
        total_orders_filled=20,
        closed_trades=15,
        stop_loss_exits=1,
        risk_rejections=0,
        total_slippage_usd=50.0,
        total_borrow_cost_usd=10.0,
        total_sec_fees_usd=5.0,
        benchmark_symbol="SPY",
        benchmark_return_pct=12.0,
        benchmark_sharpe=0.9,
        benchmark_max_drawdown_pct=-12.0,
    )


# ---------------------------------------------------------------------------
# 1. Ladder Specs & Model Verification
# ---------------------------------------------------------------------------


def test_build_default_track_specs() -> None:
    specs = build_default_track_specs()
    assert len(specs) == 6
    track_ids = [s.track_id for s in specs]
    assert track_ids == ["T0", "T1", "T2", "T3", "T4", "T5"]

    # T0: v0.2 Prod Baseline
    t0 = specs[0]
    assert t0.track_id == "T0"
    assert t0.use_sector_residual is False
    assert t0.strategy_overrides == {}
    assert t0.risk_overrides == {}

    # T1: Broad Sizing & Risk
    t1 = specs[1]
    assert t1.strategy_overrides["top_k_long"] == 12
    assert t1.strategy_overrides["vol_sizing_mode"] == "capped_inv_vol"
    assert t1.strategy_overrides["vol_sizing_floor"] == 0.18
    assert t1.strategy_overrides["vol_sizing_cap"] == 0.45
    assert t1.strategy_overrides["portfolio_vol_target_ann"] == 0.14
    assert t1.risk_overrides["max_long_position_pct"] == 0.12
    assert t1.risk_overrides["stop_loss_atr_mult_long"] == 6.0
    assert t1.risk_overrides["max_drawdown_cooldown_days"] == 60

    # T2: 4-Tranche Weekly
    t2 = specs[2]
    assert t2.strategy_overrides["rebalance_every_n_days"] == 5
    assert t2.strategy_overrides["rebalance_tranches"] == 4

    # T3: Sector Residual & FIP
    t3 = specs[3]
    assert t3.use_sector_residual is True
    assert t3.strategy_overrides["use_sector_residual"] is True
    assert t3.strategy_overrides["momentum_mode"] == "m2_residual"
    assert t3.strategy_overrides["w_fip"] == 0.20
    assert t3.strategy_overrides["w_jump_penalty"] == 0.10

    # T4: Daniel-Moskowitz Smooth Regime
    t4 = specs[4]
    assert t4.strategy_overrides["regime_mode"] == "daniel_moskowitz"

    # T5: Finalist & PEAD
    t5 = specs[5]
    assert t5.strategy_overrides["finalist_pool_size"] == 40
    assert t5.strategy_overrides["w_pead"] == 0.15


def test_calc_excess_sharpe_over_tbill() -> None:
    eq = [100.0, 101.0, 102.5, 103.8, 105.2]
    low_rf = [0.0001] * len(eq)
    high_rf = [0.03] * len(eq)
    assert _calc_excess_sharpe(eq, low_rf) > 0.0
    assert _calc_excess_sharpe(eq, high_rf) < 0.0


def test_window_metrics_model() -> None:
    m = _dummy_metrics("Test")
    wm = WindowMetrics.from_metrics(m)
    assert wm.cagr_pct == 15.0
    assert wm.sharpe_ratio == 1.2
    assert wm.excess_sharpe_ratio == 0.5
    assert wm.max_drawdown_pct == -10.0


# ---------------------------------------------------------------------------
# 2. Strict Anti-Peeking Winner Selection
# ---------------------------------------------------------------------------


def test_strict_is_only_winner_selection() -> None:
    """Verify that selection is strictly driven by IS excess Sharpe and ignores OOS completely."""
    # Track A has mediocre IS excess Sharpe (0.2) but stellar OOS (1.8)
    row_a = TrackComparisonRow(
        track_id="T1",
        name="Track A",
        description="Low IS, Great OOS",
        is_metrics=_dummy_metrics("T1_IS", excess_sharpe=0.20, sharpe=0.8),
        oos_metrics=_dummy_metrics("T1_OOS", excess_sharpe=1.80, sharpe=2.5),
        full_metrics=_dummy_metrics("T1_Full"),
        ew_is_cagr_pct=10.0,
        ew_oos_cagr_pct=12.0,
        ew_full_cagr_pct=11.0,
        spy_is_cagr_pct=8.0,
        spy_oos_cagr_pct=9.0,
        spy_full_cagr_pct=8.5,
    )
    # Track B has highest IS excess Sharpe (0.85) but mediocre OOS (0.1)
    row_b = TrackComparisonRow(
        track_id="T2",
        name="Track B",
        description="High IS, Low OOS",
        is_metrics=_dummy_metrics("T2_IS", excess_sharpe=0.85, sharpe=1.6),
        oos_metrics=_dummy_metrics("T2_OOS", excess_sharpe=0.10, sharpe=0.6),
        full_metrics=_dummy_metrics("T2_Full"),
        ew_is_cagr_pct=10.0,
        ew_oos_cagr_pct=12.0,
        ew_full_cagr_pct=11.0,
        spy_is_cagr_pct=8.0,
        spy_oos_cagr_pct=9.0,
        spy_full_cagr_pct=8.5,
    )

    winner_id, reason = select_is_winning_track([row_a, row_b])
    # T2 must win strictly based on IS performance!
    assert winner_id == "T2"
    assert "0.850" in reason
    assert "without inspecting OOS" in reason


# ---------------------------------------------------------------------------
# 3. End-to-End Synthetic Benchmark & Plot Leaks Check
# ---------------------------------------------------------------------------


def test_end_to_end_v3_benchmark_synthetic(tmp_path: Path) -> None:
    """Run end-to-end benchmark on 360-day synthetic dataset, verify plots and zero leaked figures."""
    n_days = 360
    dates = _trading_dates(n_days, start=date(2023, 1, 3))
    rng = np.random.default_rng(42)

    bars_by_symbol: dict[str, pd.DataFrame] = {}

    # Benchmark & Core ETFs
    spy_closes = 400.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.008, n_days))
    bars_by_symbol["SPY"] = _make_bars_df(dates, spy_closes)

    for etf in V3_CORE_ETFS:
        if etf == "SPY":
            continue
        etf_closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n_days))
        bars_by_symbol[etf] = _make_bars_df(dates, etf_closes)

    # 4 Equities: 2 Tech, 2 Energy
    symbols = ["AAPL", "MSFT", "XOM", "CVX"]
    sectors = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "XOM": "Energy",
        "CVX": "Energy",
    }
    pool: list[V3EquityCandidate] = []
    for sym in symbols:
        px = 120.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, n_days))
        bars_by_symbol[sym] = _make_bars_df(dates, px, volume=5_000_000.0)
        pool.append(
            V3EquityCandidate(
                symbol=sym,
                security=sym,
                sector=sectors[sym],
                sources=("sp500_hist", "sp500_curr"),
                sp500_intervals=(("2022-01-01", None),),
            )
        )

    # Add a delisted candidate with historical interval to test delisted count
    pool.append(
        V3EquityCandidate(
            symbol="SIVB",
            security="Silicon Valley Bank",
            sector="Financials",
            sources=("sp500_hist",),
            sp500_intervals=(("2016-01-01", "2023-03-10"),),
        )
    )

    is_start = dates[260]
    is_end = dates[305]
    oos_start = dates[306]
    oos_end = dates[-1]

    # Test with T0 and T1 for rapid test iteration
    custom_specs = [
        TrackSpec(
            track_id="T0",
            name="Baseline",
            description="T0 Baseline",
            strategy_overrides={},
            risk_overrides={},
        ),
        TrackSpec(
            track_id="T1",
            name="Sizing",
            description="T1 Sizing",
            strategy_overrides={"top_k_long": 2, "vol_sizing_mode": "capped_inv_vol"},
            risk_overrides={"max_long_position_pct": 0.50},
        ),
    ]

    out_dir = tmp_path / "v3_bench_reports"
    report, _series_bundle = run_v3_benchmark(
        bars_by_symbol,
        pool,
        app_config=AppConfig(),
        is_start=is_start,
        is_end=is_end,
        oos_start=oos_start,
        oos_end=oos_end,
        top_n=2,
        hysteresis=1,
        n_trials=2,
        run_cpcv=True,
        run_stress=True,
        track_specs=custom_specs,
        output_dir=out_dir,
    )

    assert report.pool_total == 5
    assert report.delisted_count >= 1
    assert len(report.tracks) == 2
    assert report.is_selected_track_id in ("T0", "T1")

    # Verify generated plots
    expected_plots = [
        out_dir / "v3_01_10yr_equity_and_drawdown.png",
        out_dir / "v3_02_track_ablation_is_vs_oos.png",
        out_dir / "v3_03_calendar_year_10yr.png",
        out_dir / "v3_04_cpcv_and_delisting_stress.png",
    ]
    for p in expected_plots:
        assert p.exists(), f"Plot {p.name} does not exist"
        assert p.stat().st_size > 0, f"Plot {p.name} is empty"

    # Verify ZERO open matplotlib figures leaked!
    assert len(plt.get_fignums()) == 0

    # Verify generated markdown
    md = render_v3_benchmark_markdown(report)
    assert "# V3 Research" in md
    assert "Strict Anti-Peeking In-Sample Winner" in md
    assert "Progressive Ladder Performance" in md
    assert "Stress Matrix" in md

    # Verify summary JSON and MD files written
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "summary.md").exists()


# ---------------------------------------------------------------------------
# 4. CLI Safety & Rejection of Protected Directories
# ---------------------------------------------------------------------------


def test_cli_rejects_live_parquet_dir(tmp_path: Path) -> None:
    """Verify that CLI v3-research rejects --research-dir targeting data/parquet."""
    runner = CliRunner()

    with pytest.raises(ValueError, match="must not"):
        validate_research_dir(Path("data/parquet"))

    res = runner.invoke(
        app,
        [
            "v3-research",
            "--research-dir",
            "data/parquet",
            "--output-dir",
            str(tmp_path / "out"),
            "--json",
        ],
    )
    assert res.exit_code != 0
    assert "must not" in res.output or "error" in res.output.lower()
