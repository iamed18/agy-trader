from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from agy_trader.backtest.engine import BacktestEngine, BacktestRunSeries
from agy_trader.backtest.sp500_cohorts import (
    Sp500Constituent,
    _aligned_close_series,
    fetch_and_cache_sp500_bars,
    filter_active_for_duration,
    generate_sp500_cohort_plots,
    load_or_fetch_sp500_constituents,
    parse_wikipedia_sp500_html,
    render_sp500_cohort_markdown,
    run_sp500_cohort_study,
    sample_sp500_cohorts,
    select_point_in_time_candidates,
    validate_research_dir,
)
from agy_trader.cli.main import app
from agy_trader.core.config import AppConfig, BacktestConfig, RiskConfig, StrategyConfig
from agy_trader.core.models import AssetType, UniverseSymbol, ensure_utc
from agy_trader.storage.db import StateDB
from agy_trader.storage.parquet_store import ParquetBarStore
from agy_trader.strategy.features import FeatureEngine

NY_TZ = ZoneInfo("America/New_York")


def _make_bars_df(
    dates: list[date],
    closes: np.ndarray | list[float],
    *,
    daily_spread: float = 0.01,
) -> pd.DataFrame:
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
            "volume": np.full(len(dates), 2_000_000.0),
            "vwap": c,
        }
    )


def _trading_dates(n_days: int, start: date = date(2023, 1, 3)) -> list[date]:
    out: list[date] = []
    cur = start
    while len(out) < n_days:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# 1. Wikipedia HTML Parsing, Anti-Lookahead Cutoff & Issuer Deduplication
# ---------------------------------------------------------------------------


def test_wikipedia_parsing_cutoff_and_dedup(tmp_path: Path) -> None:
    html = """
    <html><body>
      <table id="constituents">
        <tr>
          <th>Symbol</th><th>Security</th><th>GICS Sector</th>
          <th>GICS Sub-Industry</th><th>HQ</th><th>Date added</th><th>CIK</th>
        </tr>
        <tr>
          <td>AAPL[1]</td><td>Apple Inc.[2]</td><td>Information Technology</td>
          <td>Tech Hardware</td><td>Cupertino, CA</td><td>1982-11-30[a]</td><td>0000320193</td>
        </tr>
        <tr>
          <td>UNH</td><td>UnitedHealth Group</td><td>Health Care</td>
          <td>Managed Health Care</td><td>Minnetonka, MN</td><td>1994-07[1]</td><td>0000731766</td>
        </tr>
        <tr>
          <td>JPM</td><td>JPMorgan Chase</td><td>Financials</td>
          <td>Diversified Banks</td><td>New York, NY</td><td>1975 (est.)</td><td>0000019617</td>
        </tr>
        <tr>
          <td>GOOG</td><td>Alphabet Inc. (Class C)</td><td>Communication Services</td>
          <td>Interactive Media</td><td>Mountain View, CA</td><td>2006-04-03</td><td>0001652044</td>
        </tr>
        <tr>
          <td>GOOGL</td><td>Alphabet Inc. (Class A)</td><td>Communication Services</td>
          <td>Interactive Media</td><td>Mountain View, CA</td><td>2014-04-03</td><td>0001652044</td>
        </tr>
        <tr>
          <td>FOX</td><td>Fox Corp (Class B)</td><td>Communication Services</td>
          <td>Broadcasting</td><td>New York, NY</td><td>2019-03-19</td><td></td>
        </tr>
        <tr>
          <td>FOXA</td><td>Fox Corp (Class A)</td><td>Communication Services</td>
          <td>Broadcasting</td><td>New York, NY</td><td>2019-03-19</td><td></td>
        </tr>
        <tr>
          <td>NWS</td><td>News Corp (Class B)</td><td>Communication Services</td>
          <td>Publishing</td><td>New York, NY</td><td>2015-09-18</td><td></td>
        </tr>
        <tr>
          <td>NWSA</td><td>News Corp (Class A)</td><td>Communication Services</td>
          <td>Publishing</td><td>New York, NY</td><td>2015-09-18</td><td></td>
        </tr>
        <tr>
          <td>LATE</td><td>Late Addition Corp</td><td>Industrials</td>
          <td>Machinery</td><td>Chicago, IL</td><td>2024-06-24</td><td>0009999999</td>
        </tr>
        <tr>
          <td>NODATE</td><td>Missing Date Corp</td><td>Energy</td>
          <td>Oil</td><td>Houston, TX</td><td>unknown</td><td>0008888888</td>
        </tr>
      </table>
    </body></html>
    """
    parsed = parse_wikipedia_sp500_html(html)
    by_sym = {c.symbol: c for c in parsed}
    assert by_sym["AAPL"].sector == "Technology"
    assert by_sym["AAPL"].date_added == date(1982, 11, 30)
    assert by_sym["UNH"].sector == "Healthcare"
    assert by_sym["UNH"].date_added == date(1994, 7, 1)
    assert by_sym["JPM"].sector == "Financials"
    assert by_sym["JPM"].date_added == date(1975, 1, 1)
    assert by_sym["NODATE"].date_added is None

    deduped, pre_cutoff_count = select_point_in_time_candidates(
        parsed, cutoff_date=date(2021, 11, 23)
    )
    assert pre_cutoff_count == 9  # Excludes LATE (2024) and NODATE (None)
    deduped_syms = {c.symbol for c in deduped}
    assert deduped_syms == {"AAPL", "UNH", "JPM", "GOOGL", "FOXA", "NWSA"}

    # Verify JSON cache round-trip without network
    cache_path = tmp_path / "sp500_constituents.json"
    cache_path.write_text(
        "[" + ",".join(c.model_dump_json() for c in parsed) + "]",
        encoding="utf-8",
    )
    reloaded = load_or_fetch_sp500_constituents(cache_path, refresh=False)
    assert reloaded == parsed


# ---------------------------------------------------------------------------
# 2. Active-for-Duration Filtering & Deterministic Sampling
# ---------------------------------------------------------------------------


def test_filter_active_for_duration_and_sampling() -> None:
    dates = _trading_dates(120)
    spy_df = _make_bars_df(dates, np.linspace(400.0, 440.0, len(dates)))

    bars: dict[str, pd.DataFrame] = {
        "SPY": spy_df,
        "GOOD1": _make_bars_df(dates, np.linspace(100.0, 130.0, len(dates))),
        "GOOD2": _make_bars_df(dates, np.linspace(50.0, 65.0, len(dates))),
        "LATEIPO": _make_bars_df(dates[10:], np.linspace(20.0, 25.0, len(dates) - 10)),
        "DELISTED": _make_bars_df(dates[:-10], np.linspace(30.0, 28.0, len(dates) - 10)),
        "GAPPED": _make_bars_df(
            dates[:40] + dates[55:], np.linspace(80.0, 90.0, len(dates) - 15)
        ),
    }

    candidates = [
        Sp500Constituent(
            symbol=sym,
            security=sym,
            gics_sector="Industrials",
            sector="Industrials",
            date_added=date(2015, 1, 1),
            cik=str(idx),
        )
        for idx, sym in enumerate(["GOOD1", "GOOD2", "LATEIPO", "DELISTED", "GAPPED"], start=1)
    ]

    eligible = filter_active_for_duration(bars, candidates, benchmark_symbol="SPY")
    assert [c.symbol for c in eligible] == ["GOOD1", "GOOD2"]

    # Deterministic 50-symbol cohort sampling from 100 synthetic eligible constituents
    pool = [
        Sp500Constituent(
            symbol=f"SYM{i:03d}",
            security=f"Company {i}",
            gics_sector="Financials",
            sector="Financials",
            date_added=date(2015, 1, 1),
            cik=str(i),
        )
        for i in range(100)
    ]
    c1 = sample_sp500_cohorts(pool, n_cohorts=5, cohort_size=50, seed=20261122)
    c2 = sample_sp500_cohorts(pool, n_cohorts=5, cohort_size=50, seed=20261122)
    assert [[x.symbol for x in grp] for grp in c1] == [[x.symbol for x in grp] for grp in c2]
    assert all(len(grp) == 50 and len({x.symbol for x in grp}) == 50 for grp in c1)
    assert len({frozenset(x.symbol for x in grp) for grp in c1}) == 5

    with pytest.raises(ValueError):
        sample_sp500_cohorts(eligible, n_cohorts=2, cohort_size=50, seed=1)


# ---------------------------------------------------------------------------
# 3. Benchmark Isolation in FeatureEngine & BacktestEngine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("momentum_mode", ["legacy", "m2_residual"])
def test_feature_engine_benchmark_isolation(
    momentum_mode: Literal["legacy", "m2_residual"],
) -> None:
    dates = _trading_dates(160)
    # Give SPY a crash at the end so macro_regime_scale < 1.0, plus extreme returns
    spy_closes = np.linspace(400.0, 480.0, len(dates))
    spy_closes[-10:] = np.linspace(480.0, 390.0, 10)

    bars = {
        "SPY": _make_bars_df(dates, spy_closes),
        "AAA": _make_bars_df(dates, np.linspace(100.0, 140.0, len(dates))),
        "BBB": _make_bars_df(dates, np.linspace(100.0, 95.0, len(dates))),
        "CCC": _make_bars_df(dates, np.linspace(100.0, 110.0, len(dates))),
    }
    cfg = StrategyConfig(
        momentum_mode=momentum_mode,
        spy_sma200_defensive_scale=0.60,
        spy_ret10_crash_thresh=-0.055,
        spy_ret10_crash_scale=0.60,
    )
    fe = FeatureEngine(cfg)
    as_of = ensure_utc(datetime(dates[-1].year, dates[-1].month, dates[-1].day, 16, 0, tzinfo=NY_TZ))

    sigs_isolated = fe.build_signals(
        as_of=as_of,
        bars_by_symbol=bars,
        benchmark_symbol="SPY",
        universe_symbols=frozenset({"AAA", "BBB", "CCC"}),
    )
    returned_syms = {s.symbol for s in sigs_isolated}
    assert returned_syms == {"AAA", "BBB", "CCC"}
    assert "SPY" not in returned_syms
    assert all("SPY" not in s.peer_corr_60d for s in sigs_isolated)
    # SPY was still used for macro_regime_scale (crash scale triggered < 1.0)
    assert all(s.macro_regime_scale < 1.0 for s in sigs_isolated)

    # Without universe_symbols, SPY is included as before
    sigs_all = fe.build_signals(as_of=as_of, bars_by_symbol=bars, benchmark_symbol="SPY")
    assert {s.symbol for s in sigs_all} == {"AAA", "BBB", "CCC", "SPY"}


def test_backtest_engine_ew_universe_excludes_spy_benchmark() -> None:
    dates = _trading_dates(120)
    # SPY surges wildly (+300%) while cohort stocks AAA/BBB double cleanly (+100%)
    spy_closes = np.geomspace(100.0, 400.0, len(dates))
    aaa_closes = np.geomspace(100.0, 200.0, len(dates))
    bbb_closes = np.geomspace(100.0, 200.0, len(dates))

    bars = {
        "SPY": _make_bars_df(dates, spy_closes, daily_spread=0.002),
        "AAA": _make_bars_df(dates, aaa_closes, daily_spread=0.002),
        "BBB": _make_bars_df(dates, bbb_closes, daily_spread=0.002),
    }
    universe = [
        UniverseSymbol(symbol="AAA", asset_type=AssetType.equity, sector="Technology"),
        UniverseSymbol(symbol="BBB", asset_type=AssetType.equity, sector="Financials"),
    ]
    app_cfg = AppConfig(
        strategy=StrategyConfig(
            momentum_mode="m2_residual",
            rebalance_checkpoints_et=["15:40"],
            top_k_long=2,
            top_k_short=0,
            min_signal_threshold=0.0,
        ),
        risk=RiskConfig(),
        backtest=BacktestConfig(initial_capital=100_000.0, benchmark_symbol="SPY"),
    )
    engine = BacktestEngine(app_cfg, universe)
    run_series = engine.run_with_series(bars, allow_shorting=True, use_news=False)
    assert isinstance(run_series, BacktestRunSeries)

    # SPY must never be traded since it is not in universe
    assert all(t.symbol != "SPY" for t in run_series.result.trades)

    # Verify ew_universe_equities matches exact mean daily return of {AAA, BBB} (excluding SPY)
    eval_start_idx = len(dates) - len(run_series.result.equity_curve)
    expected_ew = 100_000.0
    for k in range(1, len(run_series.ew_universe_equities)):
        idx = eval_start_idx + k
        r_a = aaa_closes[idx] / aaa_closes[idx - 1] - 1.0
        r_b = bbb_closes[idx] / bbb_closes[idx - 1] - 1.0
        expected_ew *= 1.0 + 0.5 * (r_a + r_b)
    assert run_series.ew_universe_equities[-1] == pytest.approx(expected_ew, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. End-to-End Study + Plots + Markdown on Synthetic Bars
# ---------------------------------------------------------------------------


def test_end_to_end_cohort_study_plots_and_markdown(tmp_path: Path) -> None:
    dates = _trading_dates(260, start=date(2022, 1, 3))
    rng = np.random.default_rng(123)

    spy_rets = rng.normal(0.0004, 0.01, size=len(dates))
    spy_closes = 400.0 * np.cumprod(1.0 + spy_rets)
    bars: dict[str, pd.DataFrame] = {"SPY": _make_bars_df(dates, spy_closes)}

    sectors = ["Technology", "Healthcare", "Financials", "Industrials"]
    eligible: list[Sp500Constituent] = []
    for i in range(12):
        sym = f"EQ{i:02d}"
        sec = sectors[i % len(sectors)]
        rets = rng.normal(0.0005 + 0.00005 * (i % 3), 0.015, size=len(dates))
        closes = 100.0 * np.cumprod(1.0 + rets)
        bars[sym] = _make_bars_df(dates, closes)
        eligible.append(
            Sp500Constituent(
                symbol=sym,
                security=f"Equity {i}",
                gics_sector=sec,
                sector=sec,
                date_added=date(2018, 1, 1),
                cik=str(1000 + i),
            )
        )

    app_cfg = AppConfig(
        strategy=StrategyConfig(
            momentum_mode="m2_residual",
            rebalance_checkpoints_et=["15:40"],
            top_k_long=4,
            top_k_short=0,
            min_signal_threshold=0.0,
        ),
        risk=RiskConfig(),
        backtest=BacktestConfig(initial_capital=100_000.0, benchmark_symbol="SPY"),
    )

    report, ts_bundle = run_sp500_cohort_study(
        app_config=app_cfg,
        constituents_total_count=503,
        pre_cutoff_count=450,
        deduped_count=445,
        eligible=eligible,
        bars_by_symbol=bars,
        macro_points=[],
        n_cohorts=3,
        cohort_size=8,
        seed=20261122,
        cutoff_date=date(2021, 11, 23),
    )

    assert report.n_cohorts == 3
    assert report.cohort_size == 8
    assert len(report.cohorts) == 3
    assert all(len(c.symbols) == 8 for c in report.cohorts)
    assert all("2022" in c.annual_returns for c in report.cohorts)

    plot_dir = tmp_path / "plots"
    plot_paths = generate_sp500_cohort_plots(report, ts_bundle, plot_dir)
    assert len(plot_paths) == 4
    expected_names = [
        "01_equity_and_drawdown_fan.png",
        "02_per_cohort_paired_metrics.png",
        "03_risk_return_and_alpha_spreads.png",
        "04_calendar_year_regime_comparison.png",
    ]
    assert [p.name for p in plot_paths] == expected_names
    for p in plot_paths:
        assert p.exists()
        data = p.read_bytes()
        assert len(data) > 5_000
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    report_with_plots = report.model_copy(update={"plot_paths": [str(p) for p in plot_paths]})
    md = render_sp500_cohort_markdown(report_with_plots)
    assert "S&P 500 3-Cohort Validation Report" in md
    assert "Cohort 01" in md
    assert "Cohort 03" in md
    assert "01_equity_and_drawdown_fan.png" in md


def test_sp500_cohorts_cli_offline_cached(tmp_path: Path) -> None:
    runner = CliRunner()
    repo_config = Path(__file__).resolve().parents[1] / "config"
    data_dir = tmp_path / "data"
    research_dir = tmp_path / "research"
    output_dir = tmp_path / "reports"

    # Seed read-only StateDB with a macro snapshot
    db = StateDB(data_dir / "state.db")
    db.upsert_macro([])

    start_d = date.today() - timedelta(days=210)
    dates = [d for d in _trading_dates(160, start=start_d) if d <= date.today()]
    rng = np.random.default_rng(99)
    store = ParquetBarStore(research_dir)
    store.save_bars("SPY", _make_bars_df(dates, 400.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, len(dates)))), timeframe="1d")

    constituents: list[dict[str, object]] = []
    for i in range(10):
        sym = f"TST{i:02d}"
        store.save_bars(sym, _make_bars_df(dates, 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.015, len(dates)))), timeframe="1d")
        constituents.append(
            {
                "symbol": sym,
                "security": f"Test {i}",
                "gics_sector": "Technology",
                "sector": "Technology",
                "date_added": "2018-01-01",
                "cik": str(2000 + i),
            }
        )
    research_dir.mkdir(parents=True, exist_ok=True)
    (research_dir / "sp500_constituents.json").write_text(json.dumps(constituents), encoding="utf-8")

    res = runner.invoke(
        app,
        [
            "sp500-cohorts",
            "--n-cohorts",
            "2",
            "--cohort-size",
            "6",
            "--seed",
            "42",
            "--cutoff-date",
            dates[0].isoformat(),
            "--research-dir",
            str(research_dir),
            "--output-dir",
            str(output_dir),
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(repo_config),
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["n_cohorts"] == 2
    assert payload["cohort_size"] == 6
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "cohorts_manifest.json").exists()
    assert len(payload["plot_paths"]) == 4


# ---------------------------------------------------------------------------
# 5. Audit Regression Tests
# ---------------------------------------------------------------------------


def test_backtest_engine_ew_universe_missing_bar0_and_intermediate_gap() -> None:
    """HIGH-1: Missing bar_idx==0 and intermediate bar preserve 1/N weights & multi-day return."""
    n_warmup = 65
    dates = _trading_dates(n_warmup + 4)

    spy_closes = np.full(len(dates), 400.0)
    bbb_closes = np.full(len(dates), 100.0)

    # AAA misses bar_idx==0 (index 65) and bar_idx==2 (index 67), dropping -20% then -50%
    aaa_indices = [i for i in range(len(dates)) if i not in (65, 67)]
    aaa_dates = [dates[i] for i in aaa_indices]
    aaa_closes = []
    for i in aaa_indices:
        if i < 65:
            aaa_closes.append(100.0)
        elif i == 66:
            aaa_closes.append(80.0)  # -20% from pre-bar0 close 100.0
        else:
            aaa_closes.append(40.0)  # -50% across missing bar_idx==2

    bars = {
        "SPY": _make_bars_df(dates, spy_closes, daily_spread=0.001),
        "AAA": _make_bars_df(aaa_dates, aaa_closes, daily_spread=0.001),
        "BBB": _make_bars_df(dates, bbb_closes, daily_spread=0.001),
    }
    universe = [
        UniverseSymbol(symbol="AAA", asset_type=AssetType.equity, sector="Technology"),
        UniverseSymbol(symbol="BBB", asset_type=AssetType.equity, sector="Financials"),
    ]
    app_cfg = AppConfig(
        strategy=StrategyConfig(
            momentum_mode="m2_residual",
            rebalance_checkpoints_et=["15:40"],
            top_k_long=2,
            top_k_short=0,
            min_signal_threshold=0.0,
        ),
        risk=RiskConfig(),
        backtest=BacktestConfig(initial_capital=100_000.0, benchmark_symbol="SPY"),
    )
    run_series = BacktestEngine(app_cfg, universe).run_with_series(
        bars, allow_shorting=True, use_news=False
    )
    ew = run_series.ew_universe_equities
    assert len(ew) == 4
    assert ew[0] == pytest.approx(100_000.0)
    # Day 1: mean(-0.20, 0.0) = -10% -> 90,000
    assert ew[1] == pytest.approx(90_000.0, rel=1e-9)
    # Day 2: AAA missing -> 0% on 1/2 weight -> 90,000 (no 1-day reallocation)
    assert ew[2] == pytest.approx(90_000.0, rel=1e-9)
    # Day 3: AAA realizes exact 2-day -50% gap return -> mean(-0.50, 0.0) = -25% -> 67,500
    assert ew[3] == pytest.approx(67_500.0, rel=1e-9)


def test_benchmark_calendar_clipping_and_aligned_close_guard(tmp_path: Path) -> None:
    """HIGH-2 & LOW-1: Extra non-SPY timestamps are clipped; _aligned_close_series never backfills future."""
    dates = _trading_dates(120, start=date(2022, 3, 1))
    spy_df = _make_bars_df(dates, np.linspace(400.0, 440.0, len(dates)))

    # Constituent has extra timestamps before SPY, after SPY, and an extra Saturday mid-span
    extra_before = date(2022, 2, 15)
    extra_mid = date(2022, 5, 7)  # Saturday
    extra_after = dates[-1] + timedelta(days=3)
    all_extra_dates = [extra_before] + dates[:40] + [extra_mid] + dates[40:] + [extra_after]

    bars: dict[str, pd.DataFrame] = {"SPY": spy_df}
    eligible_cands: list[Sp500Constituent] = []
    for i in range(4):
        sym = f"CLIP{i}"
        d_list = all_extra_dates if i == 0 else dates
        bars[sym] = _make_bars_df(d_list, np.linspace(100.0, 120.0 + i, len(d_list)))
        eligible_cands.append(
            Sp500Constituent(
                symbol=sym,
                security=sym,
                gics_sector="Industrials",
                sector="Industrials",
                date_added=date(2015, 1, 1),
                cik=str(500 + i),
            )
        )

    filtered = filter_active_for_duration(bars, eligible_cands, benchmark_symbol="SPY")
    assert [c.symbol for c in filtered] == ["CLIP0", "CLIP1", "CLIP2", "CLIP3"]

    app_cfg = AppConfig(
        strategy=StrategyConfig(
            momentum_mode="m2_residual",
            rebalance_checkpoints_et=["15:40"],
            top_k_long=2,
            top_k_short=0,
            min_signal_threshold=0.0,
        ),
        risk=RiskConfig(max_drawdown_halt_pct=0.18),
        backtest=BacktestConfig(initial_capital=100_000.0, benchmark_symbol="SPY"),
    )
    report, ts_bundle = run_sp500_cohort_study(
        app_config=app_cfg,
        constituents_total_count=503,
        pre_cutoff_count=450,
        deduped_count=445,
        eligible=filtered,
        bars_by_symbol=bars,
        macro_points=[],
        n_cohorts=1,
        cohort_size=4,
        seed=7,
        cutoff_date=date(2021, 11, 23),
    )
    assert report.trading_days == len(dates) - 65
    assert report.max_drawdown_halt_pct == 18.0
    assert report.top_k_short == 0

    import matplotlib.pyplot as plt

    plot_paths = generate_sp500_cohort_plots(report, ts_bundle, tmp_path / "plots")
    assert len(plot_paths) == 4
    assert len(plt.get_fignums()) == 0

    md = render_sp500_cohort_markdown(report)
    assert "Survivor-Conditioning Disclosure" in md
    assert "max_drawdown_halt_pct=18.0%" in md
    assert "2022*" in md

    # LOW-1: _aligned_close_series raises ValueError instead of backfilling a future price
    early_ts = datetime(2022, 1, 3, 16, 0, tzinfo=NY_TZ).astimezone(UTC)
    with pytest.raises(ValueError, match="No price available on or before"):
        _aligned_close_series(spy_df, [early_ts])


def test_share_class_brk_bf_and_placeholder_ciks() -> None:
    """MEDIUM-1: BRK.B/BF.B preferred over BRK.A/BF.A; zero/N/A CIKs never collide."""
    html = """
    <html><body><table id="constituents">
      <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>Sub</th><th>HQ</th><th>Date added</th><th>CIK</th></tr>
      <tr><td>BRK.A</td><td>Berkshire A</td><td>Financials</td><td>x</td><td>x</td><td>2010-02-16</td><td>0001067983</td></tr>
      <tr><td>BRK.B</td><td>Berkshire B</td><td>Financials</td><td>x</td><td>x</td><td>2010-02-16</td><td>0001067983</td></tr>
      <tr><td>BF.A</td><td>Brown-Forman A</td><td>Consumer Staples</td><td>x</td><td>x</td><td>1982-10-31</td><td>0000014693</td></tr>
      <tr><td>BF.B</td><td>Brown-Forman B</td><td>Consumer Staples</td><td>x</td><td>x</td><td>1982-10-31</td><td>0000014693</td></tr>
      <tr><td>ZRO1</td><td>Zero CIK One</td><td>Energy</td><td>x</td><td>x</td><td>2015-01-01</td><td>0000000000</td></tr>
      <tr><td>ZRO2</td><td>Zero CIK Two</td><td>Energy</td><td>x</td><td>x</td><td>2015-01-01</td><td>0</td></tr>
      <tr><td>NA1</td><td>NA CIK One</td><td>Materials</td><td>x</td><td>x</td><td>2015-01-01</td><td>N/A</td></tr>
      <tr><td>NA2</td><td>NA CIK Two</td><td>Materials</td><td>x</td><td>x</td><td>2015-01-01</td><td>—</td></tr>
    </table></body></html>
    """
    parsed = parse_wikipedia_sp500_html(html)
    by_sym = {c.symbol: c for c in parsed}
    assert by_sym["ZRO1"].cik == ""
    assert by_sym["ZRO2"].cik == ""
    assert by_sym["NA1"].cik == ""
    assert by_sym["NA2"].cik == ""

    deduped, pre_count = select_point_in_time_candidates(parsed, cutoff_date=date(2021, 11, 23))
    assert pre_count == 8
    deduped_syms = {c.symbol for c in deduped}
    assert deduped_syms == {"BRK.B", "BF.B", "ZRO1", "ZRO2", "NA1", "NA2"}


def test_cli_rejects_live_parquet_dir_and_handles_missing_macro_table(tmp_path: Path) -> None:
    """MEDIUM-2: Reject data/parquet research_dir cleanly and handle StateDB without macro_snapshots."""
    runner = CliRunner()
    repo_config = Path(__file__).resolve().parents[1] / "config"

    with pytest.raises(ValueError, match="must not"):
        validate_research_dir(Path("data/parquet"))
    with pytest.raises(ValueError, match="must not"):
        validate_research_dir(tmp_path / "data" / "parquet" / "sub", data_dir=tmp_path / "data")

    res_bad = runner.invoke(
        app,
        [
            "sp500-cohorts",
            "--research-dir",
            "data/parquet",
            "--output-dir",
            str(tmp_path / "out"),
            "--config-dir",
            str(repo_config),
            "--json",
        ],
    )
    assert res_bad.exit_code != 0
    bad_payload = json.loads(res_bad.stdout)
    assert bad_payload["ok"] is False

    # Empty SQLite db without macro_snapshots table should gracefully fall back to macro_points=[]
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(data_dir / "state.db") as conn:
        conn.execute("CREATE TABLE dummy (id INTEGER PRIMARY KEY)")

    research_dir = tmp_path / "research"
    start_d = date.today() - timedelta(days=210)
    dates = [d for d in _trading_dates(160, start=start_d) if d <= date.today()]
    rng = np.random.default_rng(17)
    store = ParquetBarStore(research_dir)
    store.save_bars(
        "SPY",
        _make_bars_df(dates, 400.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, len(dates)))),
        timeframe="1d",
    )
    constituents: list[dict[str, object]] = []
    for i in range(6):
        sym = f"NOMAC{i}"
        store.save_bars(
            sym,
            _make_bars_df(dates, 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.015, len(dates)))),
            timeframe="1d",
        )
        constituents.append(
            {
                "symbol": sym,
                "security": sym,
                "gics_sector": "Technology",
                "sector": "Technology",
                "date_added": "2018-01-01",
                "cik": str(3000 + i),
            }
        )
    (research_dir / "sp500_constituents.json").write_text(json.dumps(constituents), encoding="utf-8")

    res_ok = runner.invoke(
        app,
        [
            "sp500-cohorts",
            "--n-cohorts",
            "1",
            "--cohort-size",
            "4",
            "--cutoff-date",
            dates[0].isoformat(),
            "--research-dir",
            str(research_dir),
            "--output-dir",
            str(tmp_path / "out_ok"),
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(repo_config),
            "--json",
        ],
    )
    assert res_ok.exit_code == 0, res_ok.output


def test_fetch_and_cache_mixed_staleness_vs_spy(tmp_path: Path) -> None:
    """HIGH-2: Cached constituent trailing SPY by >4 days is treated as stale and queued for fetch."""
    dates = _trading_dates(30, start=date(2026, 10, 1))
    research_dir = tmp_path / "research_cache"
    store = ParquetBarStore(research_dir)
    store.save_bars("SPY", _make_bars_df(dates, np.linspace(400.0, 410.0, len(dates))), timeframe="1d")
    store.save_bars("FRESH", _make_bars_df(dates, np.linspace(100.0, 105.0, len(dates))), timeframe="1d")
    store.save_bars(
        "STALE",
        _make_bars_df(dates[:-8], np.linspace(50.0, 52.0, len(dates) - 8)),
        timeframe="1d",
    )

    start_utc = datetime(dates[0].year, dates[0].month, dates[0].day, 0, 0, tzinfo=UTC)
    end_utc = datetime(dates[-1].year, dates[-1].month, dates[-1].day, 23, 59, tzinfo=UTC)

    # FRESH succeeds completely offline
    offline_res = fetch_and_cache_sp500_bars(
        ["FRESH"],
        research_dir,
        api_key="",
        secret_key="",
        start_utc=start_utc,
        end_utc=end_utc,
    )
    assert set(offline_res.keys()) == {"SPY", "FRESH"}

    # STALE trails SPY max by >4 days -> triggers Alpaca fetch (requiring credentials)
    with pytest.raises(ValueError, match="Alpaca credentials are required"):
        fetch_and_cache_sp500_bars(
            ["FRESH", "STALE"],
            research_dir,
            api_key="",
            secret_key="",
            start_utc=start_utc,
            end_utc=end_utc,
        )

