from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from agy_trader.backtest.v3_universe import V3_CORE_ETFS, V3EquityCandidate
from agy_trader.backtest.v4_monthly_research import (
    V4MonthlyResearchReport,
    build_v4_candidate_specs,
    build_v4_champion_app_config,
    render_v4_monthly_markdown,
    run_v4_monthly_research,
)
from agy_trader.core.config import AppConfig


def _make_syn_df(closes: np.ndarray, volumes: np.ndarray, dates: list[date]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [datetime(d.year, d.month, d.day, 21, 0, tzinfo=UTC) for d in dates],
            "open": closes * 0.999,
            "high": closes * 1.006,
            "low": closes * 0.994,
            "close": closes,
            "volume": volumes,
        }
    )


def test_v4_candidate_yaml_matches_champion_app_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    yaml_path = repo_root / "config" / "strategy_v4_monthly_candidate.yaml"
    assert yaml_path.exists()
    loaded = AppConfig.model_validate(yaml.safe_load(yaml_path.read_text(encoding="utf-8")))
    expected = build_v4_champion_app_config()
    assert loaded.strategy == expected.strategy
    assert loaded.risk == expected.risk
    assert loaded.backtest == expected.backtest
    assert loaded.llm_quota == expected.llm_quota


def test_v4_monthly_research_end_to_end_synthetic(tmp_path: Path) -> None:
    # Build 650 business days (~2.5 years) so IS & OOS windows + 252d rolling alpha run end-to-end
    dates: list[date] = []
    cur = date(2020, 1, 2)
    while len(dates) < 650:
        if cur.weekday() < 5:
            dates.append(cur)
        cur += timedelta(days=1)

    n = len(dates)
    rng = np.random.default_rng(123)
    spy_c = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.007, n)))
    bars: dict[str, pd.DataFrame] = {
        "SPY": _make_syn_df(spy_c, np.full(n, 50_000_000.0), dates),
    }
    for etf in V3_CORE_ETFS:
        if etf not in bars:
            bars[etf] = _make_syn_df(
                100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.007, n))),
                np.full(n, 5_000_000.0),
                dates,
            )

    sectors = ["Technology", "Healthcare", "Financials", "Industrials", "Consumer Discretionary"]
    pool: list[V3EquityCandidate] = []
    for i in range(18):
        sym = f"EQ{i:02d}"
        drift = 0.0002 + 0.0001 * (i % 5)
        cs = 50.0 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))
        vs = np.full(n, float((i + 1) * 300_000.0))
        bars[sym] = _make_syn_df(cs, vs, dates)
        pool.append(
            V3EquityCandidate(
                symbol=sym,
                security=f"Company {i}",
                sector=sectors[i % len(sectors)],
                sources=("sp500_curr",),
                sp500_intervals=(("2015-01-01", None),),
            )
        )

    out_dir = tmp_path / "v4_out"
    specs = build_v4_candidate_specs()
    report, _ = run_v4_monthly_research(
        bars,
        pool,
        is_start=dates[260],
        is_end=dates[449],
        oos_start=dates[450],
        oos_end=dates[-1],
        top_n=15,
        hysteresis=2,
        n_is_trials_cumulative=10,
        run_stress=True,
        run_cpcv=True,
        run_plateau=True,
        plateau_ks=(4, 6),
        plateau_bufs=(0, 2),
        candidate_specs=specs,
        output_dir=out_dir,
    )

    assert isinstance(report, V4MonthlyResearchReport)
    assert len(report.candidates) == len(specs)
    assert len(report.benchmarks) == 3
    assert len(report.stress_rows) > 0
    assert report.cpcv_full_summary is not None
    assert report.cpcv_full_summary.n_paths == 15
    assert all(
        date.fromisoformat(f.start_date) >= dates[260]
        for f in report.cpcv_full_summary.fold_metrics
    )
    assert len(report.plateau_cells) == 4
    assert len(report.plot_paths) == 5
    for p_str in report.plot_paths:
        p = Path(p_str)
        assert p.exists() and p.stat().st_size > 1_000
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "summary.md").exists()
    md = render_v4_monthly_markdown(report)
    assert report.champion_id in md
    assert "Plateau" in md
