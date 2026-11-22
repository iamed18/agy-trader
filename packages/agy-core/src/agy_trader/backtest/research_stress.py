from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, ConfigDict, Field

from agy_trader.core.config import AppConfig, load_app_config
from agy_trader.core.models import AssetType, MacroPoint, UniverseSymbol
from agy_trader.storage.parquet_store import ParquetBarStore

from .engine import BacktestEngine
from .metrics import BacktestMetrics, BacktestResult, CPCVSummary

logger = logging.getLogger(__name__)
NY_TZ = ZoneInfo("America/New_York")

RESEARCH_ETF_UNIVERSE: tuple[UniverseSymbol, ...] = (
    UniverseSymbol(symbol="SPY", asset_type=AssetType.etf, sector="Broad Market"),
    UniverseSymbol(symbol="QQQ", asset_type=AssetType.etf, sector="Broad Market"),
    UniverseSymbol(symbol="IWM", asset_type=AssetType.etf, sector="Broad Market"),
    UniverseSymbol(symbol="TLT", asset_type=AssetType.etf, sector="Fixed Income"),
    UniverseSymbol(symbol="IEF", asset_type=AssetType.etf, sector="Fixed Income"),
    UniverseSymbol(symbol="SHY", asset_type=AssetType.etf, sector="Fixed Income"),
    UniverseSymbol(symbol="GLD", asset_type=AssetType.etf, sector="Commodities"),
    UniverseSymbol(symbol="XLK", asset_type=AssetType.etf, sector="Technology"),
    UniverseSymbol(symbol="XLF", asset_type=AssetType.etf, sector="Financials"),
    UniverseSymbol(symbol="XLE", asset_type=AssetType.etf, sector="Energy"),
    UniverseSymbol(symbol="XLV", asset_type=AssetType.etf, sector="Healthcare"),
    UniverseSymbol(symbol="XLI", asset_type=AssetType.etf, sector="Industrials"),
    UniverseSymbol(symbol="XLY", asset_type=AssetType.etf, sector="Consumer Discretionary"),
    UniverseSymbol(symbol="XLP", asset_type=AssetType.etf, sector="Consumer Staples"),
    UniverseSymbol(symbol="XLU", asset_type=AssetType.etf, sector="Utilities"),
    UniverseSymbol(symbol="XLB", asset_type=AssetType.etf, sector="Materials"),
)


@dataclass(frozen=True, slots=True)
class RegimeWindow:
    name: str
    start_date: date
    end_date: date


REGIME_WINDOWS: tuple[RegimeWindow, ...] = (
    RegimeWindow("2008 GFC", date(2008, 1, 2), date(2009, 3, 31)),
    RegimeWindow("2011 Euro Crisis", date(2011, 5, 2), date(2011, 12, 30)),
    RegimeWindow("2015-2016 Chop", date(2015, 5, 1), date(2016, 6, 30)),
    RegimeWindow("2018 Q4 Selloff", date(2018, 9, 20), date(2019, 1, 31)),
    RegimeWindow("2020 COVID + Nov Reversal", date(2020, 2, 18), date(2020, 12, 31)),
    RegimeWindow("2022 Bear Market", date(2022, 1, 3), date(2022, 12, 30)),
    RegimeWindow("2023-2026 Bull", date(2023, 1, 3), date(2026, 11, 20)),
)


class StressReport(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    full_result: BacktestResult
    regime_metrics: list[BacktestMetrics] = Field(default_factory=list)
    cpcv_summary: CPCVSummary | None = None


def _sanitize_yfinance_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a single-symbol yfinance OHLCV frame to canonical UTC 16:00 ET bars."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).strip().lower() for col in df.columns]
    else:
        df.columns = [str(col).strip().lower() for col in df.columns]

    required = ("open", "high", "low", "close", "volume")
    if any(col not in df.columns for col in required):
        return pd.DataFrame()

    idx_ts = pd.to_datetime(df.index)
    utc_timestamps = [
        datetime(ts.year, ts.month, ts.day, 16, 0, tzinfo=NY_TZ).astimezone(UTC)
        for ts in idx_ts
    ]

    work = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(utc_timestamps, tz="UTC"),
            "open": pd.to_numeric(df["open"], errors="coerce").astype("float64"),
            "high": pd.to_numeric(df["high"], errors="coerce").astype("float64"),
            "low": pd.to_numeric(df["low"], errors="coerce").astype("float64"),
            "close": pd.to_numeric(df["close"], errors="coerce").astype("float64"),
            "volume": pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).clip(lower=0.0).astype("float64"),
        }
    )
    finite_mask = np.isfinite(work[["open", "high", "low", "close", "volume"]].to_numpy()).all(axis=1)
    pos_mask = (work[["open", "high", "low", "close"]] > 0.0).all(axis=1)
    work = work.loc[finite_mask & pos_mask].copy()
    if work.empty:
        return pd.DataFrame()

    # Enforce strict OHLC invariants against minor dividend/split adjustment rounding
    work["high"] = np.maximum(work["high"], np.maximum(work["open"], work["close"]))
    work["low"] = np.minimum(work["low"], np.minimum(work["open"], work["close"]))
    work["vwap"] = (work["open"] + work["high"] + work["low"] + 2.0 * work["close"]) / 5.0
    work = work.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    return work


def fetch_research_etf_bars(
    research_dir: Path = Path("data/parquet_research"),
    *,
    start_date: date = date(2006, 1, 1),
    end_date: date | None = None,
    symbols: Sequence[str] | None = None,
    allow_overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch adjusted daily ETF bars via yfinance into an isolated research Parquet store."""
    target_syms = (
        [s.strip().upper() for s in symbols if s.strip()]
        if symbols is not None
        else [u.symbol for u in RESEARCH_ETF_UNIVERSE]
    )
    eff_end = end_date or datetime.now(UTC).date()
    yf_end = eff_end + timedelta(days=1)

    store = ParquetBarStore(research_dir)
    loaded: dict[str, pd.DataFrame] = {}

    for sym in target_syms:
        try:
            raw = yf.download(
                sym,
                start=start_date.isoformat(),
                end=yf_end.isoformat(),
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            clean = _sanitize_yfinance_frame(raw)
            if clean.empty:
                logger.warning("yfinance returned no valid bars for %s", sym)
                continue
            store.save_bars(sym, clean, timeframe="1d", allow_overwrite=allow_overwrite)
            loaded[sym] = store.load_bars(sym, timeframe="1d")
        except Exception as exc:  # noqa: BLE001 - best-effort per symbol
            logger.warning("Failed to fetch research ETF bars for %s: %s", sym, exc)
            continue

    return loaded


def load_research_etf_bars(
    research_dir: Path = Path("data/parquet_research"),
    *,
    symbols: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load ETF bars from the isolated research Parquet directory."""
    target_syms = (
        [s.strip().upper() for s in symbols if s.strip()]
        if symbols is not None
        else [u.symbol for u in RESEARCH_ETF_UNIVERSE]
    )
    store = ParquetBarStore(research_dir)
    return store.load_universe_bars(target_syms, timeframe="1d")


def run_multi_regime_stress(
    app_config: AppConfig,
    bars_by_symbol: Mapping[str, pd.DataFrame],
    *,
    universe: Sequence[UniverseSymbol] = RESEARCH_ETF_UNIVERSE,
    macro_points: Sequence[MacroPoint] | None = None,
    start_date: date | None = date(2006, 1, 1),
    end_date: date | None = None,
    regime_windows: Sequence[RegimeWindow] = REGIME_WINDOWS,
    run_cpcv: bool = False,
    n_trials: int = 1,
    execution_timing: Literal["same_close", "next_open"] | None = None,
    label: str = "ETF Multi-Regime Stress (2006-2026)",
) -> StressReport:
    """Run BacktestEngine across full multi-decade ETF history and key crisis/bull regimes."""
    engine = BacktestEngine(app_config, universe)
    full_result = engine.run(
        bars_by_symbol,
        macro_points=macro_points,
        start_date=start_date,
        end_date=end_date,
        label=label,
        allow_shorting=False,
        use_news=False,
        execution_timing=execution_timing,
        n_trials=n_trials,
    )

    regime_metrics: list[BacktestMetrics] = []
    for rw in regime_windows:
        if start_date is not None and rw.end_date < start_date:
            continue
        if end_date is not None and rw.start_date > end_date:
            continue
        eff_start = max(rw.start_date, start_date) if start_date is not None else rw.start_date
        eff_end = min(rw.end_date, end_date) if end_date is not None else rw.end_date
        res = engine.run(
            bars_by_symbol,
            macro_points=macro_points,
            start_date=eff_start,
            end_date=eff_end,
            label=rw.name,
            allow_shorting=False,
            use_news=False,
            execution_timing=execution_timing,
            n_trials=n_trials,
        )
        if res.metrics.trading_days > 0:
            regime_metrics.append(res.metrics)

    cpcv_summary: CPCVSummary | None = None
    if run_cpcv and full_result.metrics.trading_days > 0:
        cpcv_summary = engine.run_cpcv(
            bars_by_symbol,
            macro_points=macro_points,
            allow_shorting=False,
            use_news=False,
            execution_timing=execution_timing,
        )

    return StressReport(
        full_result=full_result,
        regime_metrics=regime_metrics,
        cpcv_summary=cpcv_summary,
    )


def render_stress_markdown(report: StressReport) -> str:
    """Render full-period and regime-by-regime stress results as Markdown."""
    m = report.full_result.metrics
    dsr_str = f"{m.deflated_sharpe_prob:.3f}" if m.deflated_sharpe_prob is not None else "N/A"
    lines = [
        f"# {m.label}",
        "",
        f"- **Window:** `{m.start_date}` to `{m.end_date}` ({m.trading_days} trading days)",
        f"- **Execution Timing:** `{m.execution_timing}`",
        (
            f"- **Strategy:** Return `{m.total_return_pct:+.2f}%`, CAGR `{m.cagr_pct:+.2f}%`, "
            f"Sharpe `{m.sharpe_ratio:.3f}`, Excess Sharpe `{m.excess_sharpe_ratio:.3f}`, "
            f"DSR `{dsr_str}`, Max DD `{m.max_drawdown_pct:.2f}%`, Cash Yield `${m.total_cash_yield_usd:,.2f}`"
        ),
        (
            f"- **Benchmark ({m.benchmark_symbol}):** Return `{m.benchmark_return_pct:+.2f}%`, "
            f"Sharpe `{m.benchmark_sharpe:.3f}`, Max DD `{m.benchmark_max_drawdown_pct:.2f}%`"
        ),
        (
            f"- **Equal-Weight ETF Universe:** Return `{m.ew_universe_return_pct:+.2f}%`, "
            f"Sharpe `{m.ew_universe_sharpe:.3f}`, Max DD `{m.ew_universe_max_drawdown_pct:.2f}%`"
        ),
        "",
    ]

    if report.regime_metrics:
        lines.extend(
            [
                "## Regime Breakdown",
                "",
                "| Regime | Window | Return | CAGR | Sharpe | Excess Sharpe | Max DD | SPY Ret | EW Ret |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for rm in report.regime_metrics:
            lines.append(
                f"| {rm.label} | `{rm.start_date}`..`{rm.end_date}` | "
                f"{rm.total_return_pct:+.2f}% | {rm.cagr_pct:+.2f}% | "
                f"{rm.sharpe_ratio:.3f} | {rm.excess_sharpe_ratio:.3f} | "
                f"{rm.max_drawdown_pct:.2f}% | {rm.benchmark_return_pct:+.2f}% | "
                f"{rm.ew_universe_return_pct:+.2f}% |"
            )
        lines.append("")

    if report.cpcv_summary is not None:
        cs = report.cpcv_summary
        lines.extend(
            [
                "## CPCV Distribution",
                "",
                (
                    f"- Paths: `{cs.n_paths}` (`N={cs.n_groups}`, `k={cs.k_test_groups}`, "
                    f"`purge={cs.purge_days}d`, `embargo={cs.embargo_days}d`)"
                ),
                (
                    f"- Median Sharpe: `{cs.median_sharpe:.3f}` | IQR: `[{cs.p25_sharpe:.3f}, {cs.p75_sharpe:.3f}]` "
                    f"| Min/Max: `[{cs.min_sharpe:.3f}, {cs.max_sharpe:.3f}]` | Positive: `{cs.pct_positive_paths:.1f}%`"
                ),
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:  # pragma: no cover - manual research CLI entrypoint
    cfg = load_app_config(Path("config"))
    bars = load_research_etf_bars()
    if not any(not df.empty for df in bars.values()):
        bars = fetch_research_etf_bars()
    report = run_multi_regime_stress(cfg, bars, run_cpcv=True)
    print(render_stress_markdown(report))


if __name__ == "__main__":  # pragma: no cover
    main()
