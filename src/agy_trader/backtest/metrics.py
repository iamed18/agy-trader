from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class EquityPoint(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    timestamp: datetime
    equity: float
    cash: float
    gross_exposure: float
    net_exposure: float
    daily_pnl_pct: float
    drawdown_pct: float


class SimulatedTrade(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    symbol: str
    side: str  # "long" | "short"
    entry_time: datetime
    exit_time: datetime
    qty: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float  # Net of SEC fees & pro-rata borrow cost (slippage already in fill prices)
    holding_days: float
    exit_reason: str


class BacktestMetrics(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    label: str
    start_date: str
    end_date: str
    trading_days: int
    initial_capital: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    ann_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    avg_holding_days: float
    annual_turnover: float
    total_orders_filled: int
    closed_trades: int
    stop_loss_exits: int
    risk_rejections: int
    total_slippage_usd: float
    total_borrow_cost_usd: float
    total_sec_fees_usd: float
    benchmark_symbol: str
    benchmark_return_pct: float
    benchmark_sharpe: float
    benchmark_max_drawdown_pct: float


class BacktestResult(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    metrics: BacktestMetrics
    equity_curve: list[EquityPoint] = Field(default_factory=list)
    trades: list[SimulatedTrade] = Field(default_factory=list)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)


def _compute_return_stats(equities: Sequence[float]) -> tuple[float, float, float, float, float]:
    """Return (cagr, ann_vol, sharpe, sortino, max_dd) safely handling bankruptcy <= 0."""
    arr = np.asarray(equities, dtype=float)
    if arr.size < 2 or arr[0] <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Stop return series at first bankruptcy point (<= 0.0) so we never divide by ~0.
    non_pos = np.where(arr <= 0.0)[0]
    if non_pos.size > 0:
        first_zero = int(non_pos[0])
        clipped = np.maximum(arr[: first_zero + 1], 0.0)
        rets = np.diff(clipped) / clipped[:-1] if clipped.size >= 2 else np.array([], dtype=float)
        max_dd = -1.0
        cagr = -1.0
    else:
        rets = np.diff(arr) / arr[:-1]
        n_periods = rets.size
        total_ratio = float(arr[-1] / arr[0])
        cagr = (total_ratio ** (252.0 / n_periods)) - 1.0 if n_periods > 0 else 0.0
        running_max = np.maximum.accumulate(arr)
        drawdowns = (arr - running_max) / running_max
        max_dd = float(np.min(drawdowns))

    rets = rets[np.isfinite(rets)]
    if rets.size == 0:
        return cagr, 0.0, 0.0, 0.0, max_dd

    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0
    ann_vol = std_r * math.sqrt(252.0)
    sharpe = (mean_r / std_r * math.sqrt(252.0)) if std_r > 1e-12 else 0.0

    # Downside deviation using target return 0.0 over all N periods (standard Sortino)
    downside = np.minimum(rets, 0.0)
    downside_dev = math.sqrt(float(np.mean(downside**2)))
    sortino = (mean_r / downside_dev * math.sqrt(252.0)) if downside_dev > 1e-12 else 0.0

    return cagr, ann_vol, sharpe, sortino, max_dd


def compute_backtest_metrics(
    *,
    label: str,
    equity_curve: Sequence[EquityPoint],
    trades: Sequence[SimulatedTrade],
    initial_capital: float,
    total_traded_notional: float,
    total_orders_filled: int,
    stop_loss_exits: int,
    risk_rejections: int,
    total_slippage_usd: float,
    total_borrow_cost_usd: float,
    total_sec_fees_usd: float,
    benchmark_symbol: str,
    benchmark_closes: Sequence[float],
) -> BacktestMetrics:
    if not equity_curve:
        return BacktestMetrics(
            label=label,
            start_date="",
            end_date="",
            trading_days=0,
            initial_capital=initial_capital,
            final_equity=initial_capital,
            total_return_pct=0.0,
            cagr_pct=0.0,
            ann_volatility_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown_pct=0.0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            avg_holding_days=0.0,
            annual_turnover=0.0,
            total_orders_filled=total_orders_filled,
            closed_trades=len(trades),
            stop_loss_exits=stop_loss_exits,
            risk_rejections=risk_rejections,
            total_slippage_usd=round(total_slippage_usd, 2),
            total_borrow_cost_usd=round(total_borrow_cost_usd, 2),
            total_sec_fees_usd=round(total_sec_fees_usd, 2),
            benchmark_symbol=benchmark_symbol,
            benchmark_return_pct=0.0,
            benchmark_sharpe=0.0,
            benchmark_max_drawdown_pct=0.0,
        )

    equities = [pt.equity for pt in equity_curve]
    cagr, ann_vol, sharpe, sortino, max_dd = _compute_return_stats(equities)
    total_ret = (equities[-1] / initial_capital) - 1.0
    calmar = (cagr / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0

    # Trade-level stats
    wins = [t.net_pnl for t in trades if t.net_pnl > 0.0]
    losses = [-t.net_pnl for t in trades if t.net_pnl < 0.0]
    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
    gross_win = sum(wins)
    gross_loss = sum(losses)
    if gross_loss > 1e-9:
        profit_factor = gross_win / gross_loss
    elif gross_win > 1e-9:
        profit_factor = 999.0  # Capped finite sentinel for zero-loss runs
    else:
        profit_factor = 0.0

    avg_hold = (sum(t.holding_days for t in trades) / len(trades)) if trades else 0.0
    avg_equity = float(np.mean(equities))
    years = max(len(equity_curve) / 252.0, 1.0 / 252.0)
    # One-way annualized turnover: (total traded notional / 2) / (avg_equity * years)
    annual_turnover = (
        (total_traded_notional / 2.0) / (avg_equity * years) if avg_equity > 1e-9 else 0.0
    )

    # Benchmark stats
    if len(benchmark_closes) >= 2 and benchmark_closes[0] > 0.0:
        _, _, bm_sharpe, _, bm_max_dd = _compute_return_stats(benchmark_closes)
        bm_ret = (benchmark_closes[-1] / benchmark_closes[0]) - 1.0
    else:
        bm_ret, bm_sharpe, bm_max_dd = 0.0, 0.0, 0.0

    return BacktestMetrics(
        label=label,
        start_date=equity_curve[0].timestamp.date().isoformat(),
        end_date=equity_curve[-1].timestamp.date().isoformat(),
        trading_days=len(equity_curve),
        initial_capital=round(initial_capital, 2),
        final_equity=round(equities[-1], 2),
        total_return_pct=round(total_ret * 100.0, 2),
        cagr_pct=round(cagr * 100.0, 2),
        ann_volatility_pct=round(ann_vol * 100.0, 2),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        calmar_ratio=round(calmar, 3),
        max_drawdown_pct=round(max_dd * 100.0, 2),
        win_rate_pct=round(win_rate, 2),
        profit_factor=round(min(profit_factor, 999.0), 3),
        avg_holding_days=round(avg_hold, 2),
        annual_turnover=round(annual_turnover, 2),
        total_orders_filled=total_orders_filled,
        closed_trades=len(trades),
        stop_loss_exits=stop_loss_exits,
        risk_rejections=risk_rejections,
        total_slippage_usd=round(total_slippage_usd, 2),
        total_borrow_cost_usd=round(total_borrow_cost_usd, 2),
        total_sec_fees_usd=round(total_sec_fees_usd, 2),
        benchmark_symbol=benchmark_symbol,
        benchmark_return_pct=round(bm_ret * 100.0, 2),
        benchmark_sharpe=round(bm_sharpe, 3),
        benchmark_max_drawdown_pct=round(bm_max_dd * 100.0, 2),
    )


def render_ablation_markdown(
    results: Sequence[BacktestResult],
    *,
    title: str = "Phase 1 Backtest & 4-Way A/B Ablation Report",
    walk_forward_folds: Sequence[BacktestMetrics] | None = None,
) -> str:
    """Render a structured Markdown comparison table with explicit survivorship/data caveats."""
    lines: list[str] = [f"# {title}", ""]
    if not results:
        lines.append("_No backtest results available._")
        return "\n".join(lines)

    ref = results[0].metrics
    lines.append(
        f"- **Window:** `{ref.start_date}` to `{ref.end_date}` ({ref.trading_days} trading days)"
    )
    lines.append(f"- **Initial Capital:** `${ref.initial_capital:,.2f}`")
    lines.append(
        f"- **Benchmark ({ref.benchmark_symbol} Buy & Hold):** "
        f"Return `{ref.benchmark_return_pct:+.2f}%`, "
        f"Sharpe `{ref.benchmark_sharpe:.3f}`, "
        f"Max Drawdown `{ref.benchmark_max_drawdown_pct:.2f}%`"
    )
    lines.append("")
    lines.append("## Strategy Variant Comparison")
    lines.append("")
    lines.append(
        "| Variant | Final Equity | Total Ret | CAGR | Ann Vol | Sharpe | Sortino | Max DD | "
        "Win Rate | Profit Factor | Turnover | Slippage | Borrow Cost | Stops |"
    )
    lines.append(
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for res in results:
        m = res.metrics
        lines.append(
            f"| {m.label} | ${m.final_equity:,.2f} | {m.total_return_pct:+.2f}% | "
            f"{m.cagr_pct:+.2f}% | {m.ann_volatility_pct:.2f}% | {m.sharpe_ratio:.3f} | "
            f"{m.sortino_ratio:.3f} | {m.max_drawdown_pct:.2f}% | {m.win_rate_pct:.1f}% | "
            f"{m.profit_factor:.2f} | {m.annual_turnover:.2f}x | ${m.total_slippage_usd:,.2f} | "
            f"${m.total_borrow_cost_usd:,.2f} | {m.stop_loss_exits} |"
        )

    if walk_forward_folds:
        lines.extend(["", "## Walk-Forward Out-of-Sample Folds", ""])
        lines.append("| Fold | Window | Total Ret | CAGR | Sharpe | Max DD | Closed Trades |")
        lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
        for fm in walk_forward_folds:
            lines.append(
                f"| {fm.label} | `{fm.start_date}`..`{fm.end_date}` | "
                f"{fm.total_return_pct:+.2f}% | {fm.cagr_pct:+.2f}% | "
                f"{fm.sharpe_ratio:.3f} | {fm.max_drawdown_pct:.2f}% | {fm.closed_trades} |"
            )

    lines.extend(
        [
            "",
            "## Methodology & Caveats",
            "",
            "- **Zero Same-Bar Lookahead:** Signals computed at bar $t$ close execute at bar $t+1$ open with adverse slippage.",
            "- **Gap-Through Stop Realism:** Stop-losses breaching across an overnight gap fill at the worse bar open price plus adverse slippage.",
            "- **Short Costs:** Annualized short borrow fees accrue daily on open short notional; SEC sell fees apply on all sells.",
            "- **Survivorship Bias & IEX Caveat:** Backtesting today's curated large-cap watchlist over historical years introduces mild survivorship bias, and Alpaca free-tier IEX bars represent a subset of US consolidated volume. Treat backtest Sharpe conservatively and validate via multi-week paper trading.",
            "",
        ]
    )
    return "\n".join(lines)
