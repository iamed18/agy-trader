from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

_EULER_GAMMA = 0.5772156649015329
_SQRT_2 = math.sqrt(2.0)

# Acklam rational approximation coefficients for inverse standard normal CDF
_ACKLAM_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function via math.erf."""
    if not math.isfinite(x):
        return 1.0 if x > 0.0 else 0.0
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF via Peter Acklam's rational approximation (~1e-9 accuracy)."""
    if not math.isfinite(p) or p <= 0.0 or p >= 1.0:
        raise ValueError(f"Probability p must be strictly in (0, 1); got {p!r}")

    p_low = 0.02425
    p_high = 1.0 - p_low
    a, b, c, d = _ACKLAM_A, _ACKLAM_B, _ACKLAM_C, _ACKLAM_D

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def compute_deflated_sharpe_ratio(
    daily_excess_returns: Sequence[float],
    n_trials: int,
) -> float:
    """Compute Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio probability in [0, 1].

    Corrects the observed per-period Sharpe ratio for skewness, excess kurtosis,
    sample length, and multiple-testing selection bias across ``n_trials`` independent trials.
    """
    arr = np.asarray(daily_excess_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    t_len = int(arr.size)
    if t_len < 3:
        return 0.0

    std_r = float(np.std(arr, ddof=1))
    if not math.isfinite(std_r) or std_r <= 1e-12:
        return 0.0

    mean_r = float(np.mean(arr))
    sr_hat = mean_r / std_r  # Non-annualized per-period Sharpe

    pop_std = float(np.std(arr, ddof=0))
    if pop_std > 1e-12:
        z = (arr - mean_r) / pop_std
        skew = float(np.mean(z**3))
        kurtosis = float(np.mean(z**4))  # Raw Pearson kurtosis (3.0 for Normal)
    else:
        skew = 0.0
        kurtosis = 3.0

    eff_trials = max(1, int(n_trials))
    if eff_trials <= 1:
        sr_benchmark = 0.0
    else:
        p1 = min(1.0 - 1e-12, max(1e-12, 1.0 - 1.0 / eff_trials))
        p2 = min(1.0 - 1e-12, max(1e-12, 1.0 - 1.0 / (eff_trials * math.e)))
        e_max_z = (1.0 - _EULER_GAMMA) * _norm_ppf(p1) + _EULER_GAMMA * _norm_ppf(p2)
        sr_benchmark = e_max_z / math.sqrt(float(t_len - 1))

    sr_var_num = 1.0 - skew * sr_hat + ((kurtosis - 1.0) / 4.0) * (sr_hat**2)
    if not math.isfinite(sr_var_num) or sr_var_num <= 1e-12:
        return 0.5
    sr_std = math.sqrt(sr_var_num / float(t_len - 1))
    z_dsr = (sr_hat - sr_benchmark) / max(sr_std, 1e-12)
    return max(0.0, min(1.0, _norm_cdf(z_dsr)))


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
    excess_sharpe_ratio: float = 0.0
    deflated_sharpe_prob: float | None = None
    n_trials_assumed: int = 1
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
    total_cash_yield_usd: float = 0.0
    benchmark_symbol: str
    benchmark_return_pct: float
    benchmark_sharpe: float
    benchmark_max_drawdown_pct: float
    ew_universe_return_pct: float = 0.0
    ew_universe_sharpe: float = 0.0
    ew_universe_max_drawdown_pct: float = 0.0
    execution_timing: str = "same_close"


class CPCVSummary(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    n_groups: int
    k_test_groups: int
    purge_days: int
    embargo_days: int
    n_paths: int
    median_sharpe: float
    mean_sharpe: float
    p25_sharpe: float
    p75_sharpe: float
    min_sharpe: float
    max_sharpe: float
    pct_positive_paths: float
    median_excess_sharpe: float
    fold_metrics: list[BacktestMetrics] = Field(default_factory=list)


class BacktestResult(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    metrics: BacktestMetrics
    equity_curve: list[EquityPoint] = Field(default_factory=list)
    trades: list[SimulatedTrade] = Field(default_factory=list)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)


def _extract_period_returns(equities: Sequence[float]) -> tuple[np.ndarray, float, float]:
    """Return (finite_period_returns, cagr, max_dd) safely handling bankruptcy <= 0."""
    arr = np.asarray(equities, dtype=float)
    if arr.size < 2 or arr[0] <= 0.0:
        return np.array([], dtype=float), 0.0, 0.0

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
    return rets, cagr, max_dd


def _compute_return_stats(equities: Sequence[float]) -> tuple[float, float, float, float, float]:
    """Return (cagr, ann_vol, sharpe, sortino, max_dd) safely handling bankruptcy <= 0."""
    rets, cagr, max_dd = _extract_period_returns(equities)
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
    daily_rf_rates: Sequence[float] | None = None,
    ew_universe_equities: Sequence[float] | None = None,
    total_cash_yield_usd: float = 0.0,
    n_trials: int = 1,
    execution_timing: str = "same_close",
) -> BacktestMetrics:
    eff_trials = max(1, int(n_trials))
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
            excess_sharpe_ratio=0.0,
            deflated_sharpe_prob=None,
            n_trials_assumed=eff_trials,
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
            total_cash_yield_usd=round(total_cash_yield_usd, 2),
            benchmark_symbol=benchmark_symbol,
            benchmark_return_pct=0.0,
            benchmark_sharpe=0.0,
            benchmark_max_drawdown_pct=0.0,
            ew_universe_return_pct=0.0,
            ew_universe_sharpe=0.0,
            ew_universe_max_drawdown_pct=0.0,
            execution_timing=execution_timing,
        )

    equities = [pt.equity for pt in equity_curve]
    rets, _, _ = _extract_period_returns(equities)
    cagr, ann_vol, sharpe, sortino, max_dd = _compute_return_stats(equities)
    total_ret = (equities[-1] / initial_capital) - 1.0
    calmar = (cagr / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0

    # Excess returns & Deflated Sharpe Ratio
    if daily_rf_rates is not None and rets.size > 0:
        rf_arr = np.asarray(daily_rf_rates, dtype=float)
        if rf_arr.size == len(equity_curve):
            rf_aligned = rf_arr[1 : 1 + rets.size]
        elif rf_arr.size >= rets.size:
            rf_aligned = rf_arr[: rets.size]
        else:
            rf_aligned = np.zeros_like(rets)
        rf_aligned = np.where(np.isfinite(rf_aligned), rf_aligned, 0.0)
        excess_rets = rets - rf_aligned
    else:
        excess_rets = rets

    ex_std = float(np.std(excess_rets, ddof=1)) if excess_rets.size > 1 else 0.0
    if excess_rets.size > 1 and ex_std > 1e-7:
        ex_mean = float(np.mean(excess_rets))
        excess_sharpe = ex_mean / ex_std * math.sqrt(252.0)
    else:
        excess_sharpe = 0.0

    dsr_prob: float | None = None
    if excess_rets.size >= 3 and ex_std > 1e-7:
        dsr_prob = round(compute_deflated_sharpe_ratio(excess_rets.tolist(), eff_trials), 4)

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

    # Equal-Weight Universe benchmark stats
    if ew_universe_equities is not None and len(ew_universe_equities) >= 2 and ew_universe_equities[0] > 0.0:
        _, _, ew_sharpe, _, ew_max_dd = _compute_return_stats(ew_universe_equities)
        ew_ret = (ew_universe_equities[-1] / ew_universe_equities[0]) - 1.0
    else:
        ew_ret, ew_sharpe, ew_max_dd = 0.0, 0.0, 0.0

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
        excess_sharpe_ratio=round(excess_sharpe, 3),
        deflated_sharpe_prob=dsr_prob,
        n_trials_assumed=eff_trials,
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
        total_cash_yield_usd=round(total_cash_yield_usd, 2),
        benchmark_symbol=benchmark_symbol,
        benchmark_return_pct=round(bm_ret * 100.0, 2),
        benchmark_sharpe=round(bm_sharpe, 3),
        benchmark_max_drawdown_pct=round(bm_max_dd * 100.0, 2),
        ew_universe_return_pct=round(ew_ret * 100.0, 2),
        ew_universe_sharpe=round(ew_sharpe, 3),
        ew_universe_max_drawdown_pct=round(ew_max_dd * 100.0, 2),
        execution_timing=execution_timing,
    )


def render_ablation_markdown(
    results: Sequence[BacktestResult],
    *,
    title: str = "Phase 1 Backtest & 4-Way A/B Ablation Report",
    walk_forward_folds: Sequence[BacktestMetrics] | None = None,
    cpcv_summary: CPCVSummary | None = None,
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
    lines.append(f"- **Execution Timing:** `{ref.execution_timing}`")
    lines.append(f"- **DSR Assumed Trials:** `{ref.n_trials_assumed}`")
    lines.append(
        f"- **Benchmark ({ref.benchmark_symbol} Buy & Hold):** "
        f"Return `{ref.benchmark_return_pct:+.2f}%`, "
        f"Sharpe `{ref.benchmark_sharpe:.3f}`, "
        f"Max Drawdown `{ref.benchmark_max_drawdown_pct:.2f}%`"
    )
    lines.append(
        f"- **Benchmark (Equal-Weight Universe):** "
        f"Return `{ref.ew_universe_return_pct:+.2f}%`, "
        f"Sharpe `{ref.ew_universe_sharpe:.3f}`, "
        f"Max Drawdown `{ref.ew_universe_max_drawdown_pct:.2f}%`"
    )
    lines.append("")
    lines.append("## Strategy Variant Comparison")
    lines.append("")
    lines.append(
        "| Variant | Final Equity | Total Ret | CAGR | Ann Vol | Sharpe (rf=0) | Excess Sharpe | "
        "DSR | Sortino | Max DD | Win Rate | Profit Factor | Turnover | Cash Yield | Slippage | Borrow Cost | Stops |"
    )
    lines.append(
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for res in results:
        m = res.metrics
        dsr_str = f"{m.deflated_sharpe_prob:.3f}" if m.deflated_sharpe_prob is not None else "N/A"
        lines.append(
            f"| {m.label} | ${m.final_equity:,.2f} | {m.total_return_pct:+.2f}% | "
            f"{m.cagr_pct:+.2f}% | {m.ann_volatility_pct:.2f}% | {m.sharpe_ratio:.3f} | "
            f"{m.excess_sharpe_ratio:.3f} | {dsr_str} | "
            f"{m.sortino_ratio:.3f} | {m.max_drawdown_pct:.2f}% | {m.win_rate_pct:.1f}% | "
            f"{m.profit_factor:.2f} | {m.annual_turnover:.2f}x | ${m.total_cash_yield_usd:,.2f} | "
            f"${m.total_slippage_usd:,.2f} | ${m.total_borrow_cost_usd:,.2f} | {m.stop_loss_exits} |"
        )

    if walk_forward_folds:
        lines.extend(["", "## Walk-Forward Out-of-Sample Folds (Fixed-Parameter Chronological Split)", ""])
        lines.append("| Fold | Window | Total Ret | CAGR | Sharpe | Excess Sharpe | Max DD | Closed Trades |")
        lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for fm in walk_forward_folds:
            lines.append(
                f"| {fm.label} | `{fm.start_date}`..`{fm.end_date}` | "
                f"{fm.total_return_pct:+.2f}% | {fm.cagr_pct:+.2f}% | "
                f"{fm.sharpe_ratio:.3f} | {fm.excess_sharpe_ratio:.3f} | "
                f"{fm.max_drawdown_pct:.2f}% | {fm.closed_trades} |"
            )

    if cpcv_summary is not None:
        lines.extend(["", "## Combinatorial Purged Cross-Validation (CPCV)", ""])
        lines.append(
            f"- **Configuration:** `N={cpcv_summary.n_groups}` groups, "
            f"`k={cpcv_summary.k_test_groups}` test groups, "
            f"`purge={cpcv_summary.purge_days}d`, `embargo={cpcv_summary.embargo_days}d`, "
            f"`paths={cpcv_summary.n_paths}`"
        )
        lines.append(
            f"- **Sharpe Distribution:** Median `{cpcv_summary.median_sharpe:.3f}` | "
            f"Mean `{cpcv_summary.mean_sharpe:.3f}` | "
            f"IQR `[{cpcv_summary.p25_sharpe:.3f}, {cpcv_summary.p75_sharpe:.3f}]` | "
            f"Min/Max `[{cpcv_summary.min_sharpe:.3f}, {cpcv_summary.max_sharpe:.3f}]` | "
            f"Positive Paths `{cpcv_summary.pct_positive_paths:.1f}%` | "
            f"Median Excess Sharpe `{cpcv_summary.median_excess_sharpe:.3f}`"
        )
        if cpcv_summary.fold_metrics:
            lines.append("")
            lines.append("| Path | Window | Total Ret | Sharpe | Excess Sharpe | Max DD | Closed Trades |")
            lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
            for pm in cpcv_summary.fold_metrics:
                lines.append(
                    f"| {pm.label} | `{pm.start_date}`..`{pm.end_date}` | "
                    f"{pm.total_return_pct:+.2f}% | {pm.sharpe_ratio:.3f} | "
                    f"{pm.excess_sharpe_ratio:.3f} | {pm.max_drawdown_pct:.2f}% | {pm.closed_trades} |"
                )

    timing_caveat = (
        "- **Same-Close Execution Alignment (`same_close`):** Signals computed at bar $t$ close "
        "execute at bar $t$ close + adverse slippage (matching `15:40 ET` live execution). "
        "Stop-losses for new entries are anchored to fill price and first evaluated on bar $t+1$."
        if ref.execution_timing == "same_close"
        else "- **Zero Same-Bar Lookahead (`next_open`):** Signals computed at bar $t$ close "
        "execute at bar $t+1$ open with adverse slippage."
    )

    lines.extend(
        [
            "",
            "## Methodology & Caveats",
            "",
            timing_caveat,
            "- **Gap-Through Stop Realism:** Stop-losses breaching across an overnight gap fill at the worse bar open price plus adverse slippage.",
            "- **T-Bill Cash Yield & Excess Sharpe:** Idle positive cash accrues point-in-time FRED `DGS3MO` (or fallback APR) yield; Excess Sharpe and Deflated Sharpe Ratio (DSR) subtract the daily risk-free rate.",
            "- **Short Costs:** Annualized short borrow fees accrue daily on open short notional; SEC sell fees apply on all sells.",
            "- **Survivorship Bias & IEX Caveat:** Backtesting today's curated large-cap watchlist over historical years introduces mild survivorship bias, and Alpaca free-tier IEX bars represent a subset of US consolidated volume. Treat backtest Sharpe conservatively and validate via multi-week paper trading.",
            "",
        ]
    )
    return "\n".join(lines)
