from __future__ import annotations

import io
from datetime import date, datetime
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from agy_ops.daemons.common import NY_TZ
from agy_ops.reporting.digests import _open_readonly_db, resolve_snapshot_benchmark_mark
from agy_trader.core.calendar import is_nyse_rth_timestamp
from agy_trader.core.models import ensure_utc
from agy_trader.storage.parquet_store import ParquetBarStore

_BENCHMARK_SPECS: tuple[tuple[str, str, str | None, str, float, float, int], ...] = (
    ("SPY", "spy_close", "SPY", "#fbbf24", 1.7, 0.92, 4),
    ("EW-50", "ew50_index", None, "#a78bfa", 1.6, 0.90, 3),
    ("SPMO", "spmo_close", "SPMO", "#34d399", 1.6, 0.90, 3),
    ("QMOM", "qmom_close", "QMOM", "#fb7185", 1.6, 0.90, 3),
)


def _parse_ts_et(raw_ts: Any) -> datetime | None:
    if raw_ts is None:
        return None
    try:
        ts_utc = (
            ensure_utc(raw_ts)
            if isinstance(raw_ts, datetime)
            else ensure_utc(datetime.fromisoformat(str(raw_ts)))
        )
    except (TypeError, ValueError):
        return None
    return ts_utc.astimezone(NY_TZ)


def render_cumulative_performance_png(
    data_dir: Path,
    *,
    limit: int = 5000,
) -> bytes | None:
    """Render an inline dark-theme cumulative return comparison PNG chart."""
    resolved_data = Path(data_dir)
    db = _open_readonly_db(resolved_data)
    if db is None:
        return None

    snaps = db.get_portfolio_snapshots(limit=limit)
    if len(snaps) < 2:
        return None

    first_snap = db.get_first_portfolio_snapshot()
    first_invested = db.get_first_invested_portfolio_snapshot()

    if first_invested is not None:
        first_ts = str(first_invested["timestamp"])
        window = [s for s in snaps if str(s["timestamp"]) >= first_ts]
        if len(window) == 1 and window[0].get("id") != first_invested.get("id"):
            window = [first_invested, *window]
        elif not window:
            window = [first_invested, *snaps]
        if len(window) < 2:
            return None
    else:
        window = list(snaps)

    parsed_window: list[tuple[dict[str, Any], datetime]] = []
    for snap in window:
        ts_et = _parse_ts_et(snap.get("timestamp"))
        if ts_et is not None:
            parsed_window.append((snap, ts_et))
    if len(parsed_window) < 2:
        return None

    # Keep first row, all NYSE RTH rows, and final row to compress dead overnight/weekend gaps.
    rth_indices = [
        idx
        for idx, (_, ts_et) in enumerate(parsed_window)
        if idx == 0 or idx == len(parsed_window) - 1 or is_nyse_rth_timestamp(ts_et)
    ]
    rth_body_count = sum(
        1 for _, ts_et in parsed_window if is_nyse_rth_timestamp(ts_et)
    )
    used_rth_filter = rth_body_count >= 2
    if len(rth_indices) >= 2 and used_rth_filter:
        seen: set[int] = set()
        selected: list[tuple[dict[str, Any], datetime]] = []
        for idx in rth_indices:
            if idx not in seen:
                seen.add(idx)
                selected.append(parsed_window[idx])
    else:
        selected = parsed_window

    if len(selected) < 2:
        return None

    anchor_snap: dict[str, Any] = first_snap if first_snap is not None else selected[0][0]
    base_eq = float(anchor_snap["equity"])
    if base_eq <= 0.0:
        base_eq = float(selected[0][0]["equity"])
    if base_eq <= 0.0:
        return None

    port_vals = [((float(s["equity"]) / base_eq) - 1.0) * 100.0 for s, _ in selected]

    bars_1d_dir = resolved_data / "parquet" / "bars_1d"
    bar_store = ParquetBarStore(resolved_data / "parquet") if bars_1d_dir.exists() else None
    bar_cache: dict[str, tuple[list[date], list[float]]] = {}

    bench_series: list[tuple[str, list[int], list[float], str, float, float, int]] = []
    for label, field, symbol, color, lw, alpha, zorder in _BENCHMARK_SPECS:
        marks: list[float | None] = []
        last_rth_mark: float | None = None
        for idx, (snap, ts_et) in enumerate(selected):
            is_rth = is_nyse_rth_timestamp(ts_et)
            if idx == 0 or not used_rth_filter or is_rth:
                mark = resolve_snapshot_benchmark_mark(
                    snap, field, symbol, bar_store, bar_cache=bar_cache
                )
                if mark is not None:
                    last_rth_mark = mark
                elif used_rth_filter and idx > 0:
                    mark = last_rth_mark
            else:
                mark = last_rth_mark
                if mark is None:
                    mark = resolve_snapshot_benchmark_mark(
                        snap, field, symbol, bar_store, bar_cache=bar_cache
                    )
            marks.append(mark)

        base_mark = marks[0]
        if base_mark is None or base_mark <= 0.0:
            continue
        xs: list[int] = []
        ys: list[float] = []
        for idx, mark in enumerate(marks):
            if mark is not None and mark > 0.0:
                xs.append(idx)
                ys.append(((mark / base_mark) - 1.0) * 100.0)
        if len(ys) < 2:
            continue
        bench_series.append((label, xs, ys, color, lw, alpha, zorder))

    fig = Figure(figsize=(8.0, 4.0), dpi=150, facecolor="#0d1320")
    ax = fig.add_subplot(111, facecolor="#131b2e")
    ax.axhline(0.0, color="#475569", linewidth=1.0, linestyle="--", zorder=1)
    ax.grid(True, color="#1e293b", linewidth=0.7, alpha=0.8)

    x_all = list(range(len(selected)))
    ax.plot(
        x_all,
        port_vals,
        color="#4ea1ff",
        linewidth=2.5,
        zorder=5,
        label=f"Portfolio ({port_vals[-1]:+.2f}%)",
    )
    ax.fill_between(x_all, port_vals, 0.0, color="#4ea1ff", alpha=0.08, zorder=2)

    for label, xs, ys, color, lw, alpha, zorder in bench_series:
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=lw,
            alpha=alpha,
            zorder=zorder,
            label=f"{label} ({ys[-1]:+.2f}%)",
        )

    n = len(selected)
    span_days = (selected[-1][1] - selected[0][1]).total_seconds() / 86400.0
    tick_fmt = "%b %d\n%H:%M" if span_days <= 5.0 else "%Y-%m-%d"
    if n <= 6:
        tick_idxs = list(range(n))
    else:
        tick_idxs = sorted({round(i * (n - 1) / 5) for i in range(6)})
    ax.set_xticks(tick_idxs)
    ax.set_xticklabels([selected[i][1].strftime(tick_fmt) for i in tick_idxs])

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.2f}%"))
    ax.tick_params(axis="y", colors="#94a3b8", labelsize=8.5)
    ax.tick_params(axis="x", colors="#94a3b8", labelsize=8.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#23304a")
    ax.spines["bottom"].set_color("#23304a")

    ax.set_title(
        "Cumulative Return vs. Benchmarks (Since First Trade)",
        color="#e2e8f0",
        fontsize=11,
        fontweight="bold",
        pad=26,
    )
    ax.legend(
        frameon=False,
        labelcolor="#e2e8f0",
        fontsize=8.0,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        handlelength=1.5,
        columnspacing=1.0,
    )
    fig.tight_layout(pad=1.0)

    buf = io.BytesIO()
    FigureCanvasAgg(fig)
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    return buf.getvalue()
