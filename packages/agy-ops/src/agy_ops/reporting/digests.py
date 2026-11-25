from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from agy_ops.daemons.common import NY_TZ, is_us_market_open_fallback
from agy_trader.core.blackout import (
    BlackoutResolution,
    load_universe_for_blackout,
    resolve_live_blackout,
)
from agy_trader.core.calendar import (
    is_nyse_rth_timestamp,
    is_nyse_trading_day,
    nyse_trading_days_apart,
)
from agy_trader.core.config import AppConfig, EnvSettings, load_app_config, load_universe
from agy_trader.core.models import AssetType, ControlDials, MacroPoint, ensure_utc
from agy_trader.data.macro_and_calendar import FRED_SERIES_IDS
from agy_trader.storage.db import StateDB
from agy_trader.storage.parquet_store import ParquetBarStore
from agy_trader.strategy.features import compute_macro_regime_scale

DAEMON_NAMES: tuple[str, ...] = ("market-data", "news-nlp", "trader", "dashboard")


def _open_readonly_db(data_dir: Path) -> StateDB | None:
    db_path = Path(data_dir) / "state.db"
    if not db_path.exists():
        return None
    return StateDB(db_path, read_only=True)


def _safe_load_config(config_dir: Path) -> AppConfig:
    try:
        return load_app_config(config_dir)
    except (FileNotFoundError, OSError, ValueError):
        return AppConfig()


def _safe_load_universe(config_dir: Path) -> list[Any]:
    try:
        return load_universe(config_dir)
    except (FileNotFoundError, OSError, ValueError):
        return []


def _active_model(settings: EnvSettings) -> str:
    return "ProsusAI/finbert" if settings.nlp_backend == "finbert" else "lexicon-v1"


def _resolve_ops_blackout(
    config_dir: Path,
    db: StateDB | None,
    now_utc: datetime,
    settings: EnvSettings | None = None,
) -> BlackoutResolution:
    resolved_settings = settings or EnvSettings()
    universe, universe_error = load_universe_for_blackout(config_dir)
    env_path = Path(".env")
    obs = [p["symbol"] for p in db.get_live_positions()] if db is not None else ()
    return resolve_live_blackout(
        env_file=env_path if env_path.exists() else None,
        require_env_file=env_path.exists(),
        fallback_raw=resolved_settings.blackout_period,
        db=db,
        universe=universe,
        universe_error=universe_error,
        observed_broker_symbols=obs,
        now_utc=now_utc,
    )


def _expected_heartbeat_max_age(daemon_name: str, *, market_open: bool) -> float:
    if daemon_name == "news-nlp":
        return 1500.0
    if daemon_name == "trader":
        return 180.0 if market_open else 900.0
    if daemon_name == "market-data":
        return 240.0 if market_open else 900.0
    return 240.0 if market_open else 600.0


def build_readiness_report(
    data_dir: Path,
    config_dir: Path,
    *,
    now_utc: datetime | None = None,
    min_bars_required: int = 300,
) -> dict[str, Any]:
    """Build deterministic pre-market / operational readiness check report."""
    now = ensure_utc(now_utc) if now_utc is not None else datetime.now(UTC)
    now_et = now.astimezone(NY_TZ)
    market_open = is_us_market_open_fallback(now)

    resolved_data = Path(data_dir)
    resolved_config = Path(config_dir)
    app_cfg = _safe_load_config(resolved_config)
    universe = _safe_load_universe(resolved_config)
    settings = EnvSettings()
    expected_symbols = list(
        dict.fromkeys([u.symbol for u in universe] + [app_cfg.backtest.benchmark_symbol])
    )

    checks: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []

    def _add_check(name: str, severity: str, message: str, **extra: Any) -> None:
        checks.append({"name": name, "severity": severity, "message": message, **extra})
        if severity != "ok":
            issues.append({"check": name, "severity": severity, "message": message})

    db = _open_readonly_db(resolved_data)
    if db is None:
        _add_check("wal_integrity", "critical", f"Missing SQLite database at {resolved_data / 'state.db'}")
        return {
            "generated_at": now.isoformat(),
            "session_date_et": now_et.date().isoformat(),
            "market_open": market_open,
            "status": "CRITICAL",
            "checks": checks,
            "issues": issues,
            "summary": {"parquet_symbols": 0, "open_positions": 0, "unscored_news": 0},
        }

    # 1. WAL integrity
    wal_ok, wal_detail = db.check_wal_integrity()
    _add_check(
        "wal_integrity",
        "ok" if wal_ok else "critical",
        "SQLite WAL quick_check passed" if wal_ok else f"SQLite WAL error: {wal_detail}",
        detail=wal_detail,
    )

    # 2. Daemon heartbeats (market-open vs market-closed aware)
    heartbeats = db.get_daemon_heartbeats()
    for name in DAEMON_NAMES:
        hb = heartbeats.get(name)
        max_age = _expected_heartbeat_max_age(name, market_open=market_open)
        if hb is None:
            _add_check(
                f"daemon:{name}",
                "critical",
                f"No heartbeat found for {name}",
                max_age_seconds=max_age,
            )
            continue
        hb_status = str(hb["status"]).lower()
        hb_dt = ensure_utc(datetime.fromisoformat(hb["last_heartbeat_at"]))
        age_sec = max(0.0, (now - hb_dt).total_seconds())
        if hb_status in {"critical", "error"}:
            _add_check(
                f"daemon:{name}",
                "critical",
                f"{name} reported status={hb_status}",
                age_seconds=round(age_sec, 1),
                daemon_status=hb_status,
            )
        elif age_sec > max_age * 3.0:
            _add_check(
                f"daemon:{name}",
                "critical",
                f"{name} heartbeat critically stale ({age_sec:.0f}s > {max_age * 3.0:.0f}s)",
                age_seconds=round(age_sec, 1),
                daemon_status=hb_status,
            )
        elif age_sec > max_age or hb_status == "degraded":
            _add_check(
                f"daemon:{name}",
                "degraded",
                f"{name} heartbeat degraded/stale (status={hb_status}, age={age_sec:.0f}s)",
                age_seconds=round(age_sec, 1),
                daemon_status=hb_status,
            )
        else:
            _add_check(
                f"daemon:{name}",
                "ok",
                f"{name} healthy (status={hb_status}, age={age_sec:.0f}s)",
                age_seconds=round(age_sec, 1),
                daemon_status=hb_status,
            )

    # 3. Parquet bar coverage (>= 300 bars/symbol + recency)
    parquet_dir = resolved_data / "parquet"
    bars_dir = parquet_dir / "bars_1d"
    min_depth = 0
    missing_syms: list[str] = []
    shallow_syms: list[str] = []
    stale_syms: list[str] = []
    present_syms = 0
    if bars_dir.exists():
        bar_store = ParquetBarStore(parquet_dir)
        depths: list[int] = []
        for sym in expected_symbols:
            df = bar_store.load_bars(sym, timeframe="1d")
            if df.empty:
                missing_syms.append(sym)
                continue
            present_syms += 1
            depths.append(len(df))
            if len(df) < min_bars_required:
                shallow_syms.append(f"{sym}({len(df)})")
            latest_ts = ensure_utc(df["timestamp"].iloc[-1].to_pydatetime())
            if (now - latest_ts).days > 5:
                stale_syms.append(sym)
        min_depth = min(depths) if depths else 0
    else:
        missing_syms = list(expected_symbols)

    if present_syms == 0:
        _add_check("parquet_bars", "critical", "No Parquet 1d bar files present")
    elif missing_syms or shallow_syms or stale_syms:
        parts: list[str] = []
        if missing_syms:
            parts.append(f"missing={len(missing_syms)}")
        if shallow_syms:
            parts.append(f"below_{min_bars_required}={len(shallow_syms)}")
        if stale_syms:
            parts.append(f"stale={len(stale_syms)}")
        _add_check(
            "parquet_bars",
            "degraded",
            f"Parquet bar gaps detected ({', '.join(parts)})",
            min_bars=min_depth,
            present_symbols=present_syms,
        )
    else:
        _add_check(
            "parquet_bars",
            "ok",
            f"All {present_syms} symbols have >= {min_bars_required} bars (min={min_depth})",
            min_bars=min_depth,
            present_symbols=present_syms,
        )

    # 4. FRED macro recency (DGS3MO, VIXCLS)
    latest_macro = db.get_latest_macro_points(["DGS3MO", "VIXCLS"])
    missing_macro = [sid for sid in ("DGS3MO", "VIXCLS") if sid not in latest_macro]
    stale_macro = [
        sid
        for sid, pt in latest_macro.items()
        if (now_et.date() - pt.as_of_date).days > 7
    ]
    if missing_macro or stale_macro:
        _add_check(
            "fred_macro",
            "degraded",
            f"FRED macro incomplete/stale (missing={missing_macro}, stale={stale_macro})",
        )
    else:
        _add_check("fred_macro", "ok", "FRED DGS3MO and VIXCLS up to date")

    # 5. Earnings & SEC filings presence
    counts = db.get_summary_counts()
    if counts["earnings_events"] == 0:
        _add_check(
            "earnings_sec",
            "degraded",
            f"No earnings calendar events stored (sec_filings={counts['sec_filings']})",
        )
    else:
        _add_check(
            "earnings_sec",
            "ok",
            f"earnings_events={counts['earnings_events']}, sec_filings={counts['sec_filings']}",
        )

    # 6. FinBERT / active scorer backlog
    active_model = _active_model(settings)
    unscored = db.count_unscored_news(scorer_model=active_model)
    if unscored > 0:
        _add_check(
            "sentiment_backlog",
            "degraded",
            f"{unscored} unscored news article(s) pending for {active_model}",
            unscored=unscored,
        )
    else:
        _add_check("sentiment_backlog", "ok", f"0 unscored news backlog ({active_model})")

    # 7. Positions vs resting GTC stops
    positions = db.get_live_positions()
    unprotected = [
        p["symbol"]
        for p in positions
        if abs(float(p["qty"])) > 1e-5
        and (not p.get("stop_order_id") or p.get("stop_price") is None)
    ]
    if unprotected:
        _add_check(
            "protective_stops",
            "critical",
            f"Open positions missing resting protective stop: {', '.join(unprotected)}",
            unprotected_symbols=unprotected,
        )
    else:
        _add_check(
            "protective_stops",
            "ok",
            f"All {len(positions)} open position(s) protected by resting stops",
        )

    # 8. Control dials
    dials = db.get_control_dials()
    if dials.pause_new_entries:
        _add_check(
            "control_dials",
            "degraded",
            f"New entries paused (reason={dials.reason or 'unspecified'})",
        )
    elif dials.exposure_multiplier <= 0.0:
        _add_check(
            "control_dials",
            "degraded",
            "Exposure multiplier set to 0.0",
        )
    else:
        _add_check(
            "control_dials",
            "ok",
            f"Dials active (multiplier={dials.exposure_multiplier:.2f}, shorts={dials.allow_shorting})",
        )

    # 9. BLACKOUT_PERIOD insider/restricted-trading rail
    blackout = _resolve_ops_blackout(resolved_config, db, now, settings=settings)
    if blackout.halted:
        _add_check(
            "blackout_period",
            "critical",
            f"BLACKOUT_PERIOD fail-closed HALT: {blackout.halt_reason}",
        )
    elif blackout.has_warnings:
        _add_check(
            "blackout_period",
            "degraded",
            f"BLACKOUT_PERIOD active ({', '.join(sorted(blackout.frozen_symbols)) or 'none'}) with ETF warnings: {'; '.join(blackout.warnings)}",
        )
    elif blackout.frozen_symbols:
        etf_note = f" | ETFs: {'; '.join(blackout.etf_weights_summary)}" if blackout.etf_weights_summary else ""
        _add_check(
            "blackout_period",
            "ok",
            f"BLACKOUT_PERIOD active: frozen={', '.join(sorted(blackout.frozen_symbols))}{etf_note}",
        )
    else:
        _add_check("blackout_period", "ok", "No active BLACKOUT_PERIOD restrictions")

    severities = {c["severity"] for c in checks}
    if "critical" in severities:
        status = "CRITICAL"
    elif "degraded" in severities:
        status = "DEGRADED"
    else:
        status = "READY"

    return {
        "generated_at": now.isoformat(),
        "session_date_et": now_et.date().isoformat(),
        "market_open": market_open,
        "status": status,
        "checks": checks,
        "issues": issues,
        "blackout": blackout.to_status_dict(),
        "summary": {
            "parquet_symbols": present_syms,
            "open_positions": len(positions),
            "unscored_news": unscored,
            **counts,
        },
    }


def build_morning_plan(
    data_dir: Path,
    config_dir: Path,
    *,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build morning pre-market operational & risk plan report."""
    now = ensure_utc(now_utc) if now_utc is not None else datetime.now(UTC)
    now_et = now.astimezone(NY_TZ)
    today_et = now_et.date()

    resolved_data = Path(data_dir)
    resolved_config = Path(config_dir)
    app_cfg = _safe_load_config(resolved_config)
    db = _open_readonly_db(resolved_data)

    dials = db.get_control_dials() if db is not None else ControlDials()
    latest_snap = db.get_latest_portfolio_snapshot() if db is not None else None
    positions = db.get_live_positions() if db is not None else []
    blackout = _resolve_ops_blackout(resolved_config, db, now)

    spy_bars = None
    bars_dir = resolved_data / "parquet" / "bars_1d"
    if bars_dir.exists():
        bar_store = ParquetBarStore(resolved_data / "parquet")
        spy_bars = bar_store.load_bars(app_cfg.backtest.benchmark_symbol, timeframe="1d")

    macro_points: list[MacroPoint] = []
    if db is not None:
        for sid in FRED_SERIES_IDS:
            macro_points.extend(db.get_macro_series(sid))

    macro_scale = compute_macro_regime_scale(
        as_of=now,
        spy_bars=spy_bars,
        macro_points=macro_points,
        spy_sma200_defensive_scale=app_cfg.strategy.spy_sma200_defensive_scale,
        spy_ema50_dual_scale=app_cfg.strategy.spy_ema50_dual_scale,
        spy_vol_surge_scale=app_cfg.strategy.spy_vol_surge_scale,
        spy_vol_surge_thresh=app_cfg.strategy.spy_vol_surge_thresh,
        spy_ret10_crash_thresh=app_cfg.strategy.spy_ret10_crash_thresh,
        spy_ret10_crash_scale=app_cfg.strategy.spy_ret10_crash_scale,
        regime_mode=app_cfg.strategy.regime_mode,
    )

    earnings_blackouts: list[dict[str, str]] = []
    restatement_blackouts: list[dict[str, str]] = []
    overnight_news: list[dict[str, Any]] = []
    overnight_fwd: list[dict[str, Any]] = []
    overnight_sec: list[dict[str, Any]] = []

    if db is not None:
        earn_blackout_days = app_cfg.risk.earnings_blackout_days
        if earn_blackout_days is not None and earn_blackout_days >= 0:
            earnings_map = db.get_all_earnings_map()
            for sym, dates in sorted(earnings_map.items()):
                for edate in dates:
                    if nyse_trading_days_apart(today_et, edate) <= earn_blackout_days:
                        earnings_blackouts.append(
                            {"symbol": sym, "earnings_date": edate.isoformat()}
                        )
                        break

        sec_map = db.get_sec_filings_map()
        for sym, events in sorted(sec_map.items()):
            for ev in events:
                if ev.form_type == "8-K" and "4.02" in ev.items:
                    ev_et_date = ensure_utc(ev.accepted_at).astimezone(NY_TZ).date()
                    if (
                        nyse_trading_days_apart(today_et, ev_et_date)
                        <= app_cfg.risk.restatement_blackout_days
                    ):
                        restatement_blackouts.append(
                            {
                                "symbol": sym,
                                "accepted_at": ev.accepted_at.isoformat(),
                                "accession_number": ev.accession_number,
                            }
                        )
                        break

        since_24h = now - timedelta(hours=24)
        overnight_news = [
            {
                "published_at": a.published_at.isoformat(),
                "source": a.source,
                "symbols": a.symbols,
                "headline": a.headline,
                "sentiment_score": a.sentiment_score,
                "sentiment_label": a.sentiment_label,
            }
            for a in db.get_recent_news(limit=15, forward_only=False, start_utc=since_24h)
        ]
        overnight_fwd = [
            {
                "published_at": a.published_at.isoformat(),
                "source": a.source,
                "symbols": a.symbols,
                "headline": a.headline,
            }
            for a in db.get_recent_news(limit=15, forward_only=True, start_utc=since_24h)
        ]
        overnight_sec = [
            {
                "accepted_at": ev.accepted_at.isoformat(),
                "symbol": ev.symbol,
                "form_type": ev.form_type,
                "items": ev.items,
                "net_insider_buy_usd": ev.net_insider_buy_usd,
            }
            for ev in db.get_recent_sec_filings(limit=15, start_utc=since_24h)
        ]

    equity = float(latest_snap["equity"]) if latest_snap else app_cfg.backtest.initial_capital
    drawdown_pct = float(latest_snap["drawdown_pct"]) * 100.0 if latest_snap else 0.0
    if blackout.halted:
        blackout_summary_str = f"HALTED ({blackout.halt_reason})"
    elif blackout.frozen_symbols:
        etf_note = f" | {'; '.join(blackout.etf_weights_summary)}" if blackout.etf_weights_summary else ""
        blackout_summary_str = f"Frozen={', '.join(sorted(blackout.frozen_symbols))}{etf_note}"
    else:
        blackout_summary_str = "None"
    md_lines = [
        f"# agy-trader Morning Plan ({today_et.isoformat()})",
        "",
        f"- **Equity:** ${equity:,.2f} (Drawdown: {drawdown_pct:.2f}%)",
        f"- **Open Positions:** {len(positions)}",
        f"- **PIT Macro Regime Scale:** {macro_scale:.4f}",
        f"- **Control Dials:** paused={dials.pause_new_entries}, multiplier={dials.exposure_multiplier:.2f}, allow_shorting={dials.allow_shorting}",
        f"- **BLACKOUT_PERIOD Compliance:** {blackout_summary_str}",
        (
            "- **Earnings Blackouts Today:** Disabled"
            if app_cfg.risk.earnings_blackout_days is None
            else f"- **Earnings Blackouts Today:** {', '.join(e['symbol'] for e in earnings_blackouts) or 'None'}"
        ),
        f"- **Item 4.02 Restatement Blackouts:** {', '.join(r['symbol'] for r in restatement_blackouts) or 'None'}",
        f"- **Overnight Highlights (24h):** {len(overnight_news)} model news, {len(overnight_sec)} SEC filings, {len(overnight_fwd)} RSS/YF feeds",
    ]
    if positions:
        md_lines.append("")
        md_lines.append("## Current Positions & GTC Stops")
        for p in positions:
            stop_str = f"${float(p['stop_price']):,.2f}" if p.get("stop_price") is not None else "MISSING"
            md_lines.append(
                f"- **{p['symbol']}**: qty={p['qty']}, mark=${float(p['market_price']):,.2f}, stop={stop_str}"
            )

    return {
        "generated_at": now.isoformat(),
        "session_date_et": today_et.isoformat(),
        "equity": equity,
        "drawdown_pct": round(drawdown_pct, 4),
        "positions": positions,
        "macro_regime_scale": macro_scale,
        "control_dials": dials.model_dump(mode="json"),
        "blackout": blackout.to_status_dict(),
        "earnings_blackout_days": app_cfg.risk.earnings_blackout_days,
        "earnings_blackouts": earnings_blackouts,
        "restatement_blackouts": restatement_blackouts,
        "overnight_model_news": overnight_news,
        "overnight_forward_feeds": overnight_fwd,
        "overnight_sec_filings": overnight_sec,
        "markdown_summary": "\n".join(md_lines),
    }


def _pct_change(cur_val: Any, base_val: Any) -> float | None:
    if cur_val is None or base_val is None:
        return None
    cur = float(cur_val)
    base = float(base_val)
    if not (math.isfinite(cur) and math.isfinite(base)) or base <= 0.0:
        return None
    return round(((cur / base) - 1.0) * 100.0, 4)


def _pick_latest_session_snap(
    rows: Sequence[tuple[Mapping[str, Any], datetime, date]],
) -> tuple[Mapping[str, Any], datetime, date]:
    rth_rows = [row for row in rows if is_nyse_rth_timestamp(row[1])]
    if rth_rows:
        return rth_rows[-1]
    trading_rows = [row for row in rows if is_nyse_trading_day(row[2])]
    return trading_rows[-1] if trading_rows else rows[-1]


def select_benchmark_session_snapshots(
    snaps: Sequence[Mapping[str, Any]],
    as_of_date_et: date,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    """Select current and prior trading-session benchmark mark snapshots.

    Prefers NYSE regular-trading-hour marks (09:30 ET through close + 10m grace)
    to avoid extended-hours/weekend/holiday wide stub quotes, while falling back
    cleanly for synthetic tests with arbitrary timestamps.
    """
    parsed: list[tuple[Mapping[str, Any], datetime, date]] = []
    for snap in snaps:
        if not any(
            snap.get(k) is not None
            for k in ("spy_close", "ew50_index", "spmo_close", "qmom_close")
        ):
            continue
        raw_ts = snap.get("timestamp")
        if raw_ts is None:
            continue
        try:
            ts_utc = (
                ensure_utc(raw_ts)
                if isinstance(raw_ts, datetime)
                else ensure_utc(datetime.fromisoformat(str(raw_ts)))
            )
        except (TypeError, ValueError):
            continue
        ts_et = ts_utc.astimezone(NY_TZ)
        parsed.append((snap, ts_et, ts_et.date()))

    if not parsed:
        return (None, None)

    eligible = [row for row in parsed if row[2] <= as_of_date_et] or parsed
    cur_bench_snap, _, cur_date = _pick_latest_session_snap(eligible)

    prior = [row for row in eligible if row[2] < cur_date]
    if not prior:
        return (cur_bench_snap, None)

    prev_bench_snap, _, _ = _pick_latest_session_snap(prior)
    return (cur_bench_snap, prev_bench_snap)


def resolve_snapshot_benchmark_mark(
    snap: Mapping[str, Any] | None,
    field: str,
    symbol: str | None = None,
    bar_store: ParquetBarStore | None = None,
    *,
    bar_cache: dict[str, tuple[list[date], list[float]]] | None = None,
) -> float | None:
    """Resolve a benchmark mark from a portfolio snapshot, falling back to 1d Parquet bars."""
    if snap is None:
        return None
    raw = snap.get(field)
    if raw is not None:
        val = float(raw)
        if math.isfinite(val) and val > 0.0:
            return val
    if symbol is not None and bar_store is not None:
        raw_ts = snap.get("timestamp")
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
        d_et = ts_utc.astimezone(NY_TZ).date()
        if bar_cache is not None and symbol in bar_cache:
            dates, closes = bar_cache[symbol]
        else:
            df = bar_store.load_bars(symbol, timeframe="1d")
            if df.empty or "timestamp" not in df.columns or "close" not in df.columns:
                dates, closes = [], []
            else:
                ts_col = df["timestamp"]
                ts_et_series = (
                    ts_col.dt.tz_localize(UTC).dt.tz_convert(NY_TZ).dt.date
                    if ts_col.dt.tz is None
                    else ts_col.dt.tz_convert(NY_TZ).dt.date
                )
                dates = list(ts_et_series)
                closes = [float(c) for c in df["close"].tolist()]
            if bar_cache is not None:
                bar_cache[symbol] = (dates, closes)
        if dates:
            idx = bisect.bisect_right(dates, d_et) - 1
            if idx >= 0:
                val = closes[idx]
                if math.isfinite(val) and val > 0.0:
                    return val
    return None


def build_daily_summary(
    data_dir: Path,
    config_dir: Path,
    *,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build post-close daily performance, fills/slippage, and risk summary."""
    now = ensure_utc(now_utc) if now_utc is not None else datetime.now(UTC)
    now_et = now.astimezone(NY_TZ)
    today_et = now_et.date()
    today_str = today_et.isoformat()

    resolved_data = Path(data_dir)
    resolved_config = Path(config_dir)
    app_cfg = _safe_load_config(resolved_config)
    asset_types = {u.symbol: u.asset_type for u in _safe_load_universe(resolved_config)}
    db = _open_readonly_db(resolved_data)
    blackout = _resolve_ops_blackout(resolved_config, db, now)

    first_snap = db.get_first_portfolio_snapshot() if db is not None else None
    first_invested_snap = db.get_first_invested_portfolio_snapshot() if db is not None else None
    latest_snap = db.get_latest_portfolio_snapshot() if db is not None else None
    recent_snaps = db.get_portfolio_snapshots(limit=2000) if db is not None else []

    equity = float(latest_snap["equity"]) if latest_snap else app_cfg.backtest.initial_capital
    daily_return_pct = float(latest_snap["daily_pnl_pct"]) * 100.0 if latest_snap else 0.0
    drawdown_pct = float(latest_snap["drawdown_pct"]) * 100.0 if latest_snap else 0.0
    halt_pct = -app_cfg.risk.max_drawdown_halt_pct * 100.0
    drawdown_headroom_pct = round(drawdown_pct - halt_pct, 4)

    cum_return_pct = (
        ((equity / float(first_snap["equity"])) - 1.0) * 100.0
        if first_snap and float(first_snap["equity"]) > 0.0
        else 0.0
    )

    cur_bench_snap, prev_bench_snap = select_benchmark_session_snapshots(recent_snaps, today_et)
    if cur_bench_snap is None:
        cur_bench_snap = latest_snap

    bars_1d_dir = resolved_data / "parquet" / "bars_1d"
    bar_store = ParquetBarStore(resolved_data / "parquet") if bars_1d_dir.exists() else None
    bar_cache: dict[str, tuple[list[date], list[float]]] = {}

    cur_spy = resolve_snapshot_benchmark_mark(
        cur_bench_snap, "spy_close", "SPY", bar_store, bar_cache=bar_cache
    )
    prev_spy = resolve_snapshot_benchmark_mark(
        prev_bench_snap, "spy_close", "SPY", bar_store, bar_cache=bar_cache
    )
    cur_ew = resolve_snapshot_benchmark_mark(cur_bench_snap, "ew50_index")
    prev_ew = resolve_snapshot_benchmark_mark(prev_bench_snap, "ew50_index")
    cur_spmo = resolve_snapshot_benchmark_mark(
        cur_bench_snap, "spmo_close", "SPMO", bar_store, bar_cache=bar_cache
    )
    prev_spmo = resolve_snapshot_benchmark_mark(
        prev_bench_snap, "spmo_close", "SPMO", bar_store, bar_cache=bar_cache
    )
    cur_qmom = resolve_snapshot_benchmark_mark(
        cur_bench_snap, "qmom_close", "QMOM", bar_store, bar_cache=bar_cache
    )
    prev_qmom = resolve_snapshot_benchmark_mark(
        prev_bench_snap, "qmom_close", "QMOM", bar_store, bar_cache=bar_cache
    )

    spy_daily_pct = _pct_change(cur_spy, prev_spy)
    ew_daily_pct = _pct_change(cur_ew, prev_ew)
    spmo_daily_pct = _pct_change(cur_spmo, prev_spmo)
    qmom_daily_pct = _pct_change(cur_qmom, prev_qmom)

    if first_invested_snap is not None and cur_bench_snap is not None:
        first_spy = resolve_snapshot_benchmark_mark(
            first_invested_snap, "spy_close", "SPY", bar_store, bar_cache=bar_cache
        )
        first_ew = resolve_snapshot_benchmark_mark(first_invested_snap, "ew50_index")
        first_spmo = resolve_snapshot_benchmark_mark(
            first_invested_snap, "spmo_close", "SPMO", bar_store, bar_cache=bar_cache
        )
        first_qmom = resolve_snapshot_benchmark_mark(
            first_invested_snap, "qmom_close", "QMOM", bar_store, bar_cache=bar_cache
        )
        spy_cum_pct = _pct_change(cur_spy, first_spy)
        ew_cum_pct = _pct_change(cur_ew, first_ew)
        spmo_cum_pct = _pct_change(cur_spmo, first_spmo)
        qmom_cum_pct = _pct_change(cur_qmom, first_qmom)
    else:
        spy_cum_pct = None
        ew_cum_pct = None
        spmo_cum_pct = None
        qmom_cum_pct = None

    # Today's fills & slippage vs backtest budget
    todays_fills: list[dict[str, Any]] = []
    slippages: list[float] = []
    budgets: list[float] = []
    if db is not None:
        for ord_row in db.get_recent_live_orders(limit=200):
            sub_dt = ensure_utc(datetime.fromisoformat(ord_row["submitted_at"]))
            if sub_dt.astimezone(NY_TZ).date() != today_et:
                continue
            if ord_row["order_type"] == "stop":
                continue
            sym = str(ord_row["symbol"]).upper()
            atype = asset_types.get(sym, AssetType.equity)
            budget = (
                app_cfg.backtest.etf_slippage_bps
                if atype == AssetType.etf
                else app_cfg.backtest.equity_slippage_bps
            )
            slip = ord_row["realized_slippage_bps"]
            if slip is not None:
                slippages.append(float(slip))
                budgets.append(budget)
            todays_fills.append({**ord_row, "backtest_budget_bps": budget})

    mean_slip_bps = round(sum(slippages) / len(slippages), 2) if slippages else None
    mean_budget_bps = round(sum(budgets) / len(budgets), 2) if budgets else None

    positions = db.get_live_positions() if db is not None else []
    enriched_positions: list[dict[str, Any]] = []
    for p in positions:
        qty = float(p["qty"])
        mkt = float(p["market_price"])
        stp = float(p["stop_price"]) if p.get("stop_price") is not None else None
        dist_pct: float | None = None
        if mkt > 0.0 and stp is not None and stp > 0.0:
            dist_pct = (
                round(((mkt - stp) / mkt) * 100.0, 2)
                if qty >= 0.0
                else round(((stp - mkt) / mkt) * 100.0, 2)
            )
        enriched_positions.append({**p, "distance_to_stop_pct": dist_pct})

    since_24h = now - timedelta(hours=24)
    notable_news = (
        [
            {"source": a.source, "symbols": a.symbols, "headline": a.headline, "score": a.sentiment_score}
            for a in db.get_recent_news(limit=10, forward_only=False, start_utc=since_24h)
        ]
        if db is not None
        else []
    )
    notable_fwd = (
        [
            {"source": a.source, "symbols": a.symbols, "headline": a.headline}
            for a in db.get_recent_news(limit=10, forward_only=True, start_utc=since_24h)
        ]
        if db is not None
        else []
    )
    notable_sec = (
        [
            {"symbol": ev.symbol, "form_type": ev.form_type, "items": ev.items, "net_usd": ev.net_insider_buy_usd}
            for ev in db.get_recent_sec_filings(limit=10, start_utc=since_24h)
        ]
        if db is not None
        else []
    )

    slip_text = (
        f"{mean_slip_bps:.2f} bps (budget {mean_budget_bps:.2f} bps)"
        if mean_slip_bps is not None and mean_budget_bps is not None
        else "No fills today"
    )
    spy_daily_str = f"{spy_daily_pct:+.2f}%" if spy_daily_pct is not None else "N/A"
    ew_daily_str = f"{ew_daily_pct:+.2f}%" if ew_daily_pct is not None else "N/A"
    spmo_daily_str = f"{spmo_daily_pct:+.2f}%" if spmo_daily_pct is not None else "N/A"
    qmom_daily_str = f"{qmom_daily_pct:+.2f}%" if qmom_daily_pct is not None else "N/A"
    spy_cum_str = f"{spy_cum_pct:+.2f}%" if spy_cum_pct is not None else "N/A"
    ew_cum_str = f"{ew_cum_pct:+.2f}%" if ew_cum_pct is not None else "N/A"
    spmo_cum_str = f"{spmo_cum_pct:+.2f}%" if spmo_cum_pct is not None else "N/A"
    qmom_cum_str = f"{qmom_cum_pct:+.2f}%" if qmom_cum_pct is not None else "N/A"
    if blackout.halted:
        blackout_summary_str = f"HALTED ({blackout.halt_reason})"
    elif blackout.frozen_symbols:
        etf_note = f" | {'; '.join(blackout.etf_weights_summary)}" if blackout.etf_weights_summary else ""
        blackout_summary_str = f"Frozen={', '.join(sorted(blackout.frozen_symbols))}{etf_note}"
    else:
        blackout_summary_str = "None"
    md_lines = [
        f"# agy-trader Daily Close Summary ({today_str})",
        "",
        f"- **Equity:** ${equity:,.2f}",
        f"- **Daily Return:** {daily_return_pct:+.2f}% (SPY: {spy_daily_str}, EW-50: {ew_daily_str}, SPMO: {spmo_daily_str}, QMOM: {qmom_daily_str})",
        f"- **Cumulative Return:** {cum_return_pct:+.2f}% (since first trade — SPY: {spy_cum_str}, EW-50: {ew_cum_str}, SPMO: {spmo_cum_str}, QMOM: {qmom_cum_str})",
        f"- **Drawdown:** {drawdown_pct:.2f}% (Headroom to {halt_pct:.1f}% halt: {drawdown_headroom_pct:.2f}%)",
        f"- **15:40 ET Fills:** {len(todays_fills)} order(s) | Mean Slippage: {slip_text}",
        f"- **Open Positions:** {len(enriched_positions)}",
        f"- **BLACKOUT_PERIOD Compliance:** {blackout_summary_str}",
    ]

    return {
        "generated_at": now.isoformat(),
        "session_date_et": today_str,
        "equity": equity,
        "daily_return_pct": round(daily_return_pct, 4),
        "spy_daily_return_pct": spy_daily_pct,
        "ew50_daily_return_pct": ew_daily_pct,
        "spmo_daily_return_pct": spmo_daily_pct,
        "qmom_daily_return_pct": qmom_daily_pct,
        "cumulative_return_pct": round(cum_return_pct, 4),
        "spy_cumulative_return_pct": spy_cum_pct,
        "ew50_cumulative_return_pct": ew_cum_pct,
        "spmo_cumulative_return_pct": spmo_cum_pct,
        "qmom_cumulative_return_pct": qmom_cum_pct,
        "benchmark_anchor_timestamp": first_invested_snap["timestamp"] if first_invested_snap is not None else None,
        "drawdown_pct": round(drawdown_pct, 4),
        "max_drawdown_halt_pct": halt_pct,
        "drawdown_headroom_pct": drawdown_headroom_pct,
        "blackout": blackout.to_status_dict(),
        "todays_fills": todays_fills,
        "mean_realized_slippage_bps": mean_slip_bps,
        "mean_backtest_budget_bps": mean_budget_bps,
        "positions": enriched_positions,
        "notable_news": notable_news,
        "notable_forward_feeds": notable_fwd,
        "notable_sec_filings": notable_sec,
        "markdown_summary": "\n".join(md_lines),
    }
