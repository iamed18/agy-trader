from __future__ import annotations

import ipaddress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from agy_ops.daemons.common import NY_TZ
from agy_ops.reporting.digests import (
    _pct_change,
    resolve_snapshot_benchmark_mark,
    select_benchmark_session_snapshots,
)
from agy_trader.core.blackout import load_universe_for_blackout, resolve_live_blackout
from agy_trader.core.config import AppConfig, EnvSettings, load_app_config, load_universe
from agy_trader.core.models import AssetType, ControlDials, ensure_utc
from agy_trader.storage.db import StateDB
from agy_trader.storage.parquet_store import ParquetBarStore

_LAN_V4_NETWORKS: tuple[ipaddress.IPv4Network, ...] = (
    ipaddress.IPv4Network("127.0.0.0/8"),     # Loopback
    ipaddress.IPv4Network("10.0.0.0/8"),      # RFC 1918 private LAN
    ipaddress.IPv4Network("172.16.0.0/12"),   # RFC 1918 private LAN
    ipaddress.IPv4Network("192.168.0.0/16"),  # RFC 1918 private LAN
    ipaddress.IPv4Network("169.254.0.0/16"),  # Link-local
    ipaddress.IPv4Network("100.64.0.0/10"),   # RFC 6598 CGNAT / Tailscale
)
_LAN_V6_NETWORKS: tuple[ipaddress.IPv6Network, ...] = (
    ipaddress.IPv6Network("::1/128"),         # Loopback
    ipaddress.IPv6Network("fc00::/7"),        # Unique Local Address (ULA)
    ipaddress.IPv6Network("fe80::/10"),       # Link-local
)
ALLOWED_DASHBOARD_HOSTS: frozenset[str] = frozenset(
    {"127.0.0.1", "localhost", "::1", "0.0.0.0", "::"}
)
SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    ),
}
DAEMON_NAMES: tuple[str, ...] = ("market-data", "news-nlp", "trader", "dashboard")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def is_lan_client_host(host: str | None) -> bool:
    """Return True iff host is a loopback, RFC 1918/ULA, link-local, or RFC 6598 IP."""
    if not host:
        return False
    cleaned = host.strip()
    if "%" in cleaned:
        cleaned = cleaned.split("%", 1)[0]
    try:
        addr = ipaddress.ip_address(cleaned)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    if isinstance(addr, ipaddress.IPv4Address):
        return any(addr in net for net in _LAN_V4_NETWORKS)
    return any(addr in net for net in _LAN_V6_NETWORKS)


def validate_dashboard_host(host: str) -> str:
    cleaned = host.strip()
    if cleaned in ALLOWED_DASHBOARD_HOSTS:
        return cleaned
    if "%" not in cleaned and is_lan_client_host(cleaned):
        return cleaned
    raise ValueError(
        f"Dashboard host {host!r} rejected: must be localhost, a private/LAN IP, "
        f"or wildcard bind ({sorted(ALLOWED_DASHBOARD_HOSTS)})"
    )


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


def _safe_load_asset_types(config_dir: Path) -> dict[str, AssetType]:
    try:
        return {u.symbol: u.asset_type for u in load_universe(config_dir)}
    except (FileNotFoundError, OSError, ValueError):
        return {}


def _resolve_dashboard_blackout(config_dir: Path, db: StateDB | None) -> dict[str, Any]:
    settings = EnvSettings()
    universe, universe_error = load_universe_for_blackout(config_dir)
    env_path = Path(".env")
    obs = [p["symbol"] for p in db.get_live_positions()] if db is not None else ()
    return resolve_live_blackout(
        env_file=env_path if env_path.exists() else None,
        require_env_file=env_path.exists(),
        fallback_raw=settings.blackout_period,
        db=db,
        universe=universe,
        universe_error=universe_error,
        observed_broker_symbols=obs,
    ).to_status_dict()


def _count_parquet_symbols(data_dir: Path) -> int:
    bars_dir = Path(data_dir) / "parquet" / "bars_1d"
    if not bars_dir.exists():
        return 0
    return sum(1 for p in bars_dir.glob("*.parquet") if p.is_file())


def _parse_iso_utc(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return ensure_utc(datetime.fromisoformat(raw))
    except ValueError:
        return None


def create_dashboard_app(
    data_dir: Path,
    config_dir: Path,
    *,
    lan_only: bool = True,
) -> FastAPI:
    """Create a strictly read-only FastAPI dashboard application."""
    resolved_data = Path(data_dir)
    resolved_config = Path(config_dir)

    app = FastAPI(
        title="agy-trader Ops Dashboard",
        description="Strictly read-only live portfolio and daemon observability UI",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next: Any) -> Response:
        if lan_only:
            client_host = request.client.host if request.client is not None else None
            if not is_lan_client_host(client_host):
                return PlainTextResponse(
                    "Forbidden: dashboard access is restricted to localhost and private LAN clients",
                    status_code=403,
                    headers=SECURITY_HEADERS,
                )
        response: Response = await call_next(request)
        for header_name, header_val in SECURITY_HEADERS.items():
            response.headers[header_name] = header_val
        return response

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/overview")
    def api_overview() -> dict[str, Any]:
        app_cfg = _safe_load_config(resolved_config)
        db = _open_readonly_db(resolved_data)
        blackout = _resolve_dashboard_blackout(resolved_config, db)
        if db is None:
            dials = ControlDials()
            return {
                "has_data": False,
                "timestamp": None,
                "equity": app_cfg.backtest.initial_capital,
                "cash": app_cfg.backtest.initial_capital,
                "buying_power": app_cfg.backtest.initial_capital,
                "daily_pnl_pct": 0.0,
                "total_return_pct": 0.0,
                "drawdown_pct": 0.0,
                "high_water_mark": app_cfg.backtest.initial_capital,
                "gross_exposure": 0.0,
                "net_exposure": 0.0,
                "spy_return_pct": None,
                "ew50_return_pct": None,
                "spmo_return_pct": None,
                "qmom_return_pct": None,
                "spy_daily_return_pct": None,
                "ew50_daily_return_pct": None,
                "spmo_daily_return_pct": None,
                "qmom_daily_return_pct": None,
                "control_dials": dials.model_dump(mode="json"),
                "blackout": blackout,
            }

        first = db.get_first_portfolio_snapshot()
        latest = db.get_latest_portfolio_snapshot()
        dials = db.get_control_dials()
        if first is None or latest is None:
            return {
                "has_data": False,
                "timestamp": None,
                "equity": app_cfg.backtest.initial_capital,
                "cash": app_cfg.backtest.initial_capital,
                "buying_power": app_cfg.backtest.initial_capital,
                "daily_pnl_pct": 0.0,
                "total_return_pct": 0.0,
                "drawdown_pct": 0.0,
                "high_water_mark": app_cfg.backtest.initial_capital,
                "gross_exposure": 0.0,
                "net_exposure": 0.0,
                "spy_return_pct": None,
                "ew50_return_pct": None,
                "spmo_return_pct": None,
                "qmom_return_pct": None,
                "spy_daily_return_pct": None,
                "ew50_daily_return_pct": None,
                "spmo_daily_return_pct": None,
                "qmom_daily_return_pct": None,
                "control_dials": dials.model_dump(mode="json"),
                "blackout": blackout,
            }

        base_eq = float(first["equity"])
        cur_eq = float(latest["equity"])
        total_ret_pct = ((cur_eq / base_eq) - 1.0) * 100.0 if base_eq > 0.0 else 0.0

        invested_first = db.get_first_invested_portfolio_snapshot()
        recent_snaps = db.get_portfolio_snapshots(limit=2000)
        latest_et_date = ensure_utc(datetime.fromisoformat(str(latest["timestamp"]))).astimezone(NY_TZ).date()
        cur_bench, prev_bench = select_benchmark_session_snapshots(recent_snaps, latest_et_date)
        if cur_bench is None:
            cur_bench = latest

        bars_1d_dir = resolved_data / "parquet" / "bars_1d"
        bar_store = ParquetBarStore(resolved_data / "parquet") if bars_1d_dir.exists() else None
        bar_cache: dict[str, Any] = {}

        cur_spy = resolve_snapshot_benchmark_mark(
            cur_bench, "spy_close", "SPY", bar_store, bar_cache=bar_cache
        )
        prev_spy = resolve_snapshot_benchmark_mark(
            prev_bench, "spy_close", "SPY", bar_store, bar_cache=bar_cache
        )
        cur_ew = resolve_snapshot_benchmark_mark(cur_bench, "ew50_index")
        prev_ew = resolve_snapshot_benchmark_mark(prev_bench, "ew50_index")
        cur_spmo = resolve_snapshot_benchmark_mark(
            cur_bench, "spmo_close", "SPMO", bar_store, bar_cache=bar_cache
        )
        prev_spmo = resolve_snapshot_benchmark_mark(
            prev_bench, "spmo_close", "SPMO", bar_store, bar_cache=bar_cache
        )
        cur_qmom = resolve_snapshot_benchmark_mark(
            cur_bench, "qmom_close", "QMOM", bar_store, bar_cache=bar_cache
        )
        prev_qmom = resolve_snapshot_benchmark_mark(
            prev_bench, "qmom_close", "QMOM", bar_store, bar_cache=bar_cache
        )

        if invested_first is not None:
            first_spy = resolve_snapshot_benchmark_mark(
                invested_first, "spy_close", "SPY", bar_store, bar_cache=bar_cache
            )
            first_ew = resolve_snapshot_benchmark_mark(invested_first, "ew50_index")
            first_spmo = resolve_snapshot_benchmark_mark(
                invested_first, "spmo_close", "SPMO", bar_store, bar_cache=bar_cache
            )
            first_qmom = resolve_snapshot_benchmark_mark(
                invested_first, "qmom_close", "QMOM", bar_store, bar_cache=bar_cache
            )
            spy_ret_pct = _pct_change(cur_spy, first_spy)
            ew_ret_pct = _pct_change(cur_ew, first_ew)
            spmo_ret_pct = _pct_change(cur_spmo, first_spmo)
            qmom_ret_pct = _pct_change(cur_qmom, first_qmom)
        else:
            spy_ret_pct = None
            ew_ret_pct = None
            spmo_ret_pct = None
            qmom_ret_pct = None

        spy_daily_pct = _pct_change(cur_spy, prev_spy)
        ew_daily_pct = _pct_change(cur_ew, prev_ew)
        spmo_daily_pct = _pct_change(cur_spmo, prev_spmo)
        qmom_daily_pct = _pct_change(cur_qmom, prev_qmom)

        return {
            "has_data": True,
            "timestamp": latest["timestamp"],
            "equity": cur_eq,
            "cash": float(latest["cash"]),
            "buying_power": float(latest["buying_power"]),
            "daily_pnl_pct": float(latest["daily_pnl_pct"]) * 100.0,
            "total_return_pct": round(total_ret_pct, 4),
            "drawdown_pct": float(latest["drawdown_pct"]) * 100.0,
            "high_water_mark": float(latest["high_water_mark"]),
            "gross_exposure": float(latest["gross_exposure"]),
            "net_exposure": float(latest["net_exposure"]),
            "spy_return_pct": spy_ret_pct,
            "ew50_return_pct": ew_ret_pct,
            "spmo_return_pct": spmo_ret_pct,
            "qmom_return_pct": qmom_ret_pct,
            "spy_daily_return_pct": spy_daily_pct,
            "ew50_daily_return_pct": ew_daily_pct,
            "spmo_daily_return_pct": spmo_daily_pct,
            "qmom_daily_return_pct": qmom_daily_pct,
            "control_dials": dials.model_dump(mode="json"),
            "blackout": blackout,
        }

    @app.get("/api/equity-curve")
    def api_equity_curve(
        limit: int = Query(default=1000, ge=1, le=5000),
    ) -> dict[str, Any]:
        db = _open_readonly_db(resolved_data)
        if db is None:
            return {"count": 0, "series": []}
        snaps = db.get_portfolio_snapshots(limit=limit)
        if not snaps:
            return {"count": 0, "series": []}

        bars_1d_dir = resolved_data / "parquet" / "bars_1d"
        bar_store = ParquetBarStore(resolved_data / "parquet") if bars_1d_dir.exists() else None
        bar_cache: dict[str, Any] = {}

        resolved_rows: list[tuple[dict[str, Any], float | None, float | None, float | None, float | None]] = []
        for s in snaps:
            spy = float(s["spy_close"]) if s["spy_close"] is not None else None
            ew = float(s["ew50_index"]) if s["ew50_index"] is not None else None
            spmo = resolve_snapshot_benchmark_mark(
                s, "spmo_close", "SPMO", bar_store, bar_cache=bar_cache
            )
            qmom = resolve_snapshot_benchmark_mark(
                s, "qmom_close", "QMOM", bar_store, bar_cache=bar_cache
            )
            resolved_rows.append((s, spy, ew, spmo, qmom))

        base_eq = next((float(s["equity"]) for s in snaps if float(s["equity"]) > 0.0), None)
        base_spy = next(
            (spy for s, spy, _, _, _ in resolved_rows if float(s["gross_exposure"]) > 0.0 and spy is not None and spy > 0.0),
            None,
        ) or next((spy for _, spy, _, _, _ in resolved_rows if spy is not None and spy > 0.0), None)
        base_ew50 = next(
            (ew for s, _, ew, _, _ in resolved_rows if float(s["gross_exposure"]) > 0.0 and ew is not None and ew > 0.0),
            None,
        ) or next((ew for _, _, ew, _, _ in resolved_rows if ew is not None and ew > 0.0), None)
        base_spmo = next(
            (spmo for s, _, _, spmo, _ in resolved_rows if float(s["gross_exposure"]) > 0.0 and spmo is not None and spmo > 0.0),
            None,
        ) or next((spmo for _, _, _, spmo, _ in resolved_rows if spmo is not None and spmo > 0.0), None)
        base_qmom = next(
            (qmom for s, _, _, _, qmom in resolved_rows if float(s["gross_exposure"]) > 0.0 and qmom is not None and qmom > 0.0),
            None,
        ) or next((qmom for _, _, _, _, qmom in resolved_rows if qmom is not None and qmom > 0.0), None)

        series: list[dict[str, Any]] = []
        for s, spy, ew, spmo, qmom in resolved_rows:
            eq = float(s["equity"])
            series.append(
                {
                    "timestamp": s["timestamp"],
                    "equity": eq,
                    "portfolio_index": round((eq / base_eq) * 100.0, 4) if base_eq else 100.0,
                    "spy_index": (
                        round((spy / base_spy) * 100.0, 4)
                        if (spy is not None and base_spy)
                        else None
                    ),
                    "ew50_index": (
                        round((ew / base_ew50) * 100.0, 4)
                        if (ew is not None and base_ew50)
                        else None
                    ),
                    "spmo_index": (
                        round((spmo / base_spmo) * 100.0, 4)
                        if (spmo is not None and base_spmo)
                        else None
                    ),
                    "qmom_index": (
                        round((qmom / base_qmom) * 100.0, 4)
                        if (qmom is not None and base_qmom)
                        else None
                    ),
                    "drawdown_pct": round(float(s["drawdown_pct"]) * 100.0, 4),
                    "daily_pnl_pct": round(float(s["daily_pnl_pct"]) * 100.0, 4),
                }
            )
        return {"count": len(series), "series": series}

    @app.get("/api/positions")
    def api_positions() -> dict[str, Any]:
        db = _open_readonly_db(resolved_data)
        if db is None:
            return {"count": 0, "positions": []}
        raw_positions = db.get_live_positions()
        items: list[dict[str, Any]] = []
        for pos in raw_positions:
            qty = float(pos["qty"])
            mkt_px = float(pos["market_price"])
            stop_px = float(pos["stop_price"]) if pos["stop_price"] is not None else None
            dist_pct: float | None = None
            if mkt_px > 0.0 and stop_px is not None and stop_px > 0.0:
                if qty >= 0.0:
                    dist_pct = round(((mkt_px - stop_px) / mkt_px) * 100.0, 2)
                else:
                    dist_pct = round(((stop_px - mkt_px) / mkt_px) * 100.0, 2)
            items.append(
                {
                    "symbol": pos["symbol"],
                    "side": "long" if qty >= 0.0 else "short",
                    "qty": qty,
                    "avg_entry_price": float(pos["avg_entry_price"]),
                    "market_price": mkt_px,
                    "market_value": round(qty * mkt_px, 2),
                    "unrealized_pnl": float(pos["unrealized_pnl"]),
                    "stop_order_id": pos["stop_order_id"],
                    "stop_price": stop_px,
                    "distance_to_stop_pct": dist_pct,
                    "stop_synced": bool(pos["stop_order_id"] and stop_px is not None),
                    "updated_at": pos["updated_at"],
                }
            )
        return {"count": len(items), "positions": items}

    @app.get("/api/orders")
    def api_orders(
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        db = _open_readonly_db(resolved_data)
        if db is None:
            return {"count": 0, "orders": []}
        app_cfg = _safe_load_config(resolved_config)
        asset_types = _safe_load_asset_types(resolved_config)
        orders = db.get_recent_live_orders(limit=limit)
        items: list[dict[str, Any]] = []
        for ord_row in orders:
            sym = str(ord_row["symbol"]).upper()
            atype = asset_types.get(sym, AssetType.equity)
            budget_bps = (
                app_cfg.backtest.etf_slippage_bps
                if atype == AssetType.etf
                else app_cfg.backtest.equity_slippage_bps
            )
            slip = ord_row["realized_slippage_bps"]
            vs_budget = round(float(slip) - budget_bps, 2) if slip is not None else None
            items.append(
                {
                    **ord_row,
                    "asset_type": atype.value,
                    "backtest_budget_bps": budget_bps,
                    "slippage_vs_budget_bps": vs_budget,
                }
            )
        return {"count": len(items), "orders": items}

    @app.get("/api/health")
    def api_health() -> dict[str, Any]:
        now = datetime.now(UTC)
        parquet_count = _count_parquet_symbols(resolved_data)
        db = _open_readonly_db(resolved_data)
        if db is None:
            return {
                "wal_ok": False,
                "wal_detail": "state.db not found",
                "daemons": {
                    name: {
                        "daemon_name": name,
                        "status": "missing",
                        "last_heartbeat_at": None,
                        "staleness_seconds": None,
                        "rss_mb": None,
                        "peak_rss_mb": None,
                        "details": {},
                    }
                    for name in DAEMON_NAMES
                },
                "last_boot": None,
                "last_reconcile": None,
                "last_rebalance": None,
                "recent_events": [],
                "data_counts": {
                    "parquet_symbols": parquet_count,
                    "news_articles": 0,
                    "scored_news": 0,
                    "unscored_news": 0,
                    "earnings_events": 0,
                    "sec_filings": 0,
                    "macro_points": 0,
                    "backtest_runs": 0,
                },
            }

        wal_ok, wal_detail = db.check_wal_integrity()
        raw_hbs = db.get_daemon_heartbeats()
        daemons: dict[str, dict[str, Any]] = {}
        for name in DAEMON_NAMES:
            hb = raw_hbs.get(name)
            if hb is None:
                daemons[name] = {
                    "daemon_name": name,
                    "status": "missing",
                    "last_heartbeat_at": None,
                    "staleness_seconds": None,
                    "rss_mb": None,
                    "peak_rss_mb": None,
                    "details": {},
                }
                continue
            hb_dt = _parse_iso_utc(hb.get("last_heartbeat_at"))
            age_sec = round(max(0.0, (now - hb_dt).total_seconds()), 1) if hb_dt else None
            details = hb.get("details") or {}
            daemons[name] = {
                "daemon_name": name,
                "status": hb["status"],
                "last_heartbeat_at": hb["last_heartbeat_at"],
                "staleness_seconds": age_sec,
                "rss_mb": details.get("rss_mb"),
                "peak_rss_mb": details.get("peak_rss_mb", details.get("rss_mb")),
                "details": details,
            }

        boot_ev = db.get_daemon_events(event_type="boot", limit=1)
        recon_ev = db.get_daemon_events(event_type="reconcile", limit=1)
        rebal_ev = db.get_daemon_events(event_type="rebalance", limit=1)
        recent_events = db.get_daemon_events(limit=20)
        summary_counts = db.get_summary_counts()
        settings = EnvSettings()
        active_model = "ProsusAI/finbert" if settings.nlp_backend == "finbert" else "lexicon-v1"
        unscored = db.count_unscored_news(scorer_model=active_model)

        return {
            "wal_ok": wal_ok,
            "wal_detail": wal_detail,
            "daemons": daemons,
            "last_boot": boot_ev[0] if boot_ev else None,
            "last_reconcile": recon_ev[0] if recon_ev else None,
            "last_rebalance": rebal_ev[0] if rebal_ev else None,
            "recent_events": recent_events,
            "data_counts": {
                "parquet_symbols": parquet_count,
                "unscored_news": unscored,
                **summary_counts,
            },
        }

    @app.get("/api/intelligence")
    def api_intelligence(
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        db = _open_readonly_db(resolved_data)
        if db is None:
            return {
                "model_news": [],
                "forward_only_news": [],
                "sec_filings": [],
                "audit_log": [],
            }
        model_news = db.get_recent_news(limit=limit, forward_only=False)
        fwd_news = db.get_recent_news(limit=limit, forward_only=True)
        sec_filings = db.get_recent_sec_filings(limit=limit)
        audit_log = db.get_control_audit_log(limit=limit)

        return {
            "model_news": [art.model_dump(mode="json") for art in model_news],
            "forward_only_news": [art.model_dump(mode="json") for art in fwd_news],
            "sec_filings": [ev.model_dump(mode="json") for ev in sec_filings],
            "audit_log": audit_log,
        }

    return app


def serve_dashboard(
    data_dir: Path,
    config_dir: Path,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Serve the strictly read-only dashboard on localhost or private LAN."""
    safe_host = validate_dashboard_host(host)
    app = create_dashboard_app(data_dir=data_dir, config_dir=config_dir, lan_only=True)
    uvicorn.run(
        app,
        host=safe_host,
        port=int(port),
        log_level="info",
        access_log=False,
        proxy_headers=False,
    )
