from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from agy_ops.cli import app as ops_cli_app
from agy_ops.dashboard import create_dashboard_app, serve_dashboard
from agy_ops.reporting import (
    build_daily_summary,
    build_morning_plan,
    build_readiness_report,
)
from agy_trader.core.config import load_app_config, load_universe
from agy_trader.core.control_plane import ControlPlane
from agy_trader.core.models import EarningsEvent, MacroPoint, NewsArticle, SecFilingEvent
from agy_trader.data.synthetic import generate_synthetic_dataset
from agy_trader.nlp.sentiment import LexiconSentimentScorer
from agy_trader.storage.db import StateDB
from agy_trader.storage.parquet_store import ParquetBarStore

REPO_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
runner = CliRunner()


def _seed_operational_state(data_dir: Path, *, now_utc: datetime) -> StateDB:
    db = StateDB(data_dir / "state.db")
    bar_store = ParquetBarStore(data_dir / "parquet")
    universe = load_universe(REPO_CONFIG_DIR)
    generate_synthetic_dataset(universe=universe, bar_store=bar_store, db=db, days=320, seed=7)

    # Score news so unscored backlog is 0
    scorer = LexiconSentimentScorer()
    while True:
        batch = db.get_unscored_news(scorer_model=scorer.model_name, limit=200)
        if not batch:
            break
        preds = scorer.score_articles(batch)
        db.update_news_scores(
            [
                (a.article_id, p.score, p.label, p.confidence, p.scorer_model)
                for a, p in zip(batch, preds, strict=True)
            ]
        )

    # Add fresh macro points (DGS3MO, VIXCLS)
    today_d = now_utc.date()
    db.upsert_macro(
        [
            MacroPoint(as_of_date=today_d - timedelta(days=2), series_id="DGS3MO", value=4.35),
            MacroPoint(as_of_date=today_d - timedelta(days=2), series_id="VIXCLS", value=17.2),
        ]
    )

    # Add recent earnings & SEC Form 4 / 8-K Item 4.02
    db.upsert_earnings([EarningsEvent(symbol="AAPL", earnings_date=today_d)])
    db.upsert_sec_filings(
        [
            SecFilingEvent(
                accession_number="0001-26-004020",
                symbol="MSFT",
                cik="0000789019",
                form_type="8-K",
                accepted_at=now_utc - timedelta(hours=3),
                items=["4.02"],
            ),
            SecFilingEvent(
                accession_number="0001-26-000004",
                symbol="NVDA",
                cik="0001045810",
                form_type="4",
                accepted_at=now_utc - timedelta(hours=5),
                net_insider_buy_usd=450_000.0,
                insider_buyer_count=2,
            ),
        ]
    )

    # Add forward-only RSS & recent model news
    db.upsert_news_articles(
        [
            NewsArticle(
                article_id="rss:fed:test1",
                published_at=now_utc - timedelta(hours=2),
                headline="Federal Reserve releases monetary policy report",
                source="rss:fed_fomc",
                symbols=["SPY", "TLT"],
                sentiment_score=0.15,
                sentiment_label="positive",
                confidence=0.7,
                scorer_model="lexicon-v1",
            ),
            NewsArticle(
                article_id="bz:test1",
                published_at=now_utc - timedelta(hours=1),
                headline="AAPL beats estimates with record services revenue",
                source="benzinga",
                symbols=["AAPL"],
                sentiment_score=0.85,
                sentiment_label="positive",
                confidence=0.9,
                scorer_model="lexicon-v1",
            ),
        ]
    )

    # Add 2 portfolio snapshots
    db.insert_portfolio_snapshot(
        timestamp=now_utc - timedelta(days=1),
        equity=100_000.0,
        cash=40_000.0,
        buying_power=140_000.0,
        gross_exposure=0.60,
        net_exposure=0.60,
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
        high_water_mark=100_000.0,
        spy_close=500.0,
        ew50_index=100.0,
    )
    db.insert_portfolio_snapshot(
        timestamp=now_utc,
        equity=102_500.0,
        cash=38_000.0,
        buying_power=138_000.0,
        gross_exposure=0.62,
        net_exposure=0.62,
        daily_pnl_pct=0.025,
        drawdown_pct=-0.01,
        high_water_mark=103_535.0,
        spy_close=505.0,
        ew50_index=101.5,
    )

    # Live positions with GTC stops
    db.replace_live_positions(
        [
            {
                "symbol": "AAPL",
                "qty": 50.0,
                "avg_entry_price": 195.0,
                "market_price": 200.0,
                "unrealized_pnl": 250.0,
                "stop_order_id": "stp-aapl-1",
                "stop_price": 184.0,
                "updated_at": now_utc,
            }
        ]
    )

    # Live orders (SPY ETF + AAPL equity)
    db.upsert_live_order(
        client_order_id="agy:20261120:1540:SPY:B:p2m0",
        broker_order_id="brk-spy-1",
        run_id="rebal-1",
        symbol="SPY",
        side="buy",
        order_type="limit",
        qty=20.0,
        limit_price=505.5,
        status="filled",
        filled_qty=20.0,
        filled_avg_price=505.2,
        reference_price=505.0,
        realized_slippage_bps=3.96,
        is_risk_reducing=False,
        submitted_at=now_utc,
        updated_at=now_utc,
    )
    db.upsert_live_order(
        client_order_id="agy:20261120:1540:AAPL:B:p2m0",
        broker_order_id="brk-aapl-1",
        run_id="rebal-1",
        symbol="AAPL",
        side="buy",
        order_type="limit",
        qty=50.0,
        limit_price=200.2,
        status="filled",
        filled_qty=50.0,
        filled_avg_price=200.1,
        reference_price=200.0,
        realized_slippage_bps=5.0,
        is_risk_reducing=False,
        submitted_at=now_utc,
        updated_at=now_utc,
    )

    # Control audit log + 4 fresh daemon heartbeats + boot/reconcile/rebalance events
    ControlPlane(db).set_exposure(0.95, actor="ops", reason="Test exposure audit", now=now_utc)
    for dname in ("market-data", "news-nlp", "trader", "dashboard"):
        db.upsert_daemon_heartbeat(
            dname,
            "healthy",
            {"pid": 12345, "rss_mb": 42.5},
            now_utc=now_utc - timedelta(seconds=10),
        )
    db.record_daemon_event("trader", "boot", {"paper_mode": True}, timestamp=now_utc - timedelta(hours=1))
    db.record_daemon_event("trader", "reconcile", {"diffs": 0}, timestamp=now_utc - timedelta(minutes=5))
    db.record_daemon_event(
        "trader",
        "rebalance",
        {"session_date": today_d.isoformat(), "checkpoint": "15:40", "status": "completed"},
        timestamp=now_utc,
    )
    return db


def test_dashboard_missing_db_and_host_guard(tmp_path: Path) -> None:
    empty_data = tmp_path / "empty_data"
    app = create_dashboard_app(empty_data, REPO_CONFIG_DIR)
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert client.get("/api/overview").json()["has_data"] is False
    assert client.get("/api/equity-curve").json()["series"] == []
    assert client.get("/api/positions").json()["positions"] == []
    assert client.get("/api/orders").json()["orders"] == []
    assert client.get("/api/health").json()["wal_ok"] is False
    assert client.get("/api/intelligence").json()["model_news"] == []

    # Missing DB must NOT be created by read-only dashboard requests!
    assert not (empty_data / "state.db").exists()

    # Localhost-only host guard rejects 0.0.0.0
    with pytest.raises(ValueError, match="localhost"):
        serve_dashboard(empty_data, REPO_CONFIG_DIR, host="0.0.0.0", port=8080)


def test_dashboard_routes_read_only_and_405_mutations(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    now = datetime(2026, 11, 20, 20, 45, tzinfo=UTC)  # 15:45 ET Friday
    _seed_operational_state(data_dir, now_utc=now)

    # Confirm strict SQLite ?mode=ro rejects writes
    ro_db = StateDB(data_dir / "state.db", read_only=True)
    with pytest.raises(sqlite3.OperationalError):
        ro_db.upsert_daemon_heartbeat("evil", "hacked")

    app = create_dashboard_app(data_dir, REPO_CONFIG_DIR)
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200

    ov = client.get("/api/overview")
    assert ov.status_code == 200
    ov_json = ov.json()
    assert ov_json["has_data"] is True
    assert ov_json["equity"] == 102_500.0
    assert ov_json["total_return_pct"] == pytest.approx(2.5)
    assert ov_json["spy_return_pct"] == pytest.approx(1.0)
    assert ov_json["ew50_return_pct"] == pytest.approx(1.5)
    assert ov_json["control_dials"]["exposure_multiplier"] == pytest.approx(0.95)

    curve = client.get("/api/equity-curve?limit=100")
    assert curve.status_code == 200
    series = curve.json()["series"]
    assert len(series) == 2
    assert series[0]["portfolio_index"] == 100.0
    assert series[1]["portfolio_index"] == 102.5
    assert series[1]["spy_index"] == 101.0
    assert series[1]["ew50_index"] == 101.5
    assert client.get("/api/equity-curve?limit=5001").status_code == 422

    pos = client.get("/api/positions")
    assert pos.status_code == 200
    p0 = pos.json()["positions"][0]
    assert p0["symbol"] == "AAPL"
    assert p0["stop_synced"] is True
    assert p0["distance_to_stop_pct"] == pytest.approx(8.0)

    orders = client.get("/api/orders?limit=50")
    assert orders.status_code == 200
    by_sym = {o["symbol"]: o for o in orders.json()["orders"]}
    assert by_sym["SPY"]["backtest_budget_bps"] == 5.0
    assert by_sym["AAPL"]["backtest_budget_bps"] == 10.0

    health = client.get("/api/health")
    assert health.status_code == 200
    h_json = health.json()
    assert h_json["wal_ok"] is True
    assert set(h_json["daemons"].keys()) == {"market-data", "news-nlp", "trader", "dashboard"}
    assert h_json["last_rebalance"] is not None

    intel = client.get("/api/intelligence?limit=20")
    assert intel.status_code == 200
    i_json = intel.json()
    assert any(a["article_id"] == "bz:test1" for a in i_json["model_news"])
    assert any(a["article_id"] == "rss:fed:test1" for a in i_json["forward_only_news"])
    assert all(not a["source"].startswith("rss:") for a in i_json["model_news"])
    assert len(i_json["sec_filings"]) == 2
    assert len(i_json["audit_log"]) >= 1

    # All mutating HTTP verbs must return 405 Method Not Allowed
    for path in ("/api/overview", "/api/orders", "/api/positions", "/api/health"):
        assert client.post(path).status_code == 405
        assert client.put(path).status_code == 405
        assert client.delete(path).status_code == 405
        assert client.patch(path).status_code == 405


def test_reporting_builders_and_agy_ops_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NLP_BACKEND", "lexicon")
    empty_dir = tmp_path / "missing"
    missing_ready = build_readiness_report(empty_dir, REPO_CONFIG_DIR)
    assert missing_ready["status"] == "CRITICAL"

    data_dir = tmp_path / "data"
    # Match generate_synthetic_dataset's ending trading session (2025-12-31) at 15:45 ET
    now = datetime(2025, 12, 31, 20, 45, tzinfo=UTC)
    db = _seed_operational_state(data_dir, now_utc=now)

    ready = build_readiness_report(data_dir, REPO_CONFIG_DIR, now_utc=now)
    assert ready["status"] == "READY", ready["issues"]

    morning = build_morning_plan(data_dir, REPO_CONFIG_DIR, now_utc=now)
    assert morning["macro_regime_scale"] > 0.0
    assert any(r["symbol"] == "MSFT" for r in morning["restatement_blackouts"])
    assert "Morning Plan" in morning["markdown_summary"]

    # Isolated config checks: earnings_blackout_days=None vs earnings_blackout_days=2
    raw_strat = yaml.safe_load((REPO_CONFIG_DIR / "strategy.yaml").read_text(encoding="utf-8"))
    univ_txt = (REPO_CONFIG_DIR / "universe.yaml").read_text(encoding="utf-8")

    cfg_disabled = tmp_path / "cfg_earn_disabled"
    cfg_disabled.mkdir()
    raw_dis = {**raw_strat, "risk": {**raw_strat["risk"], "earnings_blackout_days": None}}
    (cfg_disabled / "strategy.yaml").write_text(yaml.safe_dump(raw_dis), encoding="utf-8")
    (cfg_disabled / "universe.yaml").write_text(univ_txt, encoding="utf-8")
    morning_disabled = build_morning_plan(data_dir, cfg_disabled, now_utc=now)
    assert morning_disabled["earnings_blackout_days"] is None
    assert morning_disabled["earnings_blackouts"] == []
    assert "**Earnings Blackouts Today:** Disabled" in morning_disabled["markdown_summary"]

    cfg_earn = tmp_path / "cfg_earn"
    cfg_earn.mkdir()
    raw_en = {**raw_strat, "risk": {**raw_strat["risk"], "earnings_blackout_days": 2}}
    (cfg_earn / "strategy.yaml").write_text(yaml.safe_dump(raw_en), encoding="utf-8")
    (cfg_earn / "universe.yaml").write_text(univ_txt, encoding="utf-8")
    morning_with_earn = build_morning_plan(data_dir, cfg_earn, now_utc=now)
    assert morning_with_earn["earnings_blackout_days"] == 2
    assert any(e["symbol"] == "AAPL" for e in morning_with_earn["earnings_blackouts"])
    assert "**Earnings Blackouts Today:** AAPL" in morning_with_earn["markdown_summary"]

    daily = build_daily_summary(data_dir, REPO_CONFIG_DIR, now_utc=now)
    assert daily["equity"] == 102_500.0
    assert daily["cumulative_return_pct"] == pytest.approx(2.5)
    expected_headroom = load_app_config(REPO_CONFIG_DIR).risk.max_drawdown_halt_pct * 100.0 - 1.0
    assert daily["drawdown_headroom_pct"] == pytest.approx(expected_headroom)
    assert len(daily["todays_fills"]) == 2
    assert daily["mean_realized_slippage_bps"] == pytest.approx(4.48)
    assert "Daily Close Summary" in daily["markdown_summary"]

    # Unprotected open position -> CRITICAL readiness
    db.replace_live_positions(
        [
            {
                "symbol": "AAPL",
                "qty": 50.0,
                "avg_entry_price": 195.0,
                "market_price": 200.0,
                "unrealized_pnl": 250.0,
                "stop_order_id": None,
                "stop_price": None,
                "updated_at": now,
            }
        ]
    )
    crit = build_readiness_report(data_dir, REPO_CONFIG_DIR, now_utc=now)
    assert crit["status"] == "CRITICAL"
    assert any(i["check"] == "protective_stops" for i in crit["issues"])

    # CLI smoke tests
    res_ver = runner.invoke(ops_cli_app, ["version"])
    assert res_ver.exit_code == 0
    assert "agy-ops" in res_ver.stdout

    res_ready = runner.invoke(
        ops_cli_app,
        ["readiness", "--data-dir", str(data_dir), "--config-dir", str(REPO_CONFIG_DIR), "--json"],
    )
    assert res_ready.exit_code == 0, res_ready.output
    assert json.loads(res_ready.stdout)["status"] == "CRITICAL"

    res_mp = runner.invoke(
        ops_cli_app,
        ["morning-plan", "--data-dir", str(data_dir), "--config-dir", str(REPO_CONFIG_DIR), "--json"],
    )
    assert res_mp.exit_code == 0, res_mp.output
    assert "macro_regime_scale" in json.loads(res_mp.stdout)

    res_ds = runner.invoke(
        ops_cli_app,
        ["daily-summary", "--data-dir", str(data_dir), "--config-dir", str(REPO_CONFIG_DIR), "--json"],
    )
    assert res_ds.exit_code == 0, res_ds.output
    assert json.loads(res_ds.stdout)["equity"] == 102_500.0

    res_dash = runner.invoke(
        ops_cli_app,
        [
            "daemon",
            "dashboard",
            "--once",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
        ],
    )
    assert res_dash.exit_code == 0, res_dash.output
    assert json.loads(res_dash.stdout)["status"] == "healthy"

    # Reject 0.0.0.0 via CLI too
    res_bad_host = runner.invoke(
        ops_cli_app,
        [
            "daemon",
            "dashboard",
            "--once",
            "--host",
            "0.0.0.0",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
        ],
    )
    assert res_bad_host.exit_code != 0

    out_manifests = tmp_path / "manifests"
    res_man = runner.invoke(
        ops_cli_app,
        ["render-manifests", "--target", "systemd", "--out-dir", str(out_manifests)],
    )
    assert res_man.exit_code == 0, res_man.output
    assert len(json.loads(res_man.stdout)["files"]) == 4


def test_m4_inception_snapshot_preserved_and_security_headers(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    now = datetime(2025, 12, 31, 20, 45, tzinfo=UTC)
    db = StateDB(data_dir / "state.db")

    # Insert inception snapshot (>90d ago) FIRST, then two invested snapshots (>90d ago), then current state
    inception_ts = now - timedelta(days=200)
    mid_ts = now - timedelta(days=120)
    mid2_ts = now - timedelta(days=100)
    db.insert_portfolio_snapshot(
        timestamp=inception_ts,
        equity=95_000.0,
        cash=95_000.0,
        buying_power=95_000.0,
        gross_exposure=0.0,
        net_exposure=0.0,
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
        high_water_mark=95_000.0,
        spy_close=450.0,
        ew50_index=90.0,
    )
    db.insert_portfolio_snapshot(
        timestamp=mid_ts,
        equity=97_000.0,
        cash=50_000.0,
        buying_power=100_000.0,
        gross_exposure=0.5,
        net_exposure=0.5,
        daily_pnl_pct=0.01,
        drawdown_pct=0.0,
        high_water_mark=97_000.0,
        spy_close=470.0,
        ew50_index=94.0,
    )
    db.insert_portfolio_snapshot(
        timestamp=mid2_ts,
        equity=98_500.0,
        cash=49_000.0,
        buying_power=98_500.0,
        gross_exposure=0.5,
        net_exposure=0.5,
        daily_pnl_pct=0.015,
        drawdown_pct=0.0,
        high_water_mark=98_500.0,
        spy_close=480.0,
        ew50_index=96.0,
    )
    _seed_operational_state(data_dir, now_utc=now)

    pruned = db.prune_old_telemetry(keep_days=90, now_utc=now)
    assert pruned["portfolio_snapshots"] == 1  # mid2_ts deleted; inception_ts & mid_ts preserved!
    first = db.get_first_portfolio_snapshot()
    assert first is not None
    assert first["equity"] == 95_000.0
    first_invested = db.get_first_invested_portfolio_snapshot()
    assert first_invested is not None
    assert first_invested["spy_close"] == 470.0

    # Verify security headers, disabled OpenAPI schema, peak_rss_mb, and preserved inception return
    app = create_dashboard_app(data_dir, REPO_CONFIG_DIR)
    client = TestClient(app)

    assert client.get("/openapi.json").status_code == 404
    ov = client.get("/api/overview")
    assert ov.status_code == 200
    assert ov.headers.get("x-content-type-options") == "nosniff"
    assert "default-src 'self'" in ov.headers.get("content-security-policy", "")
    # 102,500 vs 95,000 inception; benchmarks anchored to first invested snapshot (470 / 94)
    assert ov.json()["total_return_pct"] == pytest.approx(((102_500.0 / 95_000.0) - 1.0) * 100.0, rel=1e-4)
    assert ov.json()["spy_return_pct"] == pytest.approx(((505.0 / 470.0) - 1.0) * 100.0, rel=1e-4)
    assert ov.json()["ew50_return_pct"] == pytest.approx(((101.5 / 94.0) - 1.0) * 100.0, rel=1e-4)
    assert ov.json()["spy_daily_return_pct"] == pytest.approx(1.0)
    assert ov.json()["ew50_daily_return_pct"] == pytest.approx(1.5)

    health = client.get("/api/health").json()
    assert health["daemons"]["trader"]["peak_rss_mb"] == 42.5

    daily = build_daily_summary(data_dir, REPO_CONFIG_DIR, now_utc=now)
    assert daily["cumulative_return_pct"] == pytest.approx(((102_500.0 / 95_000.0) - 1.0) * 100.0, rel=1e-4)
    assert daily["spy_cumulative_return_pct"] == pytest.approx(((505.0 / 470.0) - 1.0) * 100.0, rel=1e-4)
    assert daily["ew50_cumulative_return_pct"] == pytest.approx(((101.5 / 94.0) - 1.0) * 100.0, rel=1e-4)
    assert daily["spy_daily_return_pct"] == pytest.approx(1.0)
    assert daily["ew50_daily_return_pct"] == pytest.approx(1.5)


def test_daily_summary_benchmark_anchoring_rth_and_holidays(tmp_path: Path) -> None:
    from zoneinfo import ZoneInfo

    ny_tz = ZoneInfo("America/New_York")

    # Case 1-3: Cash-idle rally ignored, extended-hours stub quotes ignored,
    # Thanksgiving holiday (2026-11-26) skipped, Black Friday (2026-11-27) 13:00 ET early close respected.
    data_dir = tmp_path / "data_rth"
    data_dir.mkdir(parents=True)
    db = StateDB(data_dir / "state.db")

    # Pre-entry cash-idle snapshots on Nov 23 (SPY rallies +4% while account is 100% cash)
    db.insert_portfolio_snapshot(
        timestamp=datetime(2026, 11, 23, 10, 0, tzinfo=ny_tz).astimezone(UTC),
        equity=100_000.0,
        cash=100_000.0,
        buying_power=100_000.0,
        gross_exposure=0.0,
        net_exposure=0.0,
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
        high_water_mark=100_000.0,
        spy_close=480.7692,
        ew50_index=95.0,
    )
    # First invested snapshot on Nov 23 @ 15:40 ET (SPY = 500.0, EW-50 = 100.0)
    db.insert_portfolio_snapshot(
        timestamp=datetime(2026, 11, 23, 15, 40, tzinfo=ny_tz).astimezone(UTC),
        equity=100_000.0,
        cash=40_000.0,
        buying_power=140_000.0,
        gross_exposure=0.60,
        net_exposure=0.60,
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
        high_water_mark=100_000.0,
        spy_close=500.0,
        ew50_index=100.0,
    )
    # Nov 25 (Wed before Thanksgiving): RTH close @ 15:58 ET vs distorted post-close stub @ 17:00 ET
    db.insert_portfolio_snapshot(
        timestamp=datetime(2026, 11, 25, 15, 58, tzinfo=ny_tz).astimezone(UTC),
        equity=101_000.0,
        cash=40_000.0,
        buying_power=141_000.0,
        gross_exposure=0.60,
        net_exposure=0.60,
        daily_pnl_pct=0.01,
        drawdown_pct=0.0,
        high_water_mark=101_000.0,
        spy_close=505.0,
        ew50_index=101.0,
    )
    db.insert_portfolio_snapshot(
        timestamp=datetime(2026, 11, 25, 17, 0, tzinfo=ny_tz).astimezone(UTC),
        equity=101_000.0,
        cash=40_000.0,
        buying_power=141_000.0,
        gross_exposure=0.60,
        net_exposure=0.60,
        daily_pnl_pct=0.01,
        drawdown_pct=0.0,
        high_water_mark=101_000.0,
        spy_close=520.0,
        ew50_index=115.0,
    )
    # Nov 26 (Thanksgiving Day - NYSE holiday): stale/stub quote must be skipped
    db.insert_portfolio_snapshot(
        timestamp=datetime(2026, 11, 26, 12, 0, tzinfo=ny_tz).astimezone(UTC),
        equity=101_000.0,
        cash=40_000.0,
        buying_power=141_000.0,
        gross_exposure=0.60,
        net_exposure=0.60,
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
        high_water_mark=101_000.0,
        spy_close=490.0,
        ew50_index=90.0,
    )
    # Nov 27 (Black Friday - 13:00 ET early close): 12:58 ET RTH mark vs 14:00 ET post-close stub
    db.insert_portfolio_snapshot(
        timestamp=datetime(2026, 11, 27, 12, 58, tzinfo=ny_tz).astimezone(UTC),
        equity=102_000.0,
        cash=40_000.0,
        buying_power=142_000.0,
        gross_exposure=0.60,
        net_exposure=0.60,
        daily_pnl_pct=0.0099,
        drawdown_pct=0.0,
        high_water_mark=102_000.0,
        spy_close=510.05,
        ew50_index=103.02,
    )
    db.insert_portfolio_snapshot(
        timestamp=datetime(2026, 11, 27, 14, 0, tzinfo=ny_tz).astimezone(UTC),
        equity=102_000.0,
        cash=40_000.0,
        buying_power=142_000.0,
        gross_exposure=0.60,
        net_exposure=0.60,
        daily_pnl_pct=0.0099,
        drawdown_pct=0.0,
        high_water_mark=102_000.0,
        spy_close=530.0,
        ew50_index=120.0,
    )

    # Nov 28 (Saturday) & Nov 30 08:00 ET (Monday pre-market): closed-market hourly stub quotes must be ignored!
    db.insert_portfolio_snapshot(
        timestamp=datetime(2026, 11, 28, 10, 0, tzinfo=ny_tz).astimezone(UTC),
        equity=102_000.0,
        cash=40_000.0,
        buying_power=142_000.0,
        gross_exposure=0.60,
        net_exposure=0.60,
        daily_pnl_pct=0.0099,
        drawdown_pct=0.0,
        high_water_mark=102_000.0,
        spy_close=475.0,
        ew50_index=85.0,
    )
    db.insert_portfolio_snapshot(
        timestamp=datetime(2026, 11, 30, 8, 0, tzinfo=ny_tz).astimezone(UTC),
        equity=102_000.0,
        cash=40_000.0,
        buying_power=142_000.0,
        gross_exposure=0.60,
        net_exposure=0.60,
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
        high_water_mark=102_000.0,
        spy_close=470.0,
        ew50_index=82.0,
    )

    # Evaluate on Saturday Nov 28 AND Monday Nov 30 pre-open: both must select Fri Nov 27 12:58 ET vs Wed Nov 25 15:58 ET
    for eval_now in (
        datetime(2026, 11, 28, 16, 0, tzinfo=ny_tz).astimezone(UTC),
        datetime(2026, 11, 30, 8, 30, tzinfo=ny_tz).astimezone(UTC),
    ):
        summary = build_daily_summary(data_dir, REPO_CONFIG_DIR, now_utc=eval_now)
        assert summary["spy_daily_return_pct"] == pytest.approx(((510.05 / 505.0) - 1.0) * 100.0, rel=1e-4)
        assert summary["ew50_daily_return_pct"] == pytest.approx(((103.02 / 101.0) - 1.0) * 100.0, rel=1e-4)
        # Cumulative benchmark anchored to first invested snapshot (500.0 / 100.0), NOT cash-idle 480.7692
        assert summary["spy_cumulative_return_pct"] == pytest.approx(((510.05 / 500.0) - 1.0) * 100.0, rel=1e-4)
        assert summary["ew50_cumulative_return_pct"] == pytest.approx(((103.02 / 100.0) - 1.0) * 100.0, rel=1e-4)
        assert summary["benchmark_anchor_timestamp"] is not None

    # Also verify /api/equity-curve anchors SPY/EW-50 to first invested snapshot (500.0 / 100.0)
    app = create_dashboard_app(data_dir, REPO_CONFIG_DIR)
    client = TestClient(app)
    ov = client.get("/api/overview").json()
    assert ov["spy_daily_return_pct"] == pytest.approx(summary["spy_daily_return_pct"])
    assert ov["ew50_daily_return_pct"] == pytest.approx(summary["ew50_daily_return_pct"])
    assert ov["spy_return_pct"] == pytest.approx(summary["spy_cumulative_return_pct"])
    assert ov["ew50_return_pct"] == pytest.approx(summary["ew50_cumulative_return_pct"])

    curve_series = client.get("/api/equity-curve").json()["series"]
    # Index 1 is first invested snapshot (500.0 / 100.0) -> 100.0 baseline
    assert curve_series[1]["spy_index"] == pytest.approx(100.0)
    assert curve_series[1]["ew50_index"] == pytest.approx(100.0)

    # Case 4: Uninvested account (gross_exposure == 0.0 on all rows)
    cash_dir = tmp_path / "data_cash_only"
    cash_dir.mkdir(parents=True)
    cash_db = StateDB(cash_dir / "state.db")
    cash_db.insert_portfolio_snapshot(
        timestamp=datetime(2026, 11, 24, 15, 55, tzinfo=ny_tz).astimezone(UTC),
        equity=100_000.0,
        cash=100_000.0,
        buying_power=100_000.0,
        gross_exposure=0.0,
        net_exposure=0.0,
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
        high_water_mark=100_000.0,
        spy_close=500.0,
        ew50_index=100.0,
    )
    cash_db.insert_portfolio_snapshot(
        timestamp=datetime(2026, 11, 25, 15, 55, tzinfo=ny_tz).astimezone(UTC),
        equity=100_000.0,
        cash=100_000.0,
        buying_power=100_000.0,
        gross_exposure=0.0,
        net_exposure=0.0,
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
        high_water_mark=100_000.0,
        spy_close=505.0,
        ew50_index=102.0,
    )
    cash_summary = build_daily_summary(
        cash_dir,
        REPO_CONFIG_DIR,
        now_utc=datetime(2026, 11, 25, 16, 5, tzinfo=ny_tz).astimezone(UTC),
    )
    assert cash_summary["spy_cumulative_return_pct"] is None
    assert cash_summary["ew50_cumulative_return_pct"] is None
    assert cash_summary["benchmark_anchor_timestamp"] is None
    assert cash_summary["spy_daily_return_pct"] == pytest.approx(1.0)
    assert cash_summary["ew50_daily_return_pct"] == pytest.approx(2.0)
    assert "since first trade — SPY: N/A, EW-50: N/A" in cash_summary["markdown_summary"]
