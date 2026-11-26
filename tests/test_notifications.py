from __future__ import annotations

import json
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from agy_ops.cli import app as ops_cli_app
from agy_ops.notifications import (
    DiscordWebhookNotifier,
    build_daily_summary_discord_embed,
    build_morning_discord_embed,
    build_test_discord_embed,
    redact_webhook_url,
    validate_discord_webhook_url,
)
from agy_ops.reporting import render_cumulative_performance_png
from agy_trader.core.config import EnvSettings, load_universe
from agy_trader.core.models import EarningsEvent, MacroPoint, NewsArticle, SecFilingEvent
from agy_trader.data.synthetic import generate_synthetic_dataset
from agy_trader.nlp.sentiment import LexiconSentimentScorer
from agy_trader.storage.db import StateDB
from agy_trader.storage.parquet_store import ParquetBarStore

REPO_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
SAMPLE_TOKEN = "super_secret_webhook_token_XYZ123"
SAMPLE_WEBHOOK = f"https://discord.com/api/webhooks/123456789012345678/{SAMPLE_TOKEN}"
runner = CliRunner()


def _seed_ops_data(data_dir: Path, *, now_utc: datetime) -> StateDB:
    db = StateDB(data_dir / "state.db")
    bar_store = ParquetBarStore(data_dir / "parquet")
    universe = load_universe(REPO_CONFIG_DIR)
    generate_synthetic_dataset(universe=universe, bar_store=bar_store, db=db, days=320, seed=11)

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

    today_d = now_utc.date()
    db.upsert_macro(
        [
            MacroPoint(as_of_date=today_d - timedelta(days=1), series_id="DGS3MO", value=4.25),
            MacroPoint(as_of_date=today_d - timedelta(days=1), series_id="VIXCLS", value=16.5),
        ]
    )
    db.upsert_earnings([EarningsEvent(symbol="AAPL", earnings_date=today_d)])
    db.upsert_sec_filings(
        [
            SecFilingEvent(
                accession_number="0001-26-004020",
                symbol="MSFT",
                cik="0000789019",
                form_type="8-K",
                accepted_at=now_utc - timedelta(hours=2),
                items=["4.02"],
            )
        ]
    )
    db.upsert_news_articles(
        [
            NewsArticle(
                article_id="rss:fed:notify1",
                published_at=now_utc - timedelta(hours=2),
                headline="Federal Reserve publishes minutes",
                source="rss:fed_fomc",
                symbols=["SPY"],
                sentiment_score=0.0,
                sentiment_label="neutral",
                confidence=0.8,
                scorer_model="lexicon-v1",
            ),
            NewsArticle(
                article_id="bz:notify1",
                published_at=now_utc - timedelta(hours=1),
                headline="NVDA launches new accelerator architecture",
                source="benzinga",
                symbols=["NVDA"],
                sentiment_score=0.75,
                sentiment_label="positive",
                confidence=0.9,
                scorer_model="lexicon-v1",
            ),
        ]
    )
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
        equity=101_500.0,
        cash=39_000.0,
        buying_power=139_000.0,
        gross_exposure=0.61,
        net_exposure=0.61,
        daily_pnl_pct=0.015,
        drawdown_pct=-0.005,
        high_water_mark=102_000.0,
        spy_close=504.0,
        ew50_index=101.0,
    )
    db.replace_live_positions(
        [
            {
                "symbol": "AAPL",
                "qty": 40.0,
                "avg_entry_price": 195.0,
                "market_price": 200.0,
                "unrealized_pnl": 200.0,
                "stop_order_id": "stp-aapl-1",
                "stop_price": 185.0,
                "updated_at": now_utc,
            }
        ]
    )
    db.upsert_live_order(
        client_order_id="agy:20251231:1540:AAPL:B:p1m0",
        broker_order_id="brk-aapl-1",
        run_id="rebal-1",
        symbol="AAPL",
        side="buy",
        order_type="limit",
        qty=40.0,
        limit_price=200.2,
        status="filled",
        filled_qty=40.0,
        filled_avg_price=200.1,
        reference_price=200.0,
        realized_slippage_bps=5.0,
        is_risk_reducing=False,
        submitted_at=now_utc,
        updated_at=now_utc,
    )
    for dname in ("market-data", "news-nlp", "trader", "dashboard"):
        db.upsert_daemon_heartbeat(
            dname,
            "healthy",
            {"pid": 1111, "rss_mb": 40.0},
            now_utc=now_utc - timedelta(seconds=10),
        )
    return db


def test_env_settings_has_discord_webhook_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", SAMPLE_WEBHOOK)
    settings = EnvSettings()
    assert settings.discord_webhook_url == SAMPLE_WEBHOOK


def test_redact_and_validate_discord_webhook_url() -> None:
    assert (
        redact_webhook_url(SAMPLE_WEBHOOK)
        == "https://discord.com/api/webhooks/123456789012345678/***"
    )
    assert (
        redact_webhook_url(f"{SAMPLE_WEBHOOK}?wait=true")
        == "https://discord.com/api/webhooks/123456789012345678/***?wait=true"
    )
    assert (
        redact_webhook_url(f"https://discord.com/API/WEBHOOKS/123456/{SAMPLE_TOKEN}")
        == "https://discord.com/API/WEBHOOKS/123456/***"
    )

    for host in ("discord.com", "discordapp.com", "ptb.discord.com", "canary.discord.com"):
        valid = f"https://{host}/api/webhooks/987654321/{SAMPLE_TOKEN}"
        assert validate_discord_webhook_url(f"  {valid}  ") == valid
        assert (
            validate_discord_webhook_url(
                f"HTTPS://{host.upper()}/api/webhooks/987654321/{SAMPLE_TOKEN}"
            )
            == valid
        )

    bad_urls = [
        "",
        f"http://discord.com/api/webhooks/123456/{SAMPLE_TOKEN}",
        f"https://evil.example.com/api/webhooks/123456/{SAMPLE_TOKEN}",
        f"https://discord.com/api/webhooks/notdigits/{SAMPLE_TOKEN}",
        "https://discord.com/api/webhooks/123456/",
        f"https://discord.com/webhooks/123456/{SAMPLE_TOKEN}",
        f"https://user:pass@discord.com/api/webhooks/123456/{SAMPLE_TOKEN}",
        f"https://discord.com:8443/api/webhooks/123456/{SAMPLE_TOKEN}",
        f"https://discord.com:/api/webhooks/123456/{SAMPLE_TOKEN}",
        f"https://discord.com/api/webhooks/123456/{SAMPLE_TOKEN}#frag",
        f"https://discord.com/api/webhooks/123456/{SAMPLE_TOKEN}?foo=bar",
        f"https://discord.com/api/webhooks/123456/{SAMPLE_TOKEN}?",
        f"https://discord.com/api/webhooks/123456/..%2f{SAMPLE_TOKEN}",
        f"https://discord.com/api/webhooks/123456/{SAMPLE_TOKEN}/..",
        f"https://discord.com/api/webhooks/123456/{SAMPLE_TOKEN}%2fextra",
        f"https://discord.com/api/webhooks/123456/{SAMPLE_TOKEN}.dot",
        f"https://discord.com/api/webhooks/123456/tok\r\n{SAMPLE_TOKEN}",
        f"https://discord.com/api/webhooks/123456/tok\t{SAMPLE_TOKEN}",
        f"https://discord.com/api/webhooks/123456/tok\x00{SAMPLE_TOKEN}",
        f"https://discord.com/api/webhooks/123456/tok\x7f{SAMPLE_TOKEN}",
        f"https://discord.com/api/webhooks/123456/tok {SAMPLE_TOKEN}",
    ]
    for bad in bad_urls:
        with pytest.raises(ValueError) as exc_info:
            validate_discord_webhook_url(bad)
        assert SAMPLE_TOKEN not in str(exc_info.value)


@respx.mock
def test_notifier_context_manager_and_embed_limits() -> None:
    route = respx.post(SAMPLE_WEBHOOK).mock(return_value=httpx.Response(204))

    long_fields = [
        {"name": f"F{i}-" + ("N" * 300), "value": f"V{i}-" + ("X" * 1200), "inline": False}
        for i in range(30)
    ]
    with DiscordWebhookNotifier(SAMPLE_WEBHOOK) as notifier:
        assert notifier._owned_client is not None
        assert not notifier._owned_client.is_closed
        result = notifier.send_embed(
            title="T" * 300,
            description="D" * 5000,
            color=0x22C55E,
            fields=long_fields,
            footer="agy-trader",
        )
    assert notifier._owned_client is None
    assert route.called
    assert result["ok"] is True
    assert result["status_code"] == 204
    assert SAMPLE_TOKEN not in json.dumps(result)

    sent_payload = json.loads(route.calls[0].request.content.decode("utf-8"))
    embed = sent_payload["embeds"][0]
    assert len(embed["title"]) <= 256
    assert len(embed["description"]) <= 4096
    assert len(embed["fields"]) <= 25
    for f in embed["fields"]:
        assert len(f["name"]) <= 256
        assert len(f["value"]) <= 1024
    total_chars = (
        len(embed["title"])
        + len(embed["description"])
        + len(embed["footer"]["text"])
        + sum(len(f["name"]) + len(f["value"]) for f in embed["fields"])
    )
    assert total_chars <= 6000


@respx.mock
def test_notifier_retries_429_and_5xx_with_capped_backoff() -> None:
    sleeps: list[float] = []
    route = respx.post(SAMPLE_WEBHOOK).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "25"}),  # capped at 10s
            httpx.Response(429, json={"retry_after": 1.5}),
            httpx.Response(503),
            httpx.Response(204),
        ]
    )

    external_client = httpx.Client()
    try:
        with DiscordWebhookNotifier(
            SAMPLE_WEBHOOK,
            client=external_client,
            max_retries=3,
            sleep_fn=sleeps.append,
        ) as notifier:
            res = notifier.send_embed(
                title="Retry Test",
                description="Testing 429 & 5xx",
                color=0x3B82F6,
            )
        assert not external_client.is_closed
    finally:
        external_client.close()

    assert res["status_code"] == 204
    assert route.call_count == 4
    # attempt 0 -> 429 Retry-After=25 capped to 10.0; attempt 1 -> json 1.5; attempt 2 -> 503 2**2=4.0
    assert sleeps == [10.0, 1.5, 4.0]


@respx.mock
def test_notifier_redacts_token_on_http_and_request_errors() -> None:
    respx.post(SAMPLE_WEBHOOK).mock(return_value=httpx.Response(404))
    notifier = DiscordWebhookNotifier(SAMPLE_WEBHOOK, max_retries=0)
    with pytest.raises(RuntimeError) as status_exc:
        notifier.send_embed(title="Fail", description="404", color=0xEF4444)
    msg = str(status_exc.value)
    assert status_exc.value.__cause__ is None
    assert SAMPLE_TOKEN not in msg
    assert SAMPLE_TOKEN not in "".join(traceback.format_exception(status_exc.value))
    assert "/webhooks/123456789012345678/***" in msg

    respx.post(SAMPLE_WEBHOOK).mock(return_value=httpx.Response(500))
    sleeps: list[float] = []
    retrying_notifier = DiscordWebhookNotifier(
        SAMPLE_WEBHOOK, max_retries=1, sleep_fn=sleeps.append
    )
    with pytest.raises(RuntimeError) as retry_exc:
        retrying_notifier.send_embed(title="Fail", description="500", color=0xEF4444)
    assert retry_exc.value.__cause__ is None
    assert sleeps == [1.0]
    assert SAMPLE_TOKEN not in str(retry_exc.value)
    assert SAMPLE_TOKEN not in "".join(traceback.format_exception(retry_exc.value))

    respx.post(SAMPLE_WEBHOOK).mock(
        side_effect=httpx.ConnectError(f"Failed connecting to {SAMPLE_WEBHOOK}")
    )
    with pytest.raises(RuntimeError) as req_exc:
        notifier.send_embed(title="Fail", description="ConnectError", color=0xEF4444)
    req_msg = str(req_exc.value)
    assert req_exc.value.__cause__ is None
    assert SAMPLE_TOKEN not in req_msg
    assert SAMPLE_TOKEN not in "".join(traceback.format_exception(req_exc.value))
    assert "/webhooks/123456789012345678/***" in req_msg


@respx.mock
def test_notifier_never_follows_redirects() -> None:
    evil_target = "https://evil.example.com/collect"
    evil_route = respx.post(evil_target).mock(return_value=httpx.Response(200))
    evil_get_route = respx.get(evil_target).mock(return_value=httpx.Response(200))

    for status_code in (302, 307):
        respx.post(SAMPLE_WEBHOOK).mock(
            return_value=httpx.Response(status_code, headers={"Location": evil_target})
        )
        # Owned client
        with DiscordWebhookNotifier(SAMPLE_WEBHOOK, max_retries=0) as owned_notifier:
            assert owned_notifier._owned_client is not None
            assert owned_notifier._owned_client.follow_redirects is False
            with pytest.raises(RuntimeError) as redir_exc:
                owned_notifier.send_embed(title="Redirect", description="30x", color=0xEF4444)
            assert redir_exc.value.__cause__ is None
            assert SAMPLE_TOKEN not in str(redir_exc.value)
            assert f"HTTP {status_code}" in str(redir_exc.value)

        # External client explicitly configured with follow_redirects=True must still NOT redirect
        ext_client = httpx.Client(follow_redirects=True)
        try:
            ext_notifier = DiscordWebhookNotifier(
                SAMPLE_WEBHOOK, client=ext_client, max_retries=0
            )
            with pytest.raises(RuntimeError) as ext_redir_exc:
                ext_notifier.send_embed(title="Redirect", description="30x", color=0xEF4444)
            assert ext_redir_exc.value.__cause__ is None
            assert SAMPLE_TOKEN not in str(ext_redir_exc.value)
            assert f"HTTP {status_code}" in str(ext_redir_exc.value)
        finally:
            ext_client.close()

    assert not evil_route.called
    assert not evil_get_route.called


def test_embed_formatters_morning_close_and_test() -> None:
    morning_plan = {
        "session_date_et": "2026-11-23",
        "equity": 101_500.0,
        "drawdown_pct": -0.5,
        "positions": [
            {"symbol": "AAPL", "qty": 40, "market_price": 200.0, "stop_price": 185.0}
        ],
        "macro_regime_scale": 0.92,
        "control_dials": {
            "pause_new_entries": False,
            "exposure_multiplier": 1.0,
            "allow_shorting": True,
        },
        "earnings_blackouts": [{"symbol": "AAPL", "earnings_date": "2026-11-23"}],
        "restatement_blackouts": [{"symbol": "MSFT"}],
        "overnight_model_news": [{"symbols": ["NVDA"], "headline": "NVDA news"}],
        "overnight_forward_feeds": [{"symbols": ["SPY"], "headline": "Fed minutes"}],
        "overnight_sec_filings": [{"symbol": "MSFT", "form_type": "8-K", "items": ["4.02"]}],
    }

    ready_embed = build_morning_discord_embed(
        {"status": "READY", "session_date_et": "2026-11-23", "checks": [{}, {}], "issues": []},
        morning_plan,
    )
    assert ready_embed["color"] == 0x22C55E
    assert ready_embed["title"] == "AGY Morning Readiness & Plan (2026-11-23)"
    assert [f["name"] for f in ready_embed["fields"]] == [
        "Readiness Status & Issues",
        "Portfolio Equity / Positions / Stops",
        "Macro Regime Scale & Dials",
        "Today's Earnings / Item-4.02 Blackouts",
        "Overnight News / SEC / RSS Highlights",
    ]
    assert "Earnings Blackouts: AAPL (2026-11-23)" in ready_embed["fields"][3]["value"]

    # Guard explicitly disabled (earnings_blackout_days: None) must say Disabled, NOT None
    disabled_embed = build_morning_discord_embed(
        {"status": "READY", "session_date_et": "2026-11-23", "checks": [{}, {}], "issues": []},
        {**morning_plan, "earnings_blackout_days": None, "earnings_blackouts": []},
    )
    assert "Earnings Blackouts: Disabled" in disabled_embed["fields"][3]["value"]

    # Guard active (earnings_blackout_days: 2) with zero matches must say None
    active_empty_embed = build_morning_discord_embed(
        {"status": "READY", "session_date_et": "2026-11-23", "checks": [{}, {}], "issues": []},
        {**morning_plan, "earnings_blackout_days": 2, "earnings_blackouts": []},
    )
    assert "Earnings Blackouts: None" in active_empty_embed["fields"][3]["value"]

    degraded_embed = build_morning_discord_embed(
        {
            "status": "DEGRADED",
            "session_date_et": "2026-11-23",
            "issues": [{"check": "fred_macro", "severity": "degraded", "message": "stale"}],
        },
        morning_plan,
    )
    assert degraded_embed["color"] == 0xF59E0B
    assert degraded_embed["title"] == "[ALERT] AGY Morning Readiness & Plan (2026-11-23)"

    critical_embed = build_morning_discord_embed(
        {
            "status": "CRITICAL",
            "session_date_et": "2026-11-23",
            "issues": [{"check": "protective_stops", "severity": "critical", "message": "missing"}],
        },
        morning_plan,
    )
    assert critical_embed["color"] == 0xEF4444
    assert critical_embed["title"] == "[ALERT] AGY Morning Readiness & Plan (2026-11-23)"

    pos_close = build_daily_summary_discord_embed(
        {
            "session_date_et": "2026-11-23",
            "equity": 101_500.0,
            "daily_return_pct": 1.25,
            "spy_daily_return_pct": -0.47,
            "ew50_daily_return_pct": -0.79,
            "spmo_daily_return_pct": 0.35,
            "qmom_daily_return_pct": -0.22,
            "cumulative_return_pct": 1.5,
            "spy_cumulative_return_pct": 0.8,
            "ew50_cumulative_return_pct": 1.1,
            "spmo_cumulative_return_pct": 2.4,
            "qmom_cumulative_return_pct": 0.65,
            "drawdown_pct": -0.5,
            "max_drawdown_halt_pct": -8.0,
            "drawdown_headroom_pct": 7.5,
            "todays_fills": [
                {
                    "symbol": "AAPL",
                    "side": "buy",
                    "filled_qty": 40,
                    "filled_avg_price": 200.1,
                    "realized_slippage_bps": 5.0,
                }
            ],
            "mean_realized_slippage_bps": 5.0,
            "mean_backtest_budget_bps": 10.0,
            "positions": [
                {
                    "symbol": "AAPL",
                    "qty": 40,
                    "market_price": 200.0,
                    "stop_price": 185.0,
                    "distance_to_stop_pct": 7.5,
                }
            ],
            "notable_news": [],
            "notable_forward_feeds": [],
            "notable_sec_filings": [],
        },
        image_url="attachment://cumulative_performance.png",
    )
    assert pos_close["color"] == 0x22C55E
    assert pos_close["title"] == "AGY Daily Close Summary (2026-11-23)"
    assert "(SPY `-0.47%`)" in pos_close["description"]
    assert pos_close["image"] == {"url": "attachment://cumulative_performance.png"}
    assert [f["name"] for f in pos_close["fields"]] == [
        "Daily & Cumulative P&L vs SPY, EW-50, SPMO & QMOM",
        "Drawdown Headroom",
        "Notable News / SEC / RSS Events",
    ]
    assert all(
        "Open Positions" not in f["name"] and "Slippage" not in f["name"]
        for f in pos_close["fields"]
    )
    pnl_val = pos_close["fields"][0]["value"]
    assert (
        "**Daily:** `+1.25%` (`SPY`: `-0.47%` | `EW-50`: `-0.79%` | `SPMO`: `+0.35%` | `QMOM`: `-0.22%`)"
        in pnl_val
    )
    assert (
        "**Cumulative:** `+1.50%` (`SPY`: `+0.80%` | `EW-50`: `+1.10%` | `SPMO`: `+2.40%` | `QMOM`: `+0.65%` since 1st trade)"
        in pnl_val
    )

    neg_close = build_daily_summary_discord_embed(
        {
            "session_date_et": "2026-11-23",
            "equity": 99_000.0,
            "daily_return_pct": -1.0,
            "cumulative_return_pct": -1.0,
        }
    )
    assert neg_close["color"] == 0xEF4444
    assert "image" not in neg_close
    assert "(SPY" not in neg_close["description"]
    neg_pnl_val = neg_close["fields"][0]["value"]
    assert "**Daily:** `-1.00%` (`SPY`: `N/A` | `EW-50`: `N/A` | `SPMO`: `N/A` | `QMOM`: `N/A`)" in neg_pnl_val
    assert (
        "**Cumulative:** `-1.00%` (`SPY`: `N/A` | `EW-50`: `N/A` | `SPMO`: `N/A` | `QMOM`: `N/A` since 1st trade)"
        in neg_pnl_val
    )

    test_embed = build_test_discord_embed()
    assert test_embed["footer"] == "agy-trader"
    assert "Discord alerts" in test_embed["description"]


@respx.mock
def test_cli_notify_morning_close_test_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NLP_BACKEND", "lexicon")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "")

    data_dir = tmp_path / "data"
    report_dir = tmp_path / "reports" / "daily"
    now = datetime(2025, 12, 31, 20, 45, tzinfo=UTC)
    _seed_ops_data(data_dir, now_utc=now)

    # 1. Missing DISCORD_WEBHOOK_URL fails cleanly when not --dry-run
    for subcmd in ("morning", "close", "test"):
        args = ["notify", subcmd]
        if subcmd != "test":
            args.extend(
                [
                    "--data-dir",
                    str(data_dir),
                    "--config-dir",
                    str(REPO_CONFIG_DIR),
                    "--report-dir",
                    str(report_dir),
                ]
            )
        res_missing = runner.invoke(ops_cli_app, args)
        assert res_missing.exit_code != 0
        assert "DISCORD_WEBHOOK_URL" in res_missing.output

    # 2. --dry-run works without DISCORD_WEBHOOK_URL and archives markdown reports
    res_morning_dry = runner.invoke(
        ops_cli_app,
        [
            "notify",
            "morning",
            "--dry-run",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--report-dir",
            str(report_dir),
            "--json",
        ],
    )
    assert res_morning_dry.exit_code == 0, res_morning_dry.output
    morning_dry_json = json.loads(res_morning_dry.stdout)
    assert morning_dry_json["dry_run"] is True
    assert morning_dry_json["sent"] is False
    assert (report_dir / "morning_plan_latest.md").exists()

    res_close_dry = runner.invoke(
        ops_cli_app,
        [
            "notify",
            "close",
            "--dry-run",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--report-dir",
            str(report_dir),
            "--json",
        ],
    )
    assert res_close_dry.exit_code == 0, res_close_dry.output
    close_dry_json = json.loads(res_close_dry.stdout)
    assert close_dry_json["dry_run"] is True
    assert close_dry_json["sent"] is False
    assert (report_dir / "daily_summary_latest.md").exists()

    res_test_dry = runner.invoke(ops_cli_app, ["notify", "test", "--dry-run", "--json"])
    assert res_test_dry.exit_code == 0, res_test_dry.output
    assert json.loads(res_test_dry.stdout)["sent"] is False

    # 3. Live mocked HTTP POST with DISCORD_WEBHOOK_URL set
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", SAMPLE_WEBHOOK)
    route = respx.post(SAMPLE_WEBHOOK).mock(return_value=httpx.Response(204))

    res_morning_live = runner.invoke(
        ops_cli_app,
        [
            "notify",
            "morning",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--report-dir",
            str(report_dir),
            "--json",
        ],
    )
    assert res_morning_live.exit_code == 0, res_morning_live.output
    assert SAMPLE_TOKEN not in res_morning_live.output
    morning_live_json = json.loads(res_morning_live.stdout)
    assert morning_live_json["sent"] is True
    assert morning_live_json["webhook_url"] == "https://discord.com/api/webhooks/123456789012345678/***"

    res_close_live = runner.invoke(
        ops_cli_app,
        [
            "notify",
            "close",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--report-dir",
            str(report_dir),
            "--json",
        ],
    )
    assert res_close_live.exit_code == 0, res_close_live.output
    assert SAMPLE_TOKEN not in res_close_live.output
    assert json.loads(res_close_live.stdout)["sent"] is True

    res_test_live = runner.invoke(ops_cli_app, ["notify", "test", "--json"])
    assert res_test_live.exit_code == 0, res_test_live.output
    assert SAMPLE_TOKEN not in res_test_live.output
    assert json.loads(res_test_live.stdout)["sent"] is True
    assert route.call_count == 3

    # 4. HTTP error from Discord exits non-zero and redacts token
    respx.post(SAMPLE_WEBHOOK).mock(return_value=httpx.Response(400))
    res_fail = runner.invoke(ops_cli_app, ["notify", "test"])
    assert res_fail.exit_code != 0
    assert SAMPLE_TOKEN not in res_fail.output
    assert "/webhooks/123456789012345678/***" in res_fail.output


@respx.mock
def test_notifier_multipart_png_attachment_and_close_cli_chart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zoneinfo import ZoneInfo

    ny_tz = ZoneInfo("America/New_York")
    monkeypatch.setenv("NLP_BACKEND", "lexicon")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "")

    # 1. Empty and single-snapshot DBs return None
    empty_dir = tmp_path / "empty"
    assert render_cumulative_performance_png(empty_dir) is None

    single_dir = tmp_path / "single"
    single_db = StateDB(single_dir / "state.db")
    single_db.insert_portfolio_snapshot(
        timestamp=datetime(2025, 12, 30, 15, 40, tzinfo=ny_tz).astimezone(UTC),
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
    assert render_cumulative_performance_png(single_dir) is None

    # 2. Seeded operational DB renders a valid PNG chart
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "reports" / "daily"
    now = datetime(2025, 12, 31, 20, 45, tzinfo=UTC)
    db = _seed_ops_data(data_dir, now_utc=now)
    # Also add an after-hours snapshot to exercise RTH benchmark forward-fill
    db.insert_portfolio_snapshot(
        timestamp=datetime(2025, 12, 31, 17, 0, tzinfo=ny_tz).astimezone(UTC),
        equity=101_600.0,
        cash=39_000.0,
        buying_power=139_000.0,
        gross_exposure=0.61,
        net_exposure=0.61,
        daily_pnl_pct=0.016,
        drawdown_pct=-0.004,
        high_water_mark=102_000.0,
        spy_close=520.0,
        ew50_index=115.0,
    )

    png_bytes = render_cumulative_performance_png(data_dir)
    assert png_bytes is not None
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png_bytes) > 1000

    # 3. notify close --dry-run archives cumulative_performance_latest.png & preserves full markdown
    res_dry = runner.invoke(
        ops_cli_app,
        [
            "notify",
            "close",
            "--dry-run",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--report-dir",
            str(report_dir),
            "--json",
        ],
    )
    assert res_dry.exit_code == 0, res_dry.output
    dry_json = json.loads(res_dry.stdout)
    expected_chart_path = report_dir / "cumulative_performance_latest.png"
    assert dry_json["chart_path"] == str(expected_chart_path)
    assert expected_chart_path.exists()
    assert expected_chart_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert dry_json["embed"]["image"] == {"url": "attachment://cumulative_performance.png"}
    assert [f["name"] for f in dry_json["embed"]["fields"]] == [
        "Daily & Cumulative P&L vs SPY, EW-50, SPMO & QMOM",
        "Drawdown Headroom",
        "Notable News / SEC / RSS Events",
    ]
    md_text = (report_dir / "daily_summary_latest.md").read_text(encoding="utf-8")
    assert "15:40 ET Fills" in md_text
    assert "Open Positions" in md_text

    # 4. Live mocked notify close sends multipart/form-data with payload_json + image/png
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", SAMPLE_WEBHOOK)
    route = respx.post(SAMPLE_WEBHOOK).mock(return_value=httpx.Response(204))
    res_live = runner.invoke(
        ops_cli_app,
        [
            "notify",
            "close",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--report-dir",
            str(report_dir),
            "--json",
        ],
    )
    assert res_live.exit_code == 0, res_live.output
    assert SAMPLE_TOKEN not in res_live.output
    live_json = json.loads(res_live.stdout)
    assert live_json["sent"] is True
    assert route.call_count == 1

    req = route.calls[0].request
    content_type = req.headers.get("content-type", "")
    assert "multipart/form-data" in content_type
    body_bytes = req.content
    assert b'name="payload_json"' in body_bytes
    assert b'name="files[0]"; filename="cumulative_performance.png"' in body_bytes
    assert b"Content-Type: image/png" in body_bytes
    assert b"\x89PNG\r\n\x1a\n" in body_bytes
    assert b'"attachments": [{"id": 0, "filename": "cumulative_performance.png"}]' in body_bytes
    assert b'"url": "attachment://cumulative_performance.png"' in body_bytes
