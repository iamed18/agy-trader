from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agy_trader.cli.main import app

runner = CliRunner()
REPO_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_backfill_synthetic_and_status_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"

    res_backfill = runner.invoke(
        app,
        [
            "backfill",
            "--synthetic",
            "--days",
            "140",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_backfill.exit_code == 0, res_backfill.output
    backfill_payload = json.loads(res_backfill.stdout)
    assert backfill_payload["mode"] == "synthetic"
    assert backfill_payload["days"] == 140
    assert backfill_payload["symbols"] > 0
    assert backfill_payload["bars_saved"] > 0
    assert backfill_payload["news_saved"] > 0
    assert backfill_payload["news_scored"] == backfill_payload["news_saved"]

    res_status = runner.invoke(
        app,
        [
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "status",
            "--json",
        ],
    )
    assert res_status.exit_code == 0, res_status.output
    status_payload = json.loads(res_status.stdout)
    assert status_payload["parquet_symbols_count"] == backfill_payload["symbols"]
    assert status_payload["counts"]["news_articles"] == backfill_payload["news_saved"]
    assert status_payload["counts"]["scored_news"] == backfill_payload["news_saved"]
    assert status_payload["llm_budget"]["allowed"] is True


def test_backtest_ablate_json_and_report(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    report_dir = tmp_path / "reports"

    res_backfill = runner.invoke(
        app,
        [
            "backfill",
            "--synthetic",
            "--days",
            "140",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_backfill.exit_code == 0, res_backfill.output

    res_bt = runner.invoke(
        app,
        [
            "backtest",
            "--ablate",
            "--report-dir",
            str(report_dir),
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_bt.exit_code == 0, res_bt.output
    bt_payload = json.loads(res_bt.stdout)
    assert len(bt_payload["results"]) == 4
    report_path = Path(bt_payload["report_path"])
    assert report_path.exists()
    assert report_path.parent == report_dir
    content = report_path.read_text(encoding="utf-8")
    assert "Strategy Variant Comparison" in content


def test_ops_controls_and_validation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"

    res_set = runner.invoke(
        app,
        [
            "ops",
            "set-exposure",
            "0.5",
            "--reason",
            "vol spike",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_set.exit_code == 0, res_set.output
    assert json.loads(res_set.stdout)["exposure_multiplier"] == 0.5

    res_bad = runner.invoke(
        app,
        [
            "ops",
            "set-exposure",
            "1.5",
            "--reason",
            "bad",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
        ],
    )
    assert res_bad.exit_code != 0

    res_pause = runner.invoke(
        app,
        [
            "ops",
            "pause",
            "--reason",
            "risk event",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_pause.exit_code == 0, res_pause.output
    assert json.loads(res_pause.stdout)["pause_new_entries"] is True

    res_resume = runner.invoke(
        app,
        [
            "ops",
            "resume",
            "--reason",
            "clear",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_resume.exit_code == 0, res_resume.output
    assert json.loads(res_resume.stdout)["pause_new_entries"] is False

    res_bl_add = runner.invoke(
        app,
        [
            "ops",
            "blacklist",
            "add",
            "tsla",
            "--hours",
            "48",
            "--reason",
            "earnings vol",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_bl_add.exit_code == 0, res_bl_add.output
    assert "TSLA" in json.loads(res_bl_add.stdout)["blacklisted_symbols"]

    res_bl_rm = runner.invoke(
        app,
        [
            "ops",
            "blacklist",
            "remove",
            "TSLA",
            "--reason",
            "resolved",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_bl_rm.exit_code == 0, res_bl_rm.output
    assert "TSLA" not in json.loads(res_bl_rm.stdout)["blacklisted_symbols"]

    res_short = runner.invoke(
        app,
        [
            "ops",
            "set-shorting",
            "false",
            "--reason",
            "short squeeze risk",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_short.exit_code == 0, res_short.output
    assert json.loads(res_short.stdout)["allow_shorting"] is False

    res_show = runner.invoke(
        app,
        [
            "ops",
            "show",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_show.exit_code == 0, res_show.output
    assert json.loads(res_show.stdout)["exposure_multiplier"] == 0.5

    res_audit = runner.invoke(
        app,
        [
            "ops",
            "audit-log",
            "--limit",
            "10",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_audit.exit_code == 0, res_audit.output
    assert len(json.loads(res_audit.stdout)["entries"]) == 6


def test_ops_authorize_restart_rate_limit(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"

    for attempt in (1, 2):
        res = runner.invoke(
            app,
            [
                "ops",
                "authorize-restart",
                "trader",
                "--reason",
                f"restart attempt {attempt}",
                "--data-dir",
                str(data_dir),
                "--config-dir",
                str(REPO_CONFIG_DIR),
                "--json",
            ],
        )
        assert res.exit_code == 0, res.output
        assert json.loads(res.stdout)["authorized"] is True

    res_third = runner.invoke(
        app,
        [
            "ops",
            "authorize-restart",
            "trader",
            "--reason",
            "restart attempt 3",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_third.exit_code != 0

    res_disallowed = runner.invoke(
        app,
        [
            "ops",
            "authorize-restart",
            "unknown-svc",
            "--reason",
            "bad service",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
        ],
    )
    assert res_disallowed.exit_code != 0


def test_cli_score_news_walk_forward_empty_data_and_rich_tables(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from agy_trader.core.models import NewsArticle
    from agy_trader.storage.db import StateDB

    data_dir = tmp_path / "data"
    report_dir = tmp_path / "reports"

    # 1. Empty-data backtest fails with exit code 1
    res_empty = runner.invoke(
        app,
        [
            "backtest",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
        ],
    )
    assert res_empty.exit_code == 1

    # 2. score-news --json scores newly inserted unscored articles
    db = StateDB(data_dir / "state.db")
    db.upsert_news_articles(
        [
            NewsArticle(
                article_id="cli-news-1",
                published_at=datetime(2025, 5, 1, 14, 0, tzinfo=UTC),
                headline="Apple beats estimates and raises guidance",
                symbols=["AAPL"],
            )
        ]
    )
    res_score = runner.invoke(
        app,
        [
            "score-news",
            "--backend",
            "lexicon",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_score.exit_code == 0, res_score.output
    score_payload = json.loads(res_score.stdout)
    assert score_payload["scored_count"] == 1
    assert score_payload["scorer_model"] == "lexicon-v1"

    # 3. Synthetic backfill (non-JSON Rich table) + walk-forward --folds 2 --start --end
    res_bf_rich = runner.invoke(
        app,
        [
            "backfill",
            "--synthetic",
            "--days",
            "160",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
        ],
    )
    assert res_bf_rich.exit_code == 0, res_bf_rich.output
    assert "Backfill Complete" in res_bf_rich.stdout

    # Load actual date range from Parquet to pick valid --start and --end window
    from agy_trader.storage.parquet_store import ParquetBarStore

    spy_df = ParquetBarStore(data_dir / "parquet").load_bars("SPY", timeframe="1d")
    start_str = spy_df["timestamp"].iloc[70].date().isoformat()
    end_str = spy_df["timestamp"].iloc[-1].date().isoformat()

    res_wf_json = runner.invoke(
        app,
        [
            "backtest",
            "--walk-forward",
            "--folds",
            "2",
            "--start",
            start_str,
            "--end",
            end_str,
            "--report-dir",
            str(report_dir),
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_wf_json.exit_code == 0, res_wf_json.output
    wf_payload = json.loads(res_wf_json.stdout)
    assert len(wf_payload["results"]) == 1
    assert len(wf_payload["walk_forward_folds"]) == 2
    assert wf_payload["walk_forward_folds"][0]["end_date"] < wf_payload["walk_forward_folds"][1]["start_date"]

    # 4. Non-JSON Rich table output for backtest, status, and ops show
    res_bt_rich = runner.invoke(
        app,
        [
            "backtest",
            "--walk-forward",
            "--folds",
            "2",
            "--start",
            start_str,
            "--end",
            end_str,
            "--report-dir",
            str(report_dir),
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
        ],
    )
    assert res_bt_rich.exit_code == 0, res_bt_rich.output
    assert "Backtest Results" in res_bt_rich.stdout
    assert "Walk-Forward Out-of-Sample Folds" in res_bt_rich.stdout

    res_status_rich = runner.invoke(
        app,
        [
            "status",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
        ],
    )
    assert res_status_rich.exit_code == 0, res_status_rich.output
    assert "agy-trader Status" in res_status_rich.stdout
    assert "Recent Backtest Runs" in res_status_rich.stdout

    res_ops_rich = runner.invoke(
        app,
        [
            "ops",
            "show",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
        ],
    )
    assert res_ops_rich.exit_code == 0, res_ops_rich.output
    assert "Active Ops Control Dials" in res_ops_rich.stdout


def test_cli_honors_nlp_backend_env_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime

    from agy_trader.core.models import NewsArticle
    from agy_trader.nlp.sentiment import SentimentPrediction
    from agy_trader.storage.db import StateDB

    data_dir = tmp_path / "data"
    db = StateDB(data_dir / "state.db")
    db.upsert_news_articles(
        [
            NewsArticle(
                article_id="env-news-1",
                published_at=datetime(2025, 5, 1, 14, 0, tzinfo=UTC),
                headline="NVDA beats guidance",
                symbols=["NVDA"],
            )
        ]
    )

    seen_backends: list[str] = []

    class _StubScorer:
        def __init__(self, backend: str) -> None:
            self.model_name = f"{backend}-stub"

        def score_articles(self, articles: list[NewsArticle]) -> list[SentimentPrediction]:
            return [
                SentimentPrediction(
                    score=0.5,
                    label="positive",
                    confidence=0.9,
                    scorer_model=self.model_name,
                )
                for _ in articles
            ]

    def _fake_create_scorer(backend: str) -> _StubScorer:
        seen_backends.append(backend)
        return _StubScorer(backend)

    monkeypatch.setattr("agy_trader.cli.main.create_sentiment_scorer", _fake_create_scorer)
    monkeypatch.setenv("NLP_BACKEND", "finbert")

    # 1. score-news without --backend picks up NLP_BACKEND=finbert
    res_score = runner.invoke(
        app,
        [
            "score-news",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_score.exit_code == 0, res_score.output
    payload_score = json.loads(res_score.stdout)
    assert payload_score["backend"] == "finbert"
    assert payload_score["scorer_model"] == "finbert-stub"

    # 2. backfill --synthetic without --score-backend still defaults to lexicon
    res_syn = runner.invoke(
        app,
        [
            "backfill",
            "--synthetic",
            "--days",
            "80",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_syn.exit_code == 0, res_syn.output
    payload_syn = json.loads(res_syn.stdout)
    assert payload_syn["score_backend"] == "lexicon-stub"
    assert seen_backends == ["finbert", "lexicon"]


def test_cli_backtest_cpcv_ntrials_execution_timing_and_source_filtering(
    tmp_path: Path,
) -> None:
    from datetime import UTC

    from agy_trader.core.models import NewsArticle
    from agy_trader.storage.db import StateDB
    from agy_trader.storage.parquet_store import ParquetBarStore

    data_dir = tmp_path / "data"
    report_dir = tmp_path / "reports"

    res_bf = runner.invoke(
        app,
        [
            "backfill",
            "--synthetic",
            "--days",
            "400",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_bf.exit_code == 0, res_bf.output

    # Inject a forward-only RSS article (should be excluded by model_news_sources)
    db = StateDB(data_dir / "state.db")
    spy_df = ParquetBarStore(data_dir / "parquet").load_bars("SPY", timeframe="1d")
    last_ts = spy_df["timestamp"].iloc[-2].to_pydatetime()
    db.upsert_news_articles(
        [
            NewsArticle(
                article_id="rss-cli-1",
                published_at=last_ts.astimezone(UTC),
                headline="Forward-only RSS shock",
                source="rss:fed_fomc",
                symbols=["AAPL"],
                sentiment_score=-1.0,
                sentiment_label="negative",
                confidence=1.0,
                scorer_model="lexicon-v1",
            )
        ]
    )

    res_cpcv = runner.invoke(
        app,
        [
            "backtest",
            "--cpcv",
            "--n-trials",
            "10",
            "--execution-timing",
            "next_open",
            "--report-dir",
            str(report_dir),
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_cpcv.exit_code == 0, res_cpcv.output
    payload = json.loads(res_cpcv.stdout)
    assert payload["results"][0]["n_trials_assumed"] == 10
    assert payload["results"][0]["execution_timing"] == "next_open"
    assert payload["cpcv_summary"] is not None
    assert payload["cpcv_summary"]["n_paths"] == 15
    report_text = Path(payload["report_path"]).read_text(encoding="utf-8")
    assert "Equal-Weight Universe" in report_text
    assert "Combinatorial Purged Cross-Validation (CPCV)" in report_text

    # Non-JSON Rich output with --cpcv
    res_rich = runner.invoke(
        app,
        [
            "backtest",
            "--cpcv",
            "--n-trials",
            "5",
            "--execution-timing",
            "same_close",
            "--report-dir",
            str(report_dir),
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
        ],
    )
    assert res_rich.exit_code == 0, res_rich.output
    assert "CPCV Summary" in res_rich.stdout

    # Invalid execution timing fails cleanly
    res_bad_timing = runner.invoke(
        app,
        [
            "backtest",
            "--execution-timing",
            "bad_mode",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
        ],
    )
    assert res_bad_timing.exit_code == 1


def test_cli_collect_feeds_and_backfill_source_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime

    from agy_trader.core.models import NewsArticle, SecFilingEvent
    from agy_trader.storage.db import StateDB

    data_dir = tmp_path / "data"
    seen_yf_symbols: list[list[str]] = []

    def _fake_rss() -> list[NewsArticle]:
        return [
            NewsArticle(
                article_id="rss:fed_fomc:test-1",
                published_at=datetime(2025, 5, 1, 18, 0, tzinfo=UTC),
                headline="FOMC holds benchmark rate steady",
                source="rss:fed_fomc",
                symbols=["SPY", "TLT"],
            )
        ]

    def _fake_yf(symbols: list[str], max_per_symbol: int = 10) -> list[NewsArticle]:
        seen_yf_symbols.append(list(symbols))
        return [
            NewsArticle(
                article_id="yf:spy-1",
                published_at=datetime(2025, 5, 1, 19, 0, tzinfo=UTC),
                headline="SPY weekly ETF flow report",
                source="yfinance_news",
                symbols=["SPY"],
            )
        ]

    monkeypatch.setattr("agy_trader.cli.main.collect_official_rss_feeds", _fake_rss)
    monkeypatch.setattr("agy_trader.cli.main.collect_yfinance_news", _fake_yf)

    # Default collect-feeds targets only ETFs (15 in universe.yaml) and scores when --score is passed
    res_collect = runner.invoke(
        app,
        [
            "collect-feeds",
            "--score",
            "--score-backend",
            "lexicon",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_collect.exit_code == 0, res_collect.output
    payload = json.loads(res_collect.stdout)
    assert payload["rss_saved"] == 1
    assert payload["yfinance_saved"] == 1
    assert payload["yfinance_symbols"] == 15
    assert payload["news_scored"] == 2
    assert len(seen_yf_symbols[0]) == 15
    assert "AAPL" not in seen_yf_symbols[0]

    # --all-symbols collects for all 50 universe symbols
    res_all = runner.invoke(
        app,
        [
            "collect-feeds",
            "--skip-rss",
            "--all-symbols",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
        ],
    )
    assert res_all.exit_code == 0, res_all.output
    assert "Ops Intelligence Feeds" in res_all.stdout
    assert "yfinance_symbols" in res_all.stdout
    assert len(seen_yf_symbols[1]) == 50

    # Verify DB holds sec_filings + forward-only feeds, and invalid --news-sources fails cleanly
    db = StateDB(data_dir / "state.db")
    db.upsert_sec_filings(
        [
            SecFilingEvent(
                accession_number="cli-sec-1",
                symbol="AAPL",
                cik="0000320193",
                form_type="8-K",
                accepted_at=datetime(2025, 5, 1, 20, 0, tzinfo=UTC),
                items=["2.02"],
            )
        ]
    )
    assert db.get_summary_counts()["sec_filings"] == 1

    res_bad_src = runner.invoke(
        app,
        [
            "backfill",
            "--news-sources",
            "alpaca,unknown_feed",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
        ],
    )
    assert res_bad_src.exit_code == 1

    # Passing a forward-only source to `backtest --sources` must fail cleanly
    res_bad_bt_src = runner.invoke(
        app,
        [
            "backtest",
            "--sources",
            "benzinga,rss:fed_fomc",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res_bad_bt_src.exit_code == 1
    assert "Forward-only sources" in res_bad_bt_src.stdout



