from __future__ import annotations

import logging
import plistlib
import shutil
import sys
import threading
import types
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from alpaca.trading.enums import QueryOrderStatus

from agy_ops.daemons import (
    MarketDataDaemon,
    NewsNlpDaemon,
    TraderDaemon,
    configure_daemon_logging,
    get_process_rss_mb,
    is_us_market_open_fallback,
    wait_for_network,
)
from agy_ops.deploy.service_manifests import (
    DEFAULT_SERVICE_SPECS,
    render_launchd_plist,
    render_systemd_unit,
    write_service_manifests,
)
from agy_trader.core.config import EnvSettings, load_universe
from agy_trader.core.models import (
    EarningsEvent,
    MacroPoint,
    NewsArticle,
    OrderSide,
    ProposedOrder,
    RiskVerdict,
    SecFilingEvent,
    ensure_utc,
)
from agy_trader.data.providers import (
    EarningsProvider,
    EtfHoldingsProvider,
    MacroProvider,
    MarketDataProvider,
    NewsProvider,
    SecFilingsProvider,
)
from agy_trader.data.synthetic import generate_synthetic_dataset
from agy_trader.execution.alpaca_broker import AlpacaBroker, build_client_order_id
from agy_trader.nlp.sentiment import LexiconSentimentScorer
from agy_trader.storage.db import StateDB
from agy_trader.storage.parquet_store import ParquetBarStore

REPO_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@dataclass
class _FakeAccount:
    equity: str = "100000.00"
    cash: str = "50000.00"
    buying_power: str = "150000.00"
    last_equity: str = "100000.00"


@dataclass
class _FakePosition:
    symbol: str
    qty: str
    avg_entry_price: str
    current_price: str
    unrealized_pl: str
    side: str = "long"


@dataclass
class _FakeOrder:
    id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    qty: str
    status: str = "new"
    filled_qty: str = "0"
    filled_avg_price: str | None = None
    limit_price: str | None = None
    stop_price: str | None = None


@dataclass
class _FakeQuote:
    bid_price: float
    ask_price: float


class _CrashProcess(BaseException):
    """Simulates SIGKILL/power-loss mid-execution (not caught by except Exception)."""


class _NotFoundError(Exception):
    status_code = 404


class _FakeTradingClient:
    def __init__(self) -> None:
        self.account = _FakeAccount()
        self.positions: list[_FakePosition] = []
        self.orders_by_id: dict[str, _FakeOrder] = {}
        self.orders_by_cid: dict[str, _FakeOrder] = {}
        self.submitted_requests: list[Any] = []
        self.canceled_ids: list[str] = []
        self.closed = False
        self._seq = 0
        self.clock_open: bool | None = None
        self.crash_after_submit_count: int | None = None

        class _Sess:
            def __init__(self, outer: _FakeTradingClient) -> None:
                self.outer = outer

            def close(self) -> None:
                self.outer.closed = True

        self._session = _Sess(self)

    def get_clock(self) -> Any:
        if self.clock_open is None:
            raise RuntimeError("clock unavailable")
        return types.SimpleNamespace(is_open=self.clock_open)

    def get_account(self) -> _FakeAccount:
        return self.account

    def get_all_positions(self) -> list[_FakePosition]:
        return list(self.positions)

    def get_orders(self, filter: Any = None) -> list[_FakeOrder]:
        open_statuses = {
            "new",
            "accepted",
            "partially_filled",
            "pending_new",
            "pending_cancel",
            "pending_replace",
            "held",
            "stopped",
        }
        status_filter = getattr(filter, "status", QueryOrderStatus.OPEN)
        orders = list(self.orders_by_id.values())
        if status_filter == QueryOrderStatus.OPEN:
            return [o for o in orders if o.status in open_statuses]
        return orders

    def get_order_by_client_id(self, client_order_id: str) -> _FakeOrder:
        order = self.orders_by_cid.get(client_order_id)
        if order is None:
            raise _NotFoundError(client_order_id)
        return order

    def get_order_by_id(self, order_id: str) -> _FakeOrder:
        order = self.orders_by_id.get(order_id)
        if order is None:
            raise _NotFoundError(order_id)
        return order

    def submit_order(self, req: Any) -> _FakeOrder:
        self.submitted_requests.append(req)
        self._seq += 1
        oid = f"brk-{self._seq}"
        cid = str(getattr(req, "client_order_id", oid))
        sym = str(getattr(req, "symbol", "")).upper()
        side = str(getattr(getattr(req, "side", ""), "value", getattr(req, "side", ""))).lower()
        qty = float(getattr(req, "qty", 0.0))
        limit_px = getattr(req, "limit_price", None)
        stop_px = getattr(req, "stop_price", None)

        if stop_px is not None:
            order = _FakeOrder(
                id=oid,
                client_order_id=cid,
                symbol=sym,
                side=side,
                order_type="stop",
                qty=str(qty),
                status="new",
                stop_price=str(stop_px),
            )
        else:
            fill_px = float(limit_px) if limit_px is not None else 100.0
            order = _FakeOrder(
                id=oid,
                client_order_id=cid,
                symbol=sym,
                side=side,
                order_type="limit" if limit_px is not None else "market",
                qty=str(qty),
                status="filled",
                filled_qty=str(qty),
                filled_avg_price=str(fill_px),
                limit_price=str(limit_px) if limit_px is not None else None,
            )

        self.orders_by_id[oid] = order
        self.orders_by_cid[cid] = order
        if (
            self.crash_after_submit_count is not None
            and len(self.submitted_requests) >= self.crash_after_submit_count
        ):
            self.crash_after_submit_count = None
            raise _CrashProcess("Simulated mid-rebalance crash")
        return order

    def cancel_order_by_id(self, order_id: str) -> None:
        self.canceled_ids.append(order_id)
        if order_id in self.orders_by_id:
            self.orders_by_id[order_id].status = "canceled"


class _FakeQuoteClient:
    def __init__(self, prices: dict[str, float] | None = None) -> None:
        self.prices = prices or {}

    def get_stock_latest_quote(self, req: Any) -> dict[str, _FakeQuote]:
        syms = req.symbol_or_symbols
        if isinstance(syms, str):
            syms = [syms]
        out: dict[str, _FakeQuote] = {}
        for s in syms:
            px = self.prices.get(s.upper(), 100.0)
            out[s.upper()] = _FakeQuote(bid_price=px * 0.999, ask_price=px * 1.001)
        return out


class _StubMarketProvider(MarketDataProvider):
    def __init__(self, bars_map: dict[str, pd.DataFrame]) -> None:
        self.bars_map = bars_map
        self.calls: list[tuple[list[str], datetime, datetime]] = []

    def fetch_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        self.calls.append((list(symbols), start, end))
        return {
            s: self.bars_map[s].copy()
            for s in symbols
            if s in self.bars_map
        }


class _StubMacroProvider(MacroProvider):
    def fetch_macro(self, start_date: date, end_date: date) -> list[MacroPoint]:
        return [
            MacroPoint(as_of_date=end_date, series_id="DGS3MO", value=4.25),
            MacroPoint(as_of_date=end_date, series_id="VIXCLS", value=16.5),
        ]


class _StubEarningsProvider(EarningsProvider):
    def fetch_earnings(self, symbols: list[str]) -> list[EarningsEvent]:
        return [EarningsEvent(symbol=symbols[0], earnings_date=date(2026, 12, 1))] if symbols else []


class _StubSecProvider(SecFilingsProvider):
    def fetch_filings(
        self, symbols: list[str], start: datetime, end: datetime
    ) -> list[SecFilingEvent]:
        if not symbols:
            return []
        return [
            SecFilingEvent(
                accession_number="0001-26-000001",
                symbol=symbols[0],
                cik="0000320193",
                form_type="4",
                accepted_at=end - timedelta(hours=2),
                net_insider_buy_usd=250_000.0,
                insider_buyer_count=1,
            )
        ]


class _StubNewsProvider(NewsProvider):
    def __init__(self, articles: list[NewsArticle]) -> None:
        self.articles = articles
        self.calls = 0

    def fetch_news(self, symbols: list[str], start: datetime, end: datetime) -> list[NewsArticle]:
        self.calls += 1
        return list(self.articles)


class _StubEtfHoldingsProvider(EtfHoldingsProvider):
    def fetch_etf_holdings(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for sym in symbols:
            if sym in {"TLT", "GLD"}:
                out[sym] = {}
            else:
                out[sym] = {"AAPL": 0.08, "MSFT": 0.07}
        return out


def _make_daily_bars(symbol: str, n_bars: int, end_dt_utc: datetime, base_price: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range(end=end_dt_utc.date(), periods=n_bars)
    closes = [base_price + i * 0.25 for i in range(n_bars)]
    return pd.DataFrame(
        {
            "timestamp": [
                datetime(d.year, d.month, d.day, 21, 0, tzinfo=UTC) for d in dates
            ],
            "open": [c * 0.995 for c in closes],
            "high": [c * 1.008 for c in closes],
            "low": [c * 0.992 for c in closes],
            "close": closes,
            "volume": [1_000_000.0] * n_bars,
            "vwap": closes,
        }
    )


def test_common_helpers_rss_market_clock_and_network_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rss = get_process_rss_mb()
    assert isinstance(rss, float)
    assert rss > 0.0

    # Cross-platform RSS scaling check
    fake_rusage = types.SimpleNamespace(ru_maxrss=104857600)  # 100 MB in bytes, or ~102400 MB in KB
    monkeypatch.setattr("agy_ops.daemons.common.resource.getrusage", lambda _who: fake_rusage)
    monkeypatch.setattr("agy_ops.daemons.common.sys.platform", "darwin")
    assert get_process_rss_mb() == 100.0
    monkeypatch.setattr("agy_ops.daemons.common.sys.platform", "linux")
    assert get_process_rss_mb() == round(104857600 / 1024.0, 2)

    # NYSE open/closed checks (EST = UTC-5 in Nov)
    assert is_us_market_open_fallback(datetime(2026, 11, 20, 14, 30, tzinfo=UTC)) is True  # 09:30 ET
    assert is_us_market_open_fallback(datetime(2026, 11, 20, 14, 29, tzinfo=UTC)) is False  # 09:29 ET
    assert is_us_market_open_fallback(datetime(2026, 11, 20, 21, 0, tzinfo=UTC)) is False  # 16:00 ET
    assert is_us_market_open_fallback(datetime(2026, 11, 21, 16, 0, tzinfo=UTC)) is False  # Saturday
    assert is_us_market_open_fallback(datetime(2026, 11, 26, 16, 0, tzinfo=UTC)) is False  # Thanksgiving

    # Exponential backoff network wait: fail 3 times, succeed on 4th
    attempts = 0
    sleeps: list[float] = []

    def flaky_check() -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("DNS unavailable")
        return attempts >= 4

    ok = wait_for_network(
        flaky_check,
        max_wait_seconds=60.0,
        base_delay=2.0,
        sleep_fn=sleeps.append,
    )
    assert ok is True
    assert attempts == 4
    assert sleeps == [2.0, 4.0, 8.0]

    # Timeout when network stays down
    sleeps_fail: list[float] = []
    assert (
        wait_for_network(
            lambda: False,
            max_wait_seconds=10.0,
            base_delay=2.0,
            sleep_fn=sleeps_fail.append,
        )
        is False
    )
    assert sum(sleeps_fail) == pytest.approx(10.0)

    # Stop event aborts immediately
    stop = threading.Event()
    stop.set()
    assert wait_for_network(lambda: False, stop_event=stop, sleep_fn=lambda _: None) is False


def test_daemon_wal_integrity_failure_records_critical(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    db = StateDB(data_dir / "state.db")
    daemon = MarketDataDaemon(
        EnvSettings(agy_data_dir=data_dir, agy_config_dir=REPO_CONFIG_DIR),
        data_dir,
        REPO_CONFIG_DIR,
        db=db,
        network_check_fn=lambda: True,
    )
    daemon.verify_wal_integrity()

    # Simulate WAL failure
    db.check_wal_integrity = lambda: (False, "database disk image is malformed")  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="WAL integrity check failed"):
        daemon.startup(now_utc=datetime(2026, 11, 20, 15, 0, tzinfo=UTC))

    hbs = db.get_daemon_heartbeats()
    assert hbs["market-data"]["status"] == "critical"
    events = db.get_daemon_events(daemon_name="market-data", event_type="error")
    assert len(events) == 1
    assert events[0]["details"]["kind"] == "wal_integrity_failed"


def test_market_data_daemon_boot_gapfill_eod_and_corporate_action(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    universe = load_universe(REPO_CONFIG_DIR)
    symbols = [u.symbol for u in universe]
    boot_dt = datetime(2026, 11, 20, 13, 0, tzinfo=UTC)  # 08:00 ET (pre-market)

    bars_map = {sym: _make_daily_bars(sym, 310, boot_dt - timedelta(days=1)) for sym in symbols}
    market_prov = _StubMarketProvider(bars_map)
    db = StateDB(data_dir / "state.db")
    bar_store = ParquetBarStore(data_dir / "parquet")

    daemon = MarketDataDaemon(
        EnvSettings(agy_data_dir=data_dir, agy_config_dir=REPO_CONFIG_DIR),
        data_dir,
        REPO_CONFIG_DIR,
        db=db,
        bar_store=bar_store,
        market_provider=market_prov,
        macro_provider=_StubMacroProvider(),
        earnings_provider=_StubEarningsProvider(),
        sec_provider=_StubSecProvider(),
        etf_holdings_provider=_StubEtfHoldingsProvider(),
        network_check_fn=lambda: True,
        min_bars_required=300,
    )
    daemon.startup(now_utc=boot_dt)

    # All 50 symbols should now have >= 300 bars
    assert len(bar_store.list_symbols("1d")) == len(symbols)
    assert len(bar_store.load_bars("SPY", "1d")) >= 300
    assert db.get_daemon_events(daemon_name="market-data", event_type="boot")
    assert db.get_latest_macro_points(["DGS3MO"])["DGS3MO"].value == 4.25

    # Weekend closed cycle -> idle_market_closed
    sat_res = daemon.run_once(now_utc=datetime(2026, 11, 21, 16, 0, tzinfo=UTC))
    assert sat_res["status"] == "idle_market_closed"
    assert sat_res["eod_ran"] is False

    # Post-16:05 ET Friday EOD sync + corporate action mismatch handling
    bad_spy = bars_map["SPY"].copy()
    bad_spy["close"] = bad_spy["close"] * 0.5  # > 1e-4 diff triggers CorporateActionMismatchError
    bad_spy["low"] = bad_spy["low"] * 0.5
    bad_spy["open"] = bad_spy["open"] * 0.5
    bad_spy["high"] = bad_spy["high"] * 0.5
    bad_spy["vwap"] = bad_spy["vwap"] * 0.5
    market_prov.bars_map["SPY"] = bad_spy

    eod_dt = datetime(2026, 11, 20, 21, 10, tzinfo=UTC)  # 16:10 ET
    eod_res = daemon.run_once(now_utc=eod_dt)
    assert eod_res["eod_ran"] is True
    assert eod_res["alerts_count"] == 1
    errs = db.get_daemon_events(daemon_name="market-data", event_type="error")
    assert any(e["details"].get("kind") == "corporate_action_mismatch" for e in errs)

    # Calling run_once again same session does not re-run EOD sync
    eod_res_2 = daemon.run_once(now_utc=datetime(2026, 11, 20, 21, 20, tzinfo=UTC))
    assert eod_res_2["eod_ran"] is False


def test_news_nlp_daemon_batches_gc_and_forward_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    db = StateDB(data_dir / "state.db")
    now = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)

    model_articles = [
        NewsArticle(
            article_id=f"alp:{i}",
            published_at=now - timedelta(minutes=i + 1),
            headline=f"AAPL beats estimates on strong demand #{i}",
            source="benzinga",
            symbols=["AAPL"],
        )
        for i in range(3)
    ]
    fh_articles = [
        NewsArticle(
            article_id="fh:1",
            published_at=now - timedelta(minutes=10),
            headline="MSFT raises guidance on cloud growth",
            source="finnhub",
            symbols=["MSFT"],
        )
    ]
    rss_articles = [
        NewsArticle(
            article_id="rss:fed:1",
            published_at=now - timedelta(minutes=15),
            headline="Federal Reserve issues FOMC statement",
            source="rss:fed_fomc",
            symbols=["SPY", "TLT"],
        )
    ]
    yf_articles = [
        NewsArticle(
            article_id="yf:spy:1",
            published_at=now - timedelta(minutes=20),
            headline="SPY ETF inflows accelerate",
            source="yfinance_news",
            symbols=["SPY"],
        )
    ]

    # Mock torch MPS cache to verify release_memory invokes empty_cache when torch is loaded
    mps_calls = 0

    def _empty_mps() -> None:
        nonlocal mps_calls
        mps_calls += 1

    fake_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
        mps=types.SimpleNamespace(empty_cache=_empty_mps),
        cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    daemon = NewsNlpDaemon(
        EnvSettings(agy_data_dir=data_dir, agy_config_dir=REPO_CONFIG_DIR, nlp_backend="lexicon"),
        data_dir,
        REPO_CONFIG_DIR,
        db=db,
        alpaca_news_provider=_StubNewsProvider(model_articles),
        finnhub_news_provider=_StubNewsProvider(fh_articles),
        rss_collector=lambda: rss_articles,
        yfinance_collector=lambda _syms: yf_articles,
        scorer=LexiconSentimentScorer(),
        network_check_fn=lambda: True,
        score_batch_size=2,
    )
    daemon.startup(now_utc=now)
    res = daemon.run_once(now_utc=now)

    assert res["articles_saved"] == 6
    assert res["articles_scored"] == 6
    assert res["unscored_remaining"] == 0
    assert mps_calls >= 3  # 6 articles / batch_size 2 -> 3 batches + end of cycle

    # Forward-only sources remain excluded from modeling queries!
    modeling_news = db.get_all_scored_news(include_forward_only=False)
    assert len(modeling_news) == 4
    assert all(a.source in {"benzinga", "finnhub"} for a in modeling_news)

    hbs = db.get_daemon_heartbeats()
    assert hbs["news-nlp"]["details"]["unscored_remaining"] == 0
    assert hbs["news-nlp"]["details"]["rss_mb"] > 0


def test_trader_daemon_reconcile_stop_repair_crash_idempotency_and_kill_switch(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    db = StateDB(data_dir / "state.db")
    bar_store = ParquetBarStore(data_dir / "parquet")
    universe = load_universe(REPO_CONFIG_DIR)
    generate_synthetic_dataset(universe=universe, bar_store=bar_store, db=db, days=320, seed=42)

    # Seed scored news using lexicon-v1
    from agy_trader.cli.main import _score_unscored_news

    _score_unscored_news(db, LexiconSentimentScorer())

    # Determine latest synthetic bar date so 15:40 ET runs on an NYSE trading day with fresh marks
    spy_df = bar_store.load_bars("SPY", "1d")
    last_bar_date = ensure_utc(spy_df["timestamp"].iloc[-1].to_pydatetime()).date()
    rebal_now = datetime(last_bar_date.year, last_bar_date.month, last_bar_date.day, 20, 45, tzinfo=UTC)

    # Add an existing position with NO resting stop to verify startup auto-repair!
    tc = _FakeTradingClient()
    tc.clock_open = True
    tc.positions = [
        _FakePosition(
            symbol="AAPL",
            qty="10",
            avg_entry_price="180.0",
            current_price="185.0",
            unrealized_pl="50.0",
        )
    ]
    qc = _FakeQuoteClient({u.symbol: float(bar_store.load_bars(u.symbol, "1d")["close"].iloc[-1]) for u in universe})
    settings = EnvSettings(
        alpaca_api_key="paper-key",
        alpaca_secret_key="paper-secret",
        alpaca_paper=True,
        nlp_backend="lexicon",
        agy_data_dir=data_dir,
        agy_config_dir=REPO_CONFIG_DIR,
    )
    broker = AlpacaBroker(
        settings,
        db,
        trading_client=tc,
        quote_client=qc,
        limit_timeout_seconds=0.0,
        poll_interval_seconds=0.0,
        sleep_fn=lambda _: None,
        now_fn=lambda: rebal_now,
    )

    daemon = TraderDaemon(
        settings,
        data_dir,
        REPO_CONFIG_DIR,
        db=db,
        bar_store=bar_store,
        broker=broker,
        network_check_fn=lambda: True,
    )
    daemon.startup(now_utc=rebal_now - timedelta(hours=1))

    # Missing GTC stop for AAPL must have been repaired on startup!
    db_pos = {p["symbol"]: p for p in db.get_live_positions()}
    assert "AAPL" in db_pos
    assert db_pos["AAPL"]["stop_order_id"] is not None
    assert db_pos["AAPL"]["stop_price"] is not None

    # Simulate a crash mid-rebalance after the 1st new order submission!
    # Clear existing submitted requests counter after boot stop
    boot_submits = len(tc.submitted_requests)
    tc.crash_after_submit_count = boot_submits + 1
    with pytest.raises(_CrashProcess, match="Simulated mid-rebalance crash"):
        daemon.run_once(now_utc=rebal_now)

    assert not db.has_session_rebalance_completed(last_bar_date.isoformat(), "15:40")
    submits_after_crash = len(tc.submitted_requests)
    assert submits_after_crash == boot_submits + 1
    crashed_cid = tc.submitted_requests[-1].client_order_id

    # Restart a fresh TraderDaemon and re-run 15:40 ET rebalance -> must adopt crashed order, ZERO duplicate submits!
    broker2 = AlpacaBroker(
        settings,
        db,
        trading_client=tc,
        quote_client=qc,
        limit_timeout_seconds=0.0,
        poll_interval_seconds=0.0,
        sleep_fn=lambda _: None,
        now_fn=lambda: rebal_now,
    )
    daemon2 = TraderDaemon(
        settings,
        data_dir,
        REPO_CONFIG_DIR,
        db=db,
        bar_store=bar_store,
        broker=broker2,
        network_check_fn=lambda: True,
    )
    daemon2.startup(now_utc=rebal_now)
    res1 = daemon2.run_once(now_utc=rebal_now)
    assert res1["rebalance_ran"] is True
    assert db.has_session_rebalance_completed(last_bar_date.isoformat(), "15:40")

    # Verify crashed client_order_id was submitted ONLY ONCE to TradingClient
    submitted_cids = [getattr(r, "client_order_id", None) for r in tc.submitted_requests]
    assert submitted_cids.count(crashed_cid) == 1

    # Running again in same 15:40 ET window must NOT re-execute rebalance
    submits_before_second_tick = len(tc.submitted_requests)
    res2 = daemon2.run_once(now_utc=rebal_now + timedelta(minutes=2))
    assert res2["rebalance_ran"] is False
    assert res2["snapshot_recorded"] is False  # bounded 5-min snapshot cadence!
    assert len(tc.submitted_requests) == submits_before_second_tick

    # Trigger daily-loss kill-switch (-4% <= -3.5% halt) -> auto-pauses ControlPlane!
    tc.account = _FakeAccount(equity="96000.00", last_equity="100000.00")
    res_ks = daemon2.run_once(now_utc=rebal_now + timedelta(minutes=6))
    assert res_ks["paused"] is True
    assert db.get_control_dials().pause_new_entries is True
    assert res_ks["snapshot_recorded"] is True  # >5 min elapsed

    # Clean shutdown closes broker via broker.close()
    close_calls = 0
    orig_close = broker2.close

    def _spy_close() -> None:
        nonlocal close_calls
        close_calls += 1
        orig_close()

    broker2.close = _spy_close  # type: ignore[method-assign]
    daemon2.shutdown()
    assert close_calls == 1
    assert tc.closed is False  # injected client is caller-owned

    # Ensure snapshots have SPY and EW50 baselines
    snaps = db.get_portfolio_snapshots()
    assert len(snaps) >= 2
    assert snaps[-1]["spy_close"] is not None
    assert snaps[-1]["ew50_index"] is not None

    # Verify build_client_order_id deterministic format
    assert build_client_order_id("2026-11-20", "1540", "AAPL", OrderSide.buy, "p2m0").startswith(
        "agy:20261120:1540:AAPL:B:"
    )


def test_service_manifests_launchd_and_systemd(tmp_path: Path) -> None:
    repo_root = Path("/path/to/agy-trader")
    for spec in DEFAULT_SERVICE_SPECS:
        plist_xml = render_launchd_plist(spec, repo_root=repo_root, username="trader")
        parsed = plistlib.loads(plist_xml.encode("utf-8"))
        assert parsed["Label"] == spec.label
        assert parsed["UserName"] == "trader"
        assert parsed["GroupName"] == "staff"
        assert parsed["EnvironmentVariables"]["HOME"] == "/Users/<user>"
        assert parsed["RunAtLoad"] is True
        assert parsed["KeepAlive"] is True
        assert parsed["ThrottleInterval"] == 15
        assert "StandardOutPath" not in parsed
        assert "StandardErrorPath" not in parsed
        assert "daemon" in parsed["ProgramArguments"]

        with_logs = plistlib.loads(
            render_launchd_plist(
                spec,
                repo_root=repo_root,
                username="trader",
                log_dir=Path("/private/var/log/agy-trader"),
            ).encode("utf-8")
        )
        assert with_logs["StandardOutPath"] == f"/private/var/log/agy-trader/{spec.name}.stdout.log"
        assert with_logs["StandardErrorPath"] == f"/private/var/log/agy-trader/{spec.name}.stderr.log"

        unit = render_systemd_unit(spec, repo_root=repo_root, username="trader")
        assert f"Description={spec.description}" in unit
        assert "Restart=always" in unit

    written_mac = write_service_manifests("macos", tmp_path / "mac", repo_root=repo_root)
    written_lin = write_service_manifests("systemd", tmp_path / "lin", repo_root=repo_root)
    assert len(written_mac) == 4
    assert len(written_lin) == 4
    assert all(p.exists() for p in written_mac + written_lin)


def test_c1_per_symbol_staleness_filtering_coverage_and_daily_bar_proxy(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    db = StateDB(data_dir / "state.db")
    bar_store = ParquetBarStore(data_dir / "parquet")
    universe = load_universe(REPO_CONFIG_DIR)
    generate_synthetic_dataset(universe=universe, bar_store=bar_store, db=db, days=320, seed=42)

    from agy_trader.cli.main import _score_unscored_news

    _score_unscored_news(db, LexiconSentimentScorer())

    spy_df = bar_store.load_bars("SPY", "1d")
    last_bar_date = ensure_utc(spy_df["timestamp"].iloc[-1].to_pydatetime()).date()
    rebal_now = datetime(last_bar_date.year, last_bar_date.month, last_bar_date.day, 20, 45, tzinfo=UTC)

    class _SnapClient:
        def __init__(self, snapshots: dict[str, Any]) -> None:
            self.snapshots = snapshots

        def get_stock_snapshot(self, req: Any) -> dict[str, Any]:
            syms = req.symbol_or_symbols
            if isinstance(syms, str):
                syms = [syms]
            return {s: self.snapshots[s] for s in syms if s in self.snapshots}

    base_snapshots: dict[str, Any] = {}
    for u in universe:
        last_close = float(bar_store.load_bars(u.symbol, "1d")["close"].iloc[-1])
        base_snapshots[u.symbol] = types.SimpleNamespace(
            latest_trade=types.SimpleNamespace(price=last_close, timestamp=rebal_now - timedelta(minutes=1)),
            latest_quote=None,
            minute_bar=None,
            daily_bar=types.SimpleNamespace(
                open=last_close * 0.99,
                high=last_close * 1.02,
                low=last_close * 0.98,
                close=last_close,
                volume=500_000.0,
                vwap=last_close * 1.001,
                timestamp=rebal_now - timedelta(minutes=1),
            ),
        )

    # 1) Stale latest_trade on ORCL with fresh latest_quote -> uses freshest timestamp (quote) & passes!
    orcl_px = float(bar_store.load_bars("ORCL", "1d")["close"].iloc[-1])
    base_snapshots["ORCL"] = types.SimpleNamespace(
        latest_trade=types.SimpleNamespace(price=orcl_px * 0.95, timestamp=rebal_now - timedelta(minutes=45)),
        latest_quote=types.SimpleNamespace(
            bid_price=orcl_px * 0.999,
            ask_price=orcl_px * 1.001,
            timestamp=rebal_now - timedelta(seconds=30),
        ),
        minute_bar=None,
        daily_bar=base_snapshots["ORCL"].daily_bar,
    )

    # 2) Truly stale non-benchmark symbol (CRM 40 min old) -> excluded from bars_with_proxy, does NOT poison others!
    crm_px = float(bar_store.load_bars("CRM", "1d")["close"].iloc[-1])
    base_snapshots["CRM"] = types.SimpleNamespace(
        latest_trade=types.SimpleNamespace(price=crm_px, timestamp=rebal_now - timedelta(minutes=40)),
        latest_quote=None,
        minute_bar=None,
        daily_bar=None,
    )

    tc = _FakeTradingClient()
    tc.clock_open = True
    settings = EnvSettings(
        alpaca_api_key="paper-key",
        alpaca_secret_key="paper-secret",
        alpaca_paper=True,
        nlp_backend="lexicon",
        agy_data_dir=data_dir,
        agy_config_dir=REPO_CONFIG_DIR,
    )
    broker = AlpacaBroker(
        settings,
        db,
        trading_client=tc,
        quote_client=_FakeQuoteClient({}),
        limit_timeout_seconds=0.0,
        poll_interval_seconds=0.0,
        sleep_fn=lambda _: None,
        now_fn=lambda: rebal_now,
    )
    snap_client = _SnapClient(base_snapshots)
    daemon = TraderDaemon(
        settings,
        data_dir,
        REPO_CONFIG_DIR,
        db=db,
        bar_store=bar_store,
        broker=broker,
        snapshot_client=snap_client,
        network_check_fn=lambda: True,
    )
    daemon.startup(now_utc=rebal_now - timedelta(hours=1))
    res = daemon.run_once(now_utc=rebal_now)
    assert res["rebalance_ran"] is True
    rebal = res["rebalance"]
    assert rebal["fresh_symbols_count"] == len(universe) - 1  # CRM excluded
    assert rebal["latest_bar_age_minutes"] <= 2.0
    assert rebal["approved_count"] > 0  # Pass-2 entries NOT blocked by 1 stale symbol!

    # Verify M1: real daily_bar OHLCV merged into in-memory proxy bar
    marks, mark_ts, parsed_snaps = daemon._fetch_live_marks(
        [u.symbol for u in universe], broker, {}, rebal_now
    )
    assert marks["ORCL"] == pytest.approx(orcl_px)
    bars_proxy, all_prices, age = daemon._build_bars_with_intraday_proxy(
        symbols=[u.symbol for u in universe],
        universe_symbols={u.symbol for u in universe},
        benchmark_symbol="SPY",
        max_staleness_minutes=15.0,
        live_marks=marks,
        mark_ts=mark_ts,
        snapshots=parsed_snaps,
        now_utc=rebal_now,
    )
    assert "AAPL" in bars_proxy
    assert "CRM" not in bars_proxy
    assert "CRM" in all_prices
    assert age is not None and age <= 2.0
    assert float(bars_proxy["AAPL"]["volume"].iloc[-1]) == 500_000.0

    # 3) Stale SPY benchmark MUST fail closed (latest_bar_age_minutes > 15)
    spy_px = float(bar_store.load_bars("SPY", "1d")["close"].iloc[-1])
    bad_spy_snaps = dict(base_snapshots)
    bad_spy_snaps["SPY"] = types.SimpleNamespace(
        latest_trade=types.SimpleNamespace(price=spy_px, timestamp=rebal_now - timedelta(minutes=45)),
        latest_quote=None,
        minute_bar=None,
        daily_bar=None,
    )
    snap_client.snapshots = bad_spy_snaps
    m_spy, ts_spy, ps_spy = daemon._fetch_live_marks(
        [u.symbol for u in universe], broker, {}, rebal_now
    )
    _, _, spy_stale_age = daemon._build_bars_with_intraday_proxy(
        symbols=[u.symbol for u in universe],
        universe_symbols={u.symbol for u in universe},
        benchmark_symbol="SPY",
        max_staleness_minutes=15.0,
        live_marks=m_spy,
        mark_ts=ts_spy,
        snapshots=ps_spy,
        now_utc=rebal_now,
    )
    assert spy_stale_age is not None and spy_stale_age > 15.0

    # 4) Coverage < 80% (12 stale symbols out of 50 -> 76%) MUST fail closed
    low_cov_snaps = dict(base_snapshots)
    for u in universe[10:22]:
        px = float(bar_store.load_bars(u.symbol, "1d")["close"].iloc[-1])
        low_cov_snaps[u.symbol] = types.SimpleNamespace(
            latest_trade=types.SimpleNamespace(price=px, timestamp=rebal_now - timedelta(minutes=45)),
            latest_quote=None,
            minute_bar=None,
            daily_bar=None,
        )
    snap_client.snapshots = low_cov_snaps
    m_cov, ts_cov, ps_cov = daemon._fetch_live_marks(
        [u.symbol for u in universe], broker, {}, rebal_now
    )
    _, _, cov_age = daemon._build_bars_with_intraday_proxy(
        symbols=[u.symbol for u in universe],
        universe_symbols={u.symbol for u in universe},
        benchmark_symbol="SPY",
        max_staleness_minutes=15.0,
        live_marks=m_cov,
        mark_ts=ts_cov,
        snapshots=ps_cov,
        now_utc=rebal_now,
    )
    assert cov_age is not None and cov_age > 15.0


def test_c2_market_open_gate_and_early_close_half_day(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    db = StateDB(data_dir / "state.db")
    bar_store = ParquetBarStore(data_dir / "parquet")
    universe = load_universe(REPO_CONFIG_DIR)
    generate_synthetic_dataset(universe=universe, bar_store=bar_store, db=db, days=320, seed=42)

    spy_df = bar_store.load_bars("SPY", "1d")
    last_bar_date = ensure_utc(spy_df["timestamp"].iloc[-1].to_pydatetime()).date()
    rebal_now_1540 = datetime(last_bar_date.year, last_bar_date.month, last_bar_date.day, 20, 45, tzinfo=UTC)

    tc = _FakeTradingClient()
    # Early-close half-day closed at 13:00 ET -> at 15:45 ET clock reports is_open=False!
    tc.clock_open = False
    qc = _FakeQuoteClient({u.symbol: float(bar_store.load_bars(u.symbol, "1d")["close"].iloc[-1]) for u in universe})
    settings = EnvSettings(
        alpaca_api_key="paper-key",
        alpaca_secret_key="paper-secret",
        alpaca_paper=True,
        nlp_backend="lexicon",
        agy_data_dir=data_dir,
        agy_config_dir=REPO_CONFIG_DIR,
    )
    broker = AlpacaBroker(
        settings,
        db,
        trading_client=tc,
        quote_client=qc,
        limit_timeout_seconds=0.0,
        poll_interval_seconds=0.0,
        sleep_fn=lambda _: None,
        now_fn=lambda: rebal_now_1540,
    )
    daemon = TraderDaemon(
        settings,
        data_dir,
        REPO_CONFIG_DIR,
        db=db,
        bar_store=bar_store,
        broker=broker,
        network_check_fn=lambda: True,
    )
    daemon.startup(now_utc=rebal_now_1540 - timedelta(hours=4))
    submits_before = len(tc.submitted_requests)

    res_closed = daemon.run_once(now_utc=rebal_now_1540)
    assert res_closed["status"] == "idle_market_closed"
    assert res_closed["rebalance_ran"] is False
    assert len(tc.submitted_requests) == submits_before

    # Early-close day where market IS open and next_close=13:00 ET -> triggers in [12:40, 12:58) ET!
    early_close_utc = datetime(last_bar_date.year, last_bar_date.month, last_bar_date.day, 18, 0, tzinfo=UTC)  # 13:00 ET
    early_now_utc = datetime(last_bar_date.year, last_bar_date.month, last_bar_date.day, 17, 45, tzinfo=UTC)  # 12:45 ET
    tc.get_clock = lambda: types.SimpleNamespace(is_open=True, next_close=early_close_utc)  # type: ignore[method-assign]
    res_half = daemon.run_once(now_utc=early_now_utc)
    assert res_half["rebalance_ran"] is True
    assert res_half["rebalance"]["checkpoint"] == "12:40"


def test_m6_rebalance_every_n_days_and_custom_checkpoint(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "config"
    shutil.copytree(REPO_CONFIG_DIR, cfg_dir)
    raw_cfg = yaml.safe_load((cfg_dir / "strategy.yaml").read_text(encoding="utf-8"))
    raw_cfg["strategy"]["rebalance_checkpoints_et"] = ["10:00"]
    raw_cfg["strategy"]["rebalance_every_n_days"] = 3
    (cfg_dir / "strategy.yaml").write_text(yaml.safe_dump(raw_cfg), encoding="utf-8")

    data_dir = tmp_path / "data"
    db = StateDB(data_dir / "state.db")
    bar_store = ParquetBarStore(data_dir / "parquet")
    universe = load_universe(cfg_dir)
    generate_synthetic_dataset(universe=universe, bar_store=bar_store, db=db, days=320, seed=42)

    spy_df = bar_store.load_bars("SPY", "1d")
    last_bar_date = ensure_utc(spy_df["timestamp"].iloc[-1].to_pydatetime()).date()
    # Prior rebalance was 1 trading day earlier -> must skip when rebalance_every_n_days=3!
    prev_day = last_bar_date - timedelta(days=1)
    db.record_daemon_event(
        "trader",
        "rebalance",
        {"session_date": prev_day.isoformat(), "checkpoint": "10:00", "status": "completed"},
        timestamp=datetime(prev_day.year, prev_day.month, prev_day.day, 15, 5, tzinfo=UTC),
    )

    tc = _FakeTradingClient()
    tc.clock_open = True
    qc = _FakeQuoteClient({u.symbol: float(bar_store.load_bars(u.symbol, "1d")["close"].iloc[-1]) for u in universe})
    settings = EnvSettings(
        alpaca_api_key="paper-key",
        alpaca_secret_key="paper-secret",
        alpaca_paper=True,
        nlp_backend="lexicon",
        agy_data_dir=data_dir,
        agy_config_dir=cfg_dir,
    )
    now_1005 = datetime(last_bar_date.year, last_bar_date.month, last_bar_date.day, 15, 5, tzinfo=UTC)  # 10:05 ET
    broker = AlpacaBroker(
        settings,
        db,
        trading_client=tc,
        quote_client=qc,
        limit_timeout_seconds=0.0,
        poll_interval_seconds=0.0,
        sleep_fn=lambda _: None,
        now_fn=lambda: now_1005,
    )
    daemon = TraderDaemon(
        settings, data_dir, cfg_dir, db=db, bar_store=bar_store, broker=broker, network_check_fn=lambda: True
    )
    daemon.startup(now_utc=now_1005 - timedelta(minutes=30))
    res_skip = daemon.run_once(now_utc=now_1005)
    assert res_skip["rebalance_ran"] is False


def test_h1_adopted_open_pass1_order_polls_to_filled_before_pass2(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    db = StateDB(data_dir / "state.db")
    settings = EnvSettings(
        alpaca_api_key="paper-key",
        alpaca_secret_key="paper-secret",
        alpaca_paper=True,
        agy_data_dir=data_dir,
        agy_config_dir=REPO_CONFIG_DIR,
    )
    tc = _FakeTradingClient()
    qc = _FakeQuoteClient({"AAPL": 180.0, "MSFT": 400.0})
    now = datetime(2026, 11, 20, 20, 40, tzinfo=UTC)

    # Pre-seed an open ("new") Pass-1 sell order on the broker from before crash
    cid_p1 = build_client_order_id("2026-11-20", "1540", "AAPL", OrderSide.sell, leg="p1m0")
    open_p1 = _FakeOrder(
        id="brk-precrash-1",
        client_order_id=cid_p1,
        symbol="AAPL",
        side="sell",
        order_type="limit",
        qty="10.0",
        status="new",
        filled_qty="0",
        limit_price="179.73",
    )
    tc.orders_by_id[open_p1.id] = open_p1
    tc.orders_by_cid[cid_p1] = open_p1

    # On first poll after restart, the open Pass-1 order transitions to filled!
    poll_calls = 0
    orig_get_by_id = tc.get_order_by_id

    def _poll_transition(order_id: str) -> _FakeOrder:
        nonlocal poll_calls
        ord_obj = orig_get_by_id(order_id)
        if order_id == "brk-precrash-1":
            poll_calls += 1
            if poll_calls >= 1:
                ord_obj.status = "filled"
                ord_obj.filled_qty = "10.0"
                ord_obj.filled_avg_price = "179.95"
        return ord_obj

    tc.get_order_by_id = _poll_transition  # type: ignore[method-assign]

    broker = AlpacaBroker(
        settings,
        db,
        trading_client=tc,
        quote_client=qc,
        limit_timeout_seconds=1.0,
        poll_interval_seconds=0.1,
        sleep_fn=lambda _: None,
        now_fn=lambda: now,
    )

    v1 = RiskVerdict(
        approved=True,
        order=ProposedOrder(
            symbol="AAPL",
            side=OrderSide.sell,
            qty=10.0,
            is_risk_reducing=True,
            reference_price=180.0,
        ),
        adjusted_qty=10.0,
    )
    v2 = RiskVerdict(
        approved=True,
        order=ProposedOrder(
            symbol="MSFT",
            side=OrderSide.buy,
            qty=5.0,
            is_risk_reducing=False,
            reference_price=400.0,
            stop_loss_price=370.0,
        ),
        adjusted_qty=5.0,
        stop_loss_price=370.0,
    )
    report = broker.execute_verdicts([v1, v2], session_date="2026-11-20", checkpoint="1540", run_id="r1")
    assert report.errors == []
    assert len(report.records) == 2
    assert report.records[0].adopted_existing is True
    assert report.records[0].status == "filled"
    assert report.records[0].filled_qty == 10.0
    # Pass-2 MSFT order MUST have proceeded cleanly!
    assert report.records[1].symbol == "MSFT"
    assert report.records[1].status == "filled"


def test_h4_m3_low1_market_data_recovery_eod_retry_dynamic_gap_and_client_ownership(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    universe = load_universe(REPO_CONFIG_DIR)
    symbols = [u.symbol for u in universe]
    boot_dt = datetime(2026, 11, 20, 13, 0, tzinfo=UTC)

    bars_map = {sym: _make_daily_bars(sym, 310, boot_dt - timedelta(days=25)) for sym in symbols}

    class _FlakyMarketProvider(_StubMarketProvider):
        def __init__(self, bmap: dict[str, pd.DataFrame]) -> None:
            super().__init__(bmap)
            self.fail_fetch = False

        def fetch_bars(
            self, symbols: list[str], start: datetime, end: datetime, timeframe: str = "1d"
        ) -> dict[str, pd.DataFrame]:
            if self.fail_fetch:
                raise OSError("Alpaca temporary outage")
            return super().fetch_bars(symbols, start, end, timeframe)

    market_prov = _FlakyMarketProvider(bars_map)
    db = StateDB(data_dir / "state.db")
    bar_store = ParquetBarStore(data_dir / "parquet")
    tc = _FakeTradingClient()

    net_up = False
    daemon = MarketDataDaemon(
        EnvSettings(agy_data_dir=data_dir, agy_config_dir=REPO_CONFIG_DIR),
        data_dir,
        REPO_CONFIG_DIR,
        db=db,
        bar_store=bar_store,
        market_provider=market_prov,
        macro_provider=_StubMacroProvider(),
        earnings_provider=_StubEarningsProvider(),
        sec_provider=_StubSecProvider(),
        etf_holdings_provider=_StubEtfHoldingsProvider(),
        trading_client=tc,
        network_check_fn=lambda: net_up,
        min_bars_required=300,
        max_network_wait_seconds=0.0,
        eod_retry_interval_seconds=300.0,
    )
    # 1) Cold boot during network outage -> _boot_backfill_done stays False
    daemon.startup(now_utc=boot_dt)
    assert daemon._boot_backfill_done is False
    assert len(bar_store.list_symbols("1d")) == 0

    # 2) Network recovers -> first run_once retries & completes initial backfill!
    net_up = True
    res_rec = daemon.run_once(now_utc=boot_dt + timedelta(minutes=2))
    assert res_rec["boot_recovered"] is True
    assert res_rec["boot_backfill_done"] is True
    assert len(bar_store.load_bars("SPY", "1d")) >= 300

    # 3) M3: Simulate >14-day stale Parquet gap (bars end 25 days ago) -> dynamic start_recent!
    market_prov.calls.clear()
    market_prov.bars_map = {
        sym: _make_daily_bars(sym, 310, boot_dt - timedelta(days=1)) for sym in symbols
    }
    daemon._sync_bars_gap_fill(symbols, boot_dt)
    assert market_prov.calls
    _, gap_start, gap_end = market_prov.calls[-1]
    assert (gap_end - gap_start).days >= 28  # 25d gap + 5d overlap!

    # 4) H4: EOD sync fails -> _last_eod_session_date NOT set, retries after 300s & succeeds!
    eod_dt = datetime(2026, 11, 20, 21, 10, tzinfo=UTC)  # 16:10 ET
    market_prov.fail_fetch = True
    res_fail = daemon.run_once(now_utc=eod_dt)
    assert res_fail["eod_failed"] is True
    assert res_fail["eod_ran"] is False
    assert res_fail["status"] == "degraded"
    assert daemon._last_eod_session_date is None

    market_prov.fail_fetch = False
    res_ok = daemon.run_once(now_utc=eod_dt + timedelta(minutes=6))
    assert res_ok["eod_ran"] is True
    assert res_ok["eod_failed"] is False
    assert daemon._last_eod_session_date == "2026-11-20"

    # 5) LOW #1: shutdown must NOT close caller-injected trading_client
    daemon.shutdown()
    assert tc.closed is False


def test_m2_news_nlp_per_source_watermarks_and_h3_rotating_logs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    db = StateDB(data_dir / "state.db")
    now = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)

    # Seed Finnhub article 1 hour ago, and older Benzinga article 3 days ago
    db.upsert_news_articles(
        [
            NewsArticle(
                article_id="fh:old",
                published_at=now - timedelta(hours=1),
                headline="Finnhub recent item",
                source="finnhub",
                symbols=["AAPL"],
            ),
            NewsArticle(
                article_id="bz:old",
                published_at=now - timedelta(days=3),
                headline="Benzinga older item",
                source="benzinga",
                symbols=["AAPL"],
            ),
        ]
    )

    class _CaptureNewsProv(NewsProvider):
        def __init__(self) -> None:
            self.seen_starts: list[datetime] = []

        def fetch_news(self, symbols: list[str], start: datetime, end: datetime) -> list[NewsArticle]:
            self.seen_starts.append(start)
            return []

    alp_prov = _CaptureNewsProv()
    fh_prov = _CaptureNewsProv()

    daemon = NewsNlpDaemon(
        EnvSettings(agy_data_dir=data_dir, agy_config_dir=REPO_CONFIG_DIR, nlp_backend="lexicon"),
        data_dir,
        REPO_CONFIG_DIR,
        db=db,
        alpaca_news_provider=alp_prov,
        finnhub_news_provider=fh_prov,
        rss_collector=list,
        yfinance_collector=lambda _syms: [],
        scorer=LexiconSentimentScorer(),
        network_check_fn=lambda: True,
    )
    daemon.run_once(now_utc=now)
    # Alpaca window must start ~3d1h ago (NOT clamped by Finnhub's 1h-ago article!)
    assert alp_prov.seen_starts[0] <= now - timedelta(days=3)
    assert fh_prov.seen_starts[0] >= now - timedelta(hours=2, minutes=5)

    # H3: RotatingFileHandler logging check
    handler = configure_daemon_logging(data_dir, "test-daemon")
    try:
        assert isinstance(handler, RotatingFileHandler)
        assert handler.maxBytes == 10 * 1024 * 1024
        assert handler.backupCount == 5
        assert Path(handler.baseFilename).parent == (data_dir / "logs").resolve()
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
