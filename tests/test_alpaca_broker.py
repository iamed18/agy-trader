from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import QueryOrderStatus, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, StopOrderRequest
from typer.testing import CliRunner

from agy_trader.cli.main import app
from agy_trader.core.config import LIVE_CONFIRMATION_PHRASE, EnvSettings
from agy_trader.core.control_plane import ControlPlane
from agy_trader.core.models import OrderSide, ProposedOrder, RiskVerdict
from agy_trader.execution.alpaca_broker import (
    AlpacaBroker,
    build_client_order_id,
)
from agy_trader.storage.db import StateDB


@dataclass
class _FakeQuote:
    bid_price: float
    ask_price: float


@dataclass
class _FakeAccount:
    equity: str = "100000.00"
    cash: str = "50000.00"
    buying_power: str = "150000.00"
    last_equity: str = "99500.00"


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


class _NotFoundError(Exception):
    status_code = 404


class _FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now_dt = start or datetime(2026, 11, 20, 20, 40, tzinfo=UTC)

    def now(self) -> datetime:
        return self.now_dt

    def sleep(self, seconds: float) -> None:
        self.now_dt += timedelta(seconds=seconds)


class _FakeTradingClient:
    def __init__(self) -> None:
        self.account = _FakeAccount()
        self.positions: list[_FakePosition] = []
        self.orders_by_id: dict[str, _FakeOrder] = {}
        self.orders_by_cid: dict[str, _FakeOrder] = {}
        self.submitted_requests: list[Any] = []
        self.canceled_ids: list[str] = []
        self.cancel_all_called = False
        self._seq = 0
        self.on_submit: Any = None
        self.on_poll: Any = None

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
        return [o for o in self.orders_by_id.values() if o.status in open_statuses]

    def get_order_by_client_id(self, client_id: str) -> _FakeOrder:
        if client_id not in self.orders_by_cid:
            raise _NotFoundError(f"404 order not found: {client_id}")
        return self.orders_by_cid[client_id]

    def get_order_by_id(self, order_id: str) -> _FakeOrder:
        if order_id not in self.orders_by_id:
            raise _NotFoundError(f"404 order not found: {order_id}")
        order = self.orders_by_id[order_id]
        if self.on_poll is not None:
            self.on_poll(order)
        return order

    def submit_order(self, order_data: Any) -> _FakeOrder:
        self.submitted_requests.append(order_data)
        self._seq += 1
        oid = f"brk-{self._seq}"
        cid = str(getattr(order_data, "client_order_id", f"cid-{self._seq}"))
        if isinstance(order_data, LimitOrderRequest):
            otype = "limit"
        elif isinstance(order_data, MarketOrderRequest):
            otype = "market"
        elif isinstance(order_data, StopOrderRequest):
            otype = "stop"
        else:
            otype = "unknown"

        side_str = getattr(order_data.side, "value", str(order_data.side)).lower()
        order = _FakeOrder(
            id=oid,
            client_order_id=cid,
            symbol=str(order_data.symbol).upper(),
            side=side_str,
            order_type=otype,
            qty=str(order_data.qty),
            status="new",
            filled_qty="0",
            filled_avg_price=None,
            limit_price=(
                str(order_data.limit_price)
                if getattr(order_data, "limit_price", None) is not None
                else None
            ),
            stop_price=(
                str(order_data.stop_price)
                if getattr(order_data, "stop_price", None) is not None
                else None
            ),
        )
        if self.on_submit is not None:
            self.on_submit(order, order_data)
        self.orders_by_id[oid] = order
        self.orders_by_cid[cid] = order
        return order

    def cancel_order_by_id(self, order_id: str) -> None:
        self.canceled_ids.append(order_id)
        order = self.orders_by_id.get(order_id)
        if order is not None and order.status not in {"filled", "canceled"}:
            order.status = "canceled"

    def cancel_orders(self) -> list[dict[str, str]]:
        self.cancel_all_called = True
        canceled: list[dict[str, str]] = []
        for order in self.orders_by_id.values():
            if order.status in {"new", "accepted", "partially_filled", "pending_new"}:
                order.status = "canceled"
                canceled.append({"id": order.id})
        return canceled


class _FakeQuoteClient:
    def __init__(self, quotes: dict[str, _FakeQuote] | None = None, *, fail: bool = False) -> None:
        self.quotes = quotes or {}
        self.fail = fail

    def get_stock_latest_quote(self, req: Any) -> dict[str, _FakeQuote]:
        if self.fail:
            raise RuntimeError("quote feed timeout")
        sym = req.symbol_or_symbols
        if sym in self.quotes:
            return {sym: self.quotes[sym]}
        return {}


def _verdict(
    symbol: str,
    side: OrderSide,
    qty: float,
    ref_px: float,
    *,
    is_risk_reducing: bool = False,
    approved: bool = True,
    stop_loss_price: float | None = None,
) -> RiskVerdict:
    return RiskVerdict(
        approved=approved,
        order=ProposedOrder(
            symbol=symbol,
            side=side,
            qty=qty,
            is_risk_reducing=is_risk_reducing,
            reference_price=ref_px,
            stop_loss_price=stop_loss_price,
        ),
        adjusted_qty=qty if approved else 0.0,
        stop_loss_price=stop_loss_price,
    )


def test_double_lock_enforced_on_init_and_execution(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    tc = _FakeTradingClient()

    bad_settings = EnvSettings(
        alpaca_api_key="k",
        alpaca_secret_key="s",
        alpaca_paper=False,
        live_trading_confirmation="",
    )
    with pytest.raises(ValueError, match="double-lock"):
        AlpacaBroker(bad_settings, db, trading_client=tc)

    # Even if settings are mutated after paper init, execution checks double-lock again
    settings = EnvSettings(alpaca_api_key="k", alpaca_secret_key="s", alpaca_paper=True)
    broker = AlpacaBroker(settings, db, trading_client=tc)
    settings.alpaca_paper = False
    with pytest.raises(ValueError, match="double-lock"):
        broker.execute_verdicts([], session_date="2026-11-20", checkpoint="15:40", run_id="r1")
    with pytest.raises(ValueError, match="double-lock"):
        broker.sync_protective_stops({}, session_date="2026-11-20", run_id="r1")
    with pytest.raises(ValueError, match="double-lock"):
        broker.emergency_flatten(confirm=True, reason="test")

    # With exact confirmation phrase, live mode passes validation
    live_ok = EnvSettings(
        alpaca_api_key="k",
        alpaca_secret_key="s",
        alpaca_paper=False,
        live_trading_confirmation=LIVE_CONFIRMATION_PHRASE,
    )
    live_broker = AlpacaBroker(live_ok, db, trading_client=tc)
    assert live_broker.get_account_state()["equity"] == pytest.approx(100_000.0)


def test_build_client_order_id_deterministic_and_within_48_chars() -> None:
    cid1 = build_client_order_id("2026-11-20", "15:40", "aapl", OrderSide.buy, leg="main")
    cid2 = build_client_order_id("2026-11-20", "15:40", "AAPL", OrderSide.buy, leg="main")
    assert cid1 == cid2 == "agy:20261120:1540:AAPL:B:main"
    assert len(cid1) <= 48

    long_cid = build_client_order_id(
        "2026-11-20",
        "15:40",
        "VERYLONGTICKER",
        OrderSide.sell,
        leg="very_long_custom_leg_identifier_exceeding_limit",
    )
    assert len(long_cid) <= 48
    assert long_cid.isascii()
    assert long_cid == build_client_order_id(
        "2026-11-20",
        "15:40",
        "VERYLONGTICKER",
        OrderSide.sell,
        leg="very_long_custom_leg_identifier_exceeding_limit",
    )


def test_execute_verdicts_pass1_before_pass2_collar_and_slippage(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    tc = _FakeTradingClient()
    qc = _FakeQuoteClient(
        {
            "AAPL": _FakeQuote(bid_price=199.90, ask_price=200.00),
            "MSFT": _FakeQuote(bid_price=400.00, ask_price=400.20),
        }
    )
    clock = _FakeClock()

    def _fill_immediately(order: _FakeOrder, req: Any) -> None:
        order.status = "filled"
        order.filled_qty = order.qty
        if order.symbol == "MSFT":
            # Sold at 399.60 vs ref 400.00 -> +10 bps unfavorable sell slippage
            order.filled_avg_price = "399.60"
        else:
            # Bought at 200.10 vs ref 200.00 -> +5 bps unfavorable buy slippage
            order.filled_avg_price = "200.10"

    tc.on_submit = _fill_immediately
    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        quote_client=qc,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
    )

    # Pass-2 entry listed BEFORE Pass-1 exit in input list; Pass-1 MUST run first!
    verdicts = [
        _verdict("AAPL", OrderSide.buy, 10.0, 200.00, is_risk_reducing=False),
        _verdict("SPY", OrderSide.buy, 5.0, 500.00, approved=False),
        _verdict("MSFT", OrderSide.sell, 4.0, 400.00, is_risk_reducing=True),
    ]
    report = broker.execute_verdicts(
        verdicts, session_date="2026-11-20", checkpoint="15:40", run_id="run-1"
    )

    assert report.pass1_count == 1
    assert report.pass2_count == 1
    assert report.skipped_unapproved == 1
    assert [r.symbol for r in report.records] == ["MSFT", "AAPL"]
    assert [req.symbol for req in tc.submitted_requests] == ["MSFT", "AAPL"]

    # MSFT sell limit: bid 400.00 * (1 - 0.0015) = 399.40
    # AAPL buy limit: ask 200.00 * (1 + 0.0015) = 200.30
    assert tc.submitted_requests[0].limit_price == pytest.approx(399.40)
    assert tc.submitted_requests[1].limit_price == pytest.approx(200.30)

    msft_rec, aapl_rec = report.records
    assert msft_rec.realized_slippage_bps == pytest.approx(10.0)
    assert aapl_rec.realized_slippage_bps == pytest.approx(5.0)

    stored_aapl = db.get_live_order(aapl_rec.client_order_id)
    assert stored_aapl is not None
    assert stored_aapl["status"] == "filled"
    assert stored_aapl["realized_slippage_bps"] == pytest.approx(5.0)


def test_execute_verdicts_limit_timeout_partial_fill_market_fallback(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    tc = _FakeTradingClient()
    qc = _FakeQuoteClient(fail=True)  # Force reference_price fallback too
    clock = _FakeClock()

    def _on_submit(order: _FakeOrder, req: Any) -> None:
        if isinstance(req, LimitOrderRequest):
            # Partially fill 4 of 10 shares, stay open until canceled
            order.status = "partially_filled"
            order.filled_qty = "4"
            order.filled_avg_price = "100.00"
        elif isinstance(req, MarketOrderRequest):
            order.status = "filled"
            order.filled_qty = order.qty
            order.filled_avg_price = "100.30"

    tc.on_submit = _on_submit
    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        quote_client=qc,
        limit_timeout_seconds=4.0,
        poll_interval_seconds=2.0,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
    )

    report = broker.execute_verdicts(
        [_verdict("NVDA", OrderSide.buy, 10.0, 100.00, is_risk_reducing=False)],
        session_date="2026-11-20",
        checkpoint="15:40",
        run_id="run-fallback",
    )

    assert len(tc.submitted_requests) == 2
    assert isinstance(tc.submitted_requests[0], LimitOrderRequest)
    assert tc.submitted_requests[0].limit_price == pytest.approx(100.15)  # ref 100 * 1.0015
    assert isinstance(tc.submitted_requests[1], MarketOrderRequest)
    assert tc.submitted_requests[1].qty == pytest.approx(6.0)
    assert len(tc.canceled_ids) == 1

    rec = report.records[0]
    assert rec.fallback_to_market is True
    assert rec.fallback_client_order_id == "agy:20261120:1540:NVDA:B:p2k0"
    assert rec.filled_qty == pytest.approx(10.0)
    # Weighted fill: (4 * 100.0 + 6 * 100.30) / 10 = 100.18 -> 18 bps
    assert rec.filled_avg_price == pytest.approx(100.18)
    assert rec.realized_slippage_bps == pytest.approx(18.0)
    assert db.get_live_order("agy:20261120:1540:NVDA:B:p2m0") is not None
    assert db.get_live_order("agy:20261120:1540:NVDA:B:p2k0") is not None


def test_execute_verdicts_idempotent_replay_across_db_and_broker_crash(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    tc = _FakeTradingClient()
    clock = _FakeClock()

    # Scenario A: Broker already has the order (process crashed before writing to StateDB)
    existing_cid = build_client_order_id("2026-11-20", "15:40", "AAPL", OrderSide.buy, leg="p2m0")
    broker_ord = _FakeOrder(
        id="brk-preexisting",
        client_order_id=existing_cid,
        symbol="AAPL",
        side="buy",
        order_type="limit",
        qty="10",
        status="filled",
        filled_qty="10",
        filled_avg_price="199.80",
        limit_price="200.30",
    )
    tc.orders_by_id[broker_ord.id] = broker_ord
    tc.orders_by_cid[existing_cid] = broker_ord

    # Scenario B: StateDB already has MSFT order (and broker returns 404 or matches)
    msft_cid = build_client_order_id("2026-11-20", "15:40", "MSFT", OrderSide.sell, leg="p1m0")
    db.upsert_live_order(
        client_order_id=msft_cid,
        broker_order_id="brk-msft-db",
        run_id="prior-run",
        symbol="MSFT",
        side="sell",
        order_type="limit",
        qty=5.0,
        limit_price=400.0,
        status="filled",
        filled_qty=5.0,
        filled_avg_price=400.40,
        reference_price=400.0,
        realized_slippage_bps=-10.0,
        is_risk_reducing=True,
    )

    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
    )
    report = broker.execute_verdicts(
        [
            _verdict("AAPL", OrderSide.buy, 10.0, 200.0, is_risk_reducing=False),
            _verdict("MSFT", OrderSide.sell, 5.0, 400.0, is_risk_reducing=True),
        ],
        session_date="2026-11-20",
        checkpoint="15:40",
        run_id="restart-run",
    )

    # Zero duplicate orders submitted!
    assert tc.submitted_requests == []
    assert all(r.adopted_existing for r in report.records)
    adopted_aapl = db.get_live_order(existing_cid)
    assert adopted_aapl is not None
    assert adopted_aapl["broker_order_id"] == "brk-preexisting"
    assert adopted_aapl["filled_avg_price"] == pytest.approx(199.80)


def test_sync_protective_stops_orphans_never_widen_and_fractional_gtc_constraint(
    tmp_path: Path,
) -> None:
    db = StateDB(tmp_path / "state.db")
    tc = _FakeTradingClient()
    clock = _FakeClock()

    # Open positions:
    # - AAPL: whole long 10 shares; existing tighter stop at 195.00 (desired 190.00 -> MUST keep 195.00!)
    # - NVDA: fractional long 5.75 shares; no stop yet -> floors to 5.0 whole shares GTC + warning
    # - MINI: sub-1 fractional long 0.45 shares -> DAY fallback + warning
    # - TSLA: short -8 shares (side="short", positive qty string to test sign normalization!) -> needs BUY GTC stop
    tc.positions = [
        _FakePosition("AAPL", "10", "200.00", "205.00", "50.00", side="long"),
        _FakePosition("NVDA", "5.75", "120.00", "125.00", "28.75", side="long"),
        _FakePosition("MINI", "0.45", "500.00", "505.00", "2.25", side="long"),
        _FakePosition("TSLA", "8", "250.00", "245.00", "40.00", side="short"),
    ]

    # Existing open orders:
    # - Orphan stop on flat symbol GOOGL -> must cancel
    # - Wrong-side BUY stop on long AAPL -> must cancel
    # - Tighter SELL stop on AAPL at 195.00 -> must keep (never widen to 190.00!)
    # - Stale wide BUY stop on TSLA at 280.00 (desired 265.00) -> must cancel & replace
    orphan_stop = _FakeOrder("ord-orphan", "cid-orp", "GOOGL", "sell", "stop", "5", "new", stop_price="140.00")
    wrong_side_stop = _FakeOrder("ord-wrong", "cid-wrg", "AAPL", "buy", "stop", "10", "new", stop_price="210.00")
    tight_aapl_stop = _FakeOrder("ord-aapl-ok", "cid-aapl", "AAPL", "sell", "stop", "10", "new", stop_price="195.00")
    wide_tsla_stop = _FakeOrder("ord-tsla-wide", "cid-tsla", "TSLA", "buy", "stop", "8", "new", stop_price="280.00")
    for o in (orphan_stop, wrong_side_stop, tight_aapl_stop, wide_tsla_stop):
        tc.orders_by_id[o.id] = o
        tc.orders_by_cid[o.client_order_id] = o

    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
    )

    res = broker.sync_protective_stops(
        {"AAPL": 190.00, "NVDA": 115.00, "MINI": 475.00, "TSLA": 265.00},
        session_date="2026-11-20",
        run_id="run-stops",
    )

    assert set(res.canceled_orphans) == {"ord-orphan", "ord-wrong"}
    assert res.canceled_stale == ["ord-tsla-wide"]
    assert res.kept_stops == {"AAPL": "ord-aapl-ok"}
    assert set(res.submitted_stops) == {"MINI", "NVDA", "TSLA"}
    assert res.errors == []

    # Verify submitted StopOrderRequests respect Alpaca fractional-GTC constraint
    by_sym_req = {req.symbol: req for req in tc.submitted_requests}
    assert isinstance(by_sym_req["NVDA"], StopOrderRequest)
    assert by_sym_req["NVDA"].qty == pytest.approx(5.0)
    assert by_sym_req["NVDA"].time_in_force == TimeInForce.GTC
    assert by_sym_req["NVDA"].side == AlpacaOrderSide.SELL

    assert by_sym_req["MINI"].qty == pytest.approx(0.45)
    assert by_sym_req["MINI"].time_in_force == TimeInForce.DAY

    assert by_sym_req["TSLA"].qty == pytest.approx(8.0)
    assert by_sym_req["TSLA"].time_in_force == TimeInForce.GTC
    assert by_sym_req["TSLA"].side == AlpacaOrderSide.BUY
    assert by_sym_req["TSLA"].stop_price == pytest.approx(265.00)

    assert any("NVDA" in w and "floored" in w for w in res.warnings)
    assert any("MINI" in w and "DAY" in w for w in res.warnings)

    live_pos = {p["symbol"]: p for p in db.get_live_positions()}
    assert live_pos["AAPL"]["stop_order_id"] == "ord-aapl-ok"
    assert live_pos["AAPL"]["stop_price"] == pytest.approx(195.00)
    assert live_pos["TSLA"]["qty"] == pytest.approx(-8.0)
    assert live_pos["TSLA"]["stop_price"] == pytest.approx(265.00)


def test_reconcile_with_broker_diffs_updates_orders_and_repairs_missing_stops(
    tmp_path: Path,
) -> None:
    db = StateDB(tmp_path / "state.db")
    tc = _FakeTradingClient()
    clock = _FakeClock()

    # Stale SQLite state: thinks we hold OLD (closed on broker) and AAPL qty=5 (broker has 10)
    db.replace_live_positions(
        [
            {
                "symbol": "OLD",
                "qty": 10.0,
                "avg_entry_price": 50.0,
                "market_price": 51.0,
                "unrealized_pnl": 10.0,
            },
            {
                "symbol": "AAPL",
                "qty": 5.0,
                "avg_entry_price": 200.0,
                "market_price": 202.0,
                "unrealized_pnl": 10.0,
            },
        ]
    )
    # Open order in SQLite that filled on broker while daemon was down
    db.upsert_live_order(
        client_order_id="agy:20261120:1540:AAPL:B:main",
        broker_order_id="brk-open-1",
        run_id="r0",
        symbol="AAPL",
        side="buy",
        order_type="limit",
        qty=5.0,
        status="new",
        reference_price=200.0,
        is_risk_reducing=False,
    )
    tc.orders_by_id["brk-open-1"] = _FakeOrder(
        id="brk-open-1",
        client_order_id="agy:20261120:1540:AAPL:B:main",
        symbol="AAPL",
        side="buy",
        order_type="limit",
        qty="5",
        status="filled",
        filled_qty="5",
        filled_avg_price="200.20",
    )
    tc.orders_by_cid["agy:20261120:1540:AAPL:B:main"] = tc.orders_by_id["brk-open-1"]

    # Broker ground truth: AAPL=10 (NO resting stop!), SPY=4 (NO resting stop!)
    tc.positions = [
        _FakePosition("AAPL", "10", "200.10", "204.00", "39.00", side="long"),
        _FakePosition("SPY", "4", "590.00", "592.00", "8.00", side="long"),
    ]

    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
    )

    rec = broker.reconcile_with_broker(daemon_name="agy-trader")
    assert rec.broker_positions_count == 2
    assert rec.db_positions_count == 2
    diff_kinds = {d["symbol"]: d["kind"] for d in rec.position_diffs}
    assert diff_kinds == {
        "OLD": "closed_on_broker",
        "AAPL": "qty_mismatch",
        "SPY": "opened_on_broker",
    }
    assert rec.updated_orders == ["agy:20261120:1540:AAPL:B:main"]
    assert rec.missing_stop_symbols == ["AAPL", "SPY"]
    assert rec.stop_sync_result is not None
    assert set(rec.stop_sync_result.submitted_stops) == {"AAPL", "SPY"}

    updated_ord = db.get_live_order("agy:20261120:1540:AAPL:B:main")
    assert updated_ord is not None
    assert updated_ord["status"] == "filled"
    assert updated_ord["filled_avg_price"] == pytest.approx(200.20)

    db_pos = {p["symbol"]: p for p in db.get_live_positions()}
    assert set(db_pos) == {"AAPL", "SPY"}
    assert db_pos["AAPL"]["stop_price"] == pytest.approx(round(204.00 * 0.90, 2))
    assert db_pos["SPY"]["stop_price"] == pytest.approx(round(592.00 * 0.90, 2))

    events = db.get_daemon_events(daemon_name="agy-trader", event_type="reconcile")
    assert len(events) == 1
    assert events[0]["details"]["missing_stop_symbols"] == ["AAPL", "SPY"]


def test_emergency_flatten_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = StateDB(tmp_path / "state.db")
    tc = _FakeTradingClient()
    clock = _FakeClock()

    tc.positions = [
        _FakePosition("AAPL", "10", "200.0", "201.0", "10.0", side="long"),
        _FakePosition("TSLA", "-5", "250.0", "249.0", "5.0", side="short"),
    ]
    open_stop = _FakeOrder("ord-1", "cid-1", "AAPL", "sell", "stop", "10", status="new", stop_price="190.0")
    tc.orders_by_id[open_stop.id] = open_stop
    tc.orders_by_cid[open_stop.client_order_id] = open_stop

    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
    )

    with pytest.raises(ValueError, match="confirm=True"):
        broker.emergency_flatten(confirm=False, reason="panic")
    with pytest.raises(ValueError, match="reason"):
        broker.emergency_flatten(confirm=True, reason="   ")

    summary = broker.emergency_flatten(confirm=True, reason="manual kill drill", actor="alice")
    assert summary["ok"] is True
    assert summary["paused_new_entries"] is True
    assert summary["canceled_orders"] == 1
    assert tc.cancel_all_called is True
    assert len(summary["close_orders"]) == 2

    closes_by_sym = {c["symbol"]: c for c in summary["close_orders"]}
    assert closes_by_sym["AAPL"]["side"] == "sell"
    assert closes_by_sym["AAPL"]["qty"] == pytest.approx(10.0)
    assert closes_by_sym["TSLA"]["side"] == "buy"
    assert closes_by_sym["TSLA"]["qty"] == pytest.approx(5.0)

    # ControlPlane paused & audited
    dials = ControlPlane(db).get_dials()
    assert dials.pause_new_entries is True
    audit = db.get_control_audit_log(limit=5)
    assert audit[0]["actor"] == "alice"
    assert "[EMERGENCY_FLATTEN]" in audit[0]["reason"]

    # CLI test: refuses without --confirm, succeeds with --confirm
    runner = CliRunner()
    cli_data = tmp_path / "cli_data"
    res_no_confirm = runner.invoke(
        app,
        [
            "emergency-flatten",
            "--reason",
            "drill",
            "--data-dir",
            str(cli_data),
            "--json",
        ],
    )
    assert res_no_confirm.exit_code == 1
    assert json.loads(res_no_confirm.stdout)["ok"] is False

    # Inject fake TradingClient via monkeypatch for CLI execution
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ALPACA_PAPER", "true")
    monkeypatch.setattr(
        "agy_trader.execution.alpaca_broker.TradingClient",
        lambda **kwargs: _FakeTradingClient(),
    )
    monkeypatch.setattr(
        "agy_trader.execution.alpaca_broker.StockHistoricalDataClient",
        lambda **kwargs: _FakeQuoteClient(),
    )
    res_ok = runner.invoke(
        app,
        [
            "emergency-flatten",
            "--confirm",
            "--reason",
            "cli drill",
            "--actor",
            "cli-op",
            "--data-dir",
            str(cli_data),
            "--json",
        ],
    )
    assert res_ok.exit_code == 0, res_ok.output
    payload = json.loads(res_ok.stdout)
    assert payload["ok"] is True
    assert payload["paused_new_entries"] is True
    assert payload["actor"] == "cli-op"


# ---------------------------------------------------------------------------
# Gate D Adversarial Regression Tests
# ---------------------------------------------------------------------------


def test_long_to_short_flip_and_duplicate_same_side_distinct_client_order_ids(
    tmp_path: Path,
) -> None:
    """CRITICAL #1: Long->Short flip & same-(symbol,side,pass) batches never collide."""
    db = StateDB(tmp_path / "state.db")
    tc = _FakeTradingClient()
    clock = _FakeClock()

    def _fill_immediately(order: _FakeOrder, req: Any) -> None:
        order.status = "filled"
        order.filled_qty = order.qty
        order.filled_avg_price = "200.00"

    tc.on_submit = _fill_immediately
    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
    )

    verdicts = [
        _verdict("AAPL", OrderSide.sell, 10.0, 200.0, is_risk_reducing=True),
        _verdict("AAPL", OrderSide.sell, 5.0, 200.0, is_risk_reducing=False),
        _verdict("AAPL", OrderSide.sell, 3.0, 200.0, is_risk_reducing=False),
    ]
    report = broker.execute_verdicts(
        verdicts, session_date="2026-11-20", checkpoint="15:40", run_id="flip-run"
    )
    assert report.errors == []
    assert len(report.records) == 3
    cids = [r.client_order_id for r in report.records]
    assert len(set(cids)) == 3
    assert cids == [
        "agy:20261120:1540:AAPL:S:p1m0",
        "agy:20261120:1540:AAPL:S:p2m0",
        "agy:20261120:1540:AAPL:S:p2m1",
    ]
    assert all(len(c) <= 48 for c in cids)
    assert [req.qty for req in tc.submitted_requests] == [10.0, 5.0, 3.0]


def test_limit_cancel_pending_cancel_timeout_skips_market_fallback(tmp_path: Path) -> None:
    """CRITICAL #2: Never submit fallback MarketOrder if limit cancel stays pending_cancel."""
    db = StateDB(tmp_path / "state.db")
    clock = _FakeClock()

    class _StickyCancelClient(_FakeTradingClient):
        def cancel_order_by_id(self, order_id: str) -> None:
            self.canceled_ids.append(order_id)
            order = self.orders_by_id[order_id]
            order.status = "pending_cancel"

    tc = _StickyCancelClient()
    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        limit_timeout_seconds=4.0,
        poll_interval_seconds=2.0,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
    )

    report = broker.execute_verdicts(
        [_verdict("NVDA", OrderSide.buy, 10.0, 100.0, is_risk_reducing=False)],
        session_date="2026-11-20",
        checkpoint="15:40",
        run_id="sticky-cancel",
    )
    assert len(tc.submitted_requests) == 1
    assert isinstance(tc.submitted_requests[0], LimitOrderRequest)
    rec = report.records[0]
    assert rec.fallback_to_market is False
    assert rec.status == "pending_cancel"
    assert rec.error is not None and "pending_cancel" in rec.error
    assert len(report.errors) == 1
    stored = db.get_live_order(rec.client_order_id)
    assert stored is not None
    assert stored["status"] == "pending_cancel"


def test_sync_protective_stops_aborts_replacement_when_async_cancel_times_out(
    tmp_path: Path,
) -> None:
    """HIGH #3: Poll async cancels; abort replacement if stale stop cancel times out."""
    db = StateDB(tmp_path / "state.db")
    clock = _FakeClock()

    class _DelayedAndStickyCancelClient(_FakeTradingClient):
        def __init__(self) -> None:
            super().__init__()
            self.poll_counts: dict[str, int] = {}

        def cancel_order_by_id(self, order_id: str) -> None:
            self.canceled_ids.append(order_id)
            self.orders_by_id[order_id].status = "pending_cancel"

        def get_order_by_id(self, order_id: str) -> _FakeOrder:
            order = super().get_order_by_id(order_id)
            self.poll_counts[order_id] = self.poll_counts.get(order_id, 0) + 1
            # AAPL stale stop finishes canceling on 2nd poll; TSLA stays stuck in pending_cancel
            if order_id == "ord-aapl-stale" and self.poll_counts[order_id] >= 2:
                order.status = "canceled"
            return order

    tc = _DelayedAndStickyCancelClient()
    tc.positions = [
        _FakePosition("AAPL", "10", "200.0", "205.0", "50.0", side="long"),
        _FakePosition("TSLA", "-5", "250.0", "245.0", "25.0", side="short"),
    ]
    aapl_stale = _FakeOrder("ord-aapl-stale", "cid-a", "AAPL", "sell", "stop", "10", "new", stop_price="180.0")
    tsla_stale = _FakeOrder("ord-tsla-stale", "cid-t", "TSLA", "buy", "stop", "5", "new", stop_price="290.0")
    for o in (aapl_stale, tsla_stale):
        tc.orders_by_id[o.id] = o
        tc.orders_by_cid[o.client_order_id] = o

    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        poll_interval_seconds=2.0,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
    )
    res = broker.sync_protective_stops(
        {"AAPL": 195.0, "TSLA": 265.0},
        session_date="2026-11-20",
        run_id="async-cancel",
    )
    assert "AAPL" in res.submitted_stops
    assert "TSLA" not in res.submitted_stops
    assert any("TSLA" in e and "aborting replacement" in e for e in res.errors)
    assert [req.symbol for req in tc.submitted_requests] == ["AAPL"]


def test_emergency_flatten_waits_for_open_orders_to_drain(tmp_path: Path) -> None:
    """HIGH #3: emergency_flatten polls open orders until drained before closing positions."""
    db = StateDB(tmp_path / "state.db")
    clock = _FakeClock()

    class _AsyncDrainCancelClient(_FakeTradingClient):
        def __init__(self) -> None:
            super().__init__()
            self.open_polls = 0
            self.submitted_while_open = False

        def cancel_orders(self) -> list[dict[str, str]]:
            self.cancel_all_called = True
            for o in self.orders_by_id.values():
                o.status = "pending_cancel"
            return [{"id": o.id} for o in self.orders_by_id.values()]

        def get_orders(self, filter: Any = None) -> list[_FakeOrder]:
            self.open_polls += 1
            if self.open_polls >= 2:
                for o in self.orders_by_id.values():
                    if o.status == "pending_cancel":
                        o.status = "canceled"
            return super().get_orders(filter=filter)

        def submit_order(self, order_data: Any) -> _FakeOrder:
            if any(o.status == "pending_cancel" for o in self.orders_by_id.values()):
                self.submitted_while_open = True
            return super().submit_order(order_data)

    tc = _AsyncDrainCancelClient()
    tc.positions = [_FakePosition("AAPL", "10", "200.0", "201.0", "10.0", side="long")]
    open_ord = _FakeOrder("ord-open-1", "cid-open-1", "AAPL", "sell", "stop", "10", "new", stop_price="190.0")
    tc.orders_by_id[open_ord.id] = open_ord
    tc.orders_by_cid[open_ord.client_order_id] = open_ord

    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        poll_interval_seconds=1.0,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
    )
    res1 = broker.emergency_flatten(confirm=True, reason="drain test 1")
    assert res1["ok"] is True
    assert tc.submitted_while_open is False
    assert tc.open_polls >= 2

    # LOW #12: Back-to-back calls at exact same timestamp produce distinct client_order_ids
    res2 = broker.emergency_flatten(confirm=True, reason="drain test 2")
    assert res2["ok"] is True
    assert res1["close_orders"][0]["client_order_id"] != res2["close_orders"][0]["client_order_id"]


def test_sync_protective_stops_never_adopts_terminal_order_and_disambiguates(
    tmp_path: Path,
) -> None:
    """HIGH #4: Prior terminal stop order with same client_order_id is not adopted."""
    import hashlib

    db = StateDB(tmp_path / "state.db")
    tc = _FakeTradingClient()
    clock = _FakeClock()
    tc.positions = [_FakePosition("AAPL", "10", "200.0", "205.0", "50.0", side="long")]

    leg_tag = hashlib.sha1(b"same-run:195.00:10.000000").hexdigest()[:6]
    prior_cid = build_client_order_id(
        "2026-11-20", "stop", "AAPL", OrderSide.sell, leg=f"stp{leg_tag}"
    )
    prior_terminal = _FakeOrder(
        id="ord-prior-canceled",
        client_order_id=prior_cid,
        symbol="AAPL",
        side="sell",
        order_type="stop",
        qty="10",
        status="canceled",
        stop_price="195.00",
    )
    tc.orders_by_id[prior_terminal.id] = prior_terminal
    tc.orders_by_cid[prior_cid] = prior_terminal

    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
    )
    res = broker.sync_protective_stops(
        {"AAPL": 195.00}, session_date="2026-11-20", run_id="same-run"
    )
    assert res.submitted_stops["AAPL"] != "ord-prior-canceled"
    assert len(tc.submitted_requests) == 1
    assert tc.submitted_requests[0].client_order_id != prior_cid
    assert tc.submitted_requests[0].client_order_id.endswith("s1")


def test_reconcile_detects_undersized_and_orphan_stops_and_skips_clean_event(
    tmp_path: Path,
) -> None:
    """HIGH #5 & #6 & MEDIUM #8 & LOW #13/#15."""
    db = StateDB(tmp_path / "state.db")
    tc = _FakeTradingClient()
    clock = _FakeClock()

    seen_filters: list[Any] = []
    orig_get_orders = tc.get_orders

    def _capture_get_orders(filter: Any = None) -> list[_FakeOrder]:
        seen_filters.append(filter)
        return orig_get_orders(filter=filter)

    tc.get_orders = _capture_get_orders  # type: ignore[method-assign]

    # AAPL has 10 shares but only a 4-share stop (under-sized!), plus orphan stop on FLAT
    tc.positions = [_FakePosition("AAPL", "10", "200.0", "205.0", "50.0", side="long")]
    db.replace_live_positions(
        [
            {
                "symbol": "AAPL",
                "qty": 10.0,
                "avg_entry_price": 200.0,
                "market_price": 205.0,
                "unrealized_pnl": 50.0,
                "stop_order_id": "ord-under",
                "stop_price": 190.0,
            }
        ]
    )
    under_stop = _FakeOrder("ord-under", "cid-u", "AAPL", "sell", "stop", "4", "new", stop_price="190.0")
    orphan_stop = _FakeOrder("ord-orp", "cid-o", "FLAT", "sell", "stop", "5", "new", stop_price="50.0")
    for o in (under_stop, orphan_stop):
        tc.orders_by_id[o.id] = o
        tc.orders_by_cid[o.client_order_id] = o

    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
    )

    # First reconcile: missing_stop_symbols is empty, yet under-sized + orphan stops must repair!
    rec1 = broker.reconcile_with_broker(daemon_name="agy-trader")
    assert seen_filters[0].status == QueryOrderStatus.OPEN
    assert seen_filters[0].limit == 500
    assert rec1.missing_stop_symbols == []
    assert rec1.stop_sync_result is not None
    assert "ord-orp" in rec1.stop_sync_result.canceled_orphans
    assert "ord-under" in rec1.stop_sync_result.canceled_stale
    assert "AAPL" in rec1.stop_sync_result.submitted_stops
    # Preserved intended stop price (190.00) while resizing qty to 10.0
    assert tc.submitted_requests[-1].qty == pytest.approx(10.0)
    assert tc.submitted_requests[-1].stop_price == pytest.approx(190.00)
    assert len(db.get_daemon_events(event_type="reconcile")) == 1

    # Second reconcile with clean state: must NOT spam daemon_events when record_if_clean=False
    rec2 = broker.reconcile_with_broker(daemon_name="agy-trader")
    assert rec2.stop_sync_result is None
    assert len(db.get_daemon_events(event_type="reconcile")) == 1

    # Explicit record_if_clean=True (e.g. startup boot) records event
    broker.reconcile_with_broker(daemon_name="agy-trader", record_if_clean=True)
    assert len(db.get_daemon_events(event_type="reconcile")) == 2

    # LOW #15: Dust position (< 1e-5) skips stop submission cleanly
    tc.positions = [_FakePosition("DUST", "0.000001", "100.0", "100.0", "0.0", side="long")]
    dust_res = broker.sync_protective_stops({}, session_date="2026-11-20", run_id="dust")
    assert dust_res.submitted_stops == {}
    assert any("DUST" in w and "dust" in w for w in dust_res.warnings)

    # LOW #13: reconcile_with_broker enforces double-lock
    broker.settings.alpaca_paper = False
    with pytest.raises(ValueError, match="double-lock"):
        broker.reconcile_with_broker()


def test_pass1_failure_gates_pass2_and_preserves_submitted_status_on_exception(
    tmp_path: Path,
) -> None:
    """MEDIUM #9 & #10."""
    db = StateDB(tmp_path / "state.db")
    clock = _FakeClock()

    class _RejectPass1AndCrashOnPollClient(_FakeTradingClient):
        def submit_order(self, order_data: Any) -> _FakeOrder:
            order = super().submit_order(order_data)
            if order.symbol == "MSFT":
                order.status = "rejected"
            return order

    tc = _RejectPass1AndCrashOnPollClient()
    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
    )

    # Pass-1 MSFT rejection gates Pass-2 AAPL entry
    rep = broker.execute_verdicts(
        [
            _verdict("MSFT", OrderSide.sell, 5.0, 400.0, is_risk_reducing=True),
            _verdict("AAPL", OrderSide.buy, 10.0, 200.0, is_risk_reducing=False),
        ],
        session_date="2026-11-20",
        checkpoint="15:40",
        run_id="gate-run",
    )
    assert [r.symbol for r in rep.records] == ["MSFT", "AAPL"]
    assert rep.records[1].status == "skipped_pass1_gate"
    assert len(tc.submitted_requests) == 1  # AAPL never submitted!

    # MEDIUM #10: Exception raised during polling after limit order already submitted preserves
    # broker status ("accepted") in live_orders rather than clobbering to "error".
    class _CrashDuringPollClient(_FakeTradingClient):
        def submit_order(self, order_data: Any) -> _FakeOrder:
            order = super().submit_order(order_data)
            order.status = "accepted"
            return order

        def get_order_by_id(self, order_id: str) -> _FakeOrder:
            raise RuntimeError("transient network drop during poll")

    tc2 = _CrashDuringPollClient()
    broker2 = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc2,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
    )
    rep2 = broker2.execute_verdicts(
        [_verdict("NVDA", OrderSide.buy, 5.0, 120.0, is_risk_reducing=False)],
        session_date="2026-11-20",
        checkpoint="15:40",
        run_id="poll-crash",
    )
    assert rep2.records[0].status == "accepted"
    assert rep2.records[0].error is not None
    stored = db.get_live_order("agy:20261120:1540:NVDA:B:p2m0")
    assert stored is not None
    assert stored["status"] == "accepted"


def test_alpaca_broker_context_manager_closes_only_owned_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MEDIUM #7: close() closes owned SDK client sessions and leaves injected clients open."""
    db = StateDB(tmp_path / "state.db")

    class _DummySession:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _OwnedClient:
        def __init__(self, **kwargs: Any) -> None:
            self._session = _DummySession()

    created_clients: list[_OwnedClient] = []

    def _make_client(**kwargs: Any) -> _OwnedClient:
        c = _OwnedClient(**kwargs)
        created_clients.append(c)
        return c

    monkeypatch.setattr("agy_trader.execution.alpaca_broker.TradingClient", _make_client)
    monkeypatch.setattr("agy_trader.execution.alpaca_broker.StockHistoricalDataClient", _make_client)

    with AlpacaBroker(EnvSettings(alpaca_api_key="k", alpaca_secret_key="s"), db):
        assert len(created_clients) == 2
        assert all(not c._session.closed for c in created_clients)
    assert all(c._session.closed for c in created_clients)

    injected_tc = _OwnedClient()
    injected_qc = _OwnedClient()
    with AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=injected_tc,
        quote_client=injected_qc,
    ):
        pass
    assert injected_tc._session.closed is False
    assert injected_qc._session.closed is False
