from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import respx
from alpaca.trading.enums import QueryOrderStatus
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from agy_ops.daemons import MarketDataDaemon, TraderDaemon
from agy_ops.dashboard import create_dashboard_app
from agy_ops.notifications import (
    DiscordWebhookNotifier,
    build_blackout_alert_discord_embed,
    build_daily_summary_discord_embed,
    build_morning_discord_embed,
)
from agy_ops.reporting import build_daily_summary, build_morning_plan, build_readiness_report
from agy_trader.cli.main import app as trader_cli_app
from agy_trader.core.blackout import (
    expand_share_class_symbols,
    parse_blackout_period,
    resolve_live_blackout,
)
from agy_trader.core.config import EnvSettings, RiskConfig, StrategyConfig, load_universe
from agy_trader.core.models import (
    AssetRiskMetadata,
    AssetType,
    ControlDials,
    MacroPoint,
    OrderSide,
    ProposedOrder,
    RiskVerdict,
    SignalRow,
)
from agy_trader.data.macro_and_calendar import YFinanceEtfHoldingsProvider
from agy_trader.data.synthetic import generate_synthetic_dataset
from agy_trader.execution.alpaca_broker import AlpacaBroker
from agy_trader.nlp.sentiment import LexiconSentimentScorer
from agy_trader.risk.engine import HardRiskEngine, PortfolioRiskState
from agy_trader.storage.db import StateDB
from agy_trader.storage.parquet_store import ParquetBarStore
from agy_trader.strategy.portfolio import PortfolioConstructor

REPO_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
SAMPLE_WEBHOOK = "https://discord.com/api/webhooks/123456789012345678/test_token_secret_XYZ"
runner = CliRunner()


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
        self.clock_open: bool | None = True
        self._seq = 0

    def get_clock(self) -> Any:
        @dataclass
        class _Clk:
            is_open: bool
            next_close: None = None

        return _Clk(is_open=bool(self.clock_open))

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
        status_filter = getattr(filter, "status", None)
        orders = list(self.orders_by_id.values())
        if status_filter == QueryOrderStatus.OPEN:
            return [o for o in orders if o.status in open_statuses]
        return orders

    def get_order_by_client_id(self, client_id: str) -> _FakeOrder:
        if client_id not in self.orders_by_cid:
            raise _NotFoundError(client_id)
        return self.orders_by_cid[client_id]

    def get_order_by_id(self, order_id: str) -> _FakeOrder:
        if order_id not in self.orders_by_id:
            raise _NotFoundError(order_id)
        return self.orders_by_id[order_id]

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
        return order

    def cancel_order_by_id(self, order_id: str) -> None:
        self.canceled_ids.append(order_id)
        order = self.orders_by_id.get(order_id)
        if order is not None:
            order.status = "canceled"

    def cancel_orders(self) -> list[_FakeOrder]:
        self.cancel_all_called = True
        canceled: list[_FakeOrder] = []
        for order in self.orders_by_id.values():
            if order.status in {"new", "accepted", "partially_filled", "held", "stopped"}:
                order.status = "canceled"
                self.canceled_ids.append(order.id)
                canceled.append(order)
        return canceled


class _FakeSnapshotClient:
    def __init__(self, prices: dict[str, float], now_utc: datetime) -> None:
        self.prices = prices
        self.now_utc = now_utc

    def get_stock_snapshot(self, req: Any) -> dict[str, Any]:
        syms = list(getattr(req, "symbol_or_symbols", []))
        out: dict[str, Any] = {}
        for sym in syms:
            px = self.prices.get(sym, 100.0)
            out[sym] = {
                "latest_trade": {"price": px, "timestamp": self.now_utc},
                "daily_bar": {
                    "open": px * 0.99,
                    "high": px * 1.01,
                    "low": px * 0.98,
                    "close": px,
                    "volume": 1_000_000.0,
                    "vwap": px,
                    "timestamp": self.now_utc,
                },
            }
        return out


def _seed_etf_holdings(
    db: StateDB,
    *,
    now_utc: datetime,
    xlc_googl: float = 0.1108,
    xlc_goog: float = 0.0892,
) -> None:
    universe = load_universe(REPO_CONFIG_DIR)
    for entry in universe:
        if entry.asset_type != AssetType.etf:
            continue
        if entry.sector in {"Fixed Income", "Commodities"}:
            db.upsert_etf_holdings(entry.symbol, {}, is_non_equity_etf=True, updated_at=now_utc)
        elif entry.symbol == "XLC":
            db.upsert_etf_holdings(
                "XLC",
                {
                    "META": 0.2200,
                    "GOOGL": xlc_googl,
                    "GOOG": xlc_goog,
                    "NFLX": 0.0600,
                    "DIS": 0.0500,
                    "TMUS": 0.0480,
                    "VZ": 0.0420,
                    "T": 0.0410,
                    "CMCSA": 0.0400,
                    "EA": 0.0350,
                },
                is_non_equity_etf=False,
                updated_at=now_utc,
            )
        elif entry.symbol == "SPY":
            db.upsert_etf_holdings(
                "SPY",
                {
                    "AAPL": 0.0770,
                    "NVDA": 0.0758,
                    "MSFT": 0.0593,
                    "AMZN": 0.0361,
                    "GOOGL": 0.0251,
                    "AVGO": 0.0238,
                    "META": 0.0208,
                    "GOOG": 0.0202,
                    "LLY": 0.0174,
                    "TSLA": 0.0173,
                },
                is_non_equity_etf=False,
                updated_at=now_utc,
            )
        elif entry.symbol == "QQQ":
            db.upsert_etf_holdings(
                "QQQ",
                {
                    "AAPL": 0.0816,
                    "NVDA": 0.0800,
                    "MSFT": 0.0629,
                    "AVGO": 0.0500,
                    "AMZN": 0.0480,
                    "META": 0.0420,
                    "TSLA": 0.0350,
                    "COST": 0.0280,
                    "NFLX": 0.0270,
                    "GOOG": 0.0266,  # GOOGL absent from top 10
                },
                is_non_equity_etf=False,
                updated_at=now_utc,
            )
        else:
            db.upsert_etf_holdings(
                entry.symbol,
                {f"H{i}": 0.05 for i in range(1, 11)},
                is_non_equity_etf=False,
                updated_at=now_utc,
            )


def _make_signal(
    symbol: str,
    as_of: datetime,
    *,
    score: float = 0.70,
    atr_pct: float = 0.02,
) -> SignalRow:
    return SignalRow(
        as_of=as_of,
        symbol=symbol,
        trend_score=score,
        pullback_score=0.0,
        news_score=0.0,
        news_velocity_z=0.0,
        composite_score=score,
        macro_regime_scale=1.0,
        atr_pct=atr_pct,
        ann_vol=0.25,
        ret_5d=0.02,
    )


def _make_universe_meta() -> dict[str, AssetRiskMetadata]:
    return {
        u.symbol: AssetRiskMetadata(
            symbol=u.symbol,
            asset_type=u.asset_type,
            sector=u.sector,
            shortable=u.shortable_default,
            easy_to_borrow=u.shortable_default,
        )
        for u in load_universe(REPO_CONFIG_DIR)
    }


def test_parse_share_class_and_fail_closed_validation(tmp_path: Path) -> None:
    assert parse_blackout_period("") == ()
    assert parse_blackout_period("   ") == ()
    assert parse_blackout_period('""') == ()
    assert parse_blackout_period("goog") == ("GOOG",)
    assert parse_blackout_period("GOOG,ACME") == ("GOOG", "ACME")
    assert parse_blackout_period("ACME, GOOGL  brk.b") == ("ACME", "GOOGL", "BRK.B")

    assert expand_share_class_symbols(("GOOG",)) == frozenset({"GOOG", "GOOGL"})
    assert expand_share_class_symbols(("GOOGL",)) == frozenset({"GOOG", "GOOGL"})
    assert expand_share_class_symbols(("BRK.B",)) == frozenset({"BRK.A", "BRK.B", "BRK-A", "BRK-B"})
    assert expand_share_class_symbols(("FOX",)) == frozenset({"FOXA", "FOX"})
    assert expand_share_class_symbols(("ACME",)) == frozenset({"ACME"})

    # Invalid tokens & malformed .env syntax MUST fail closed (halted=True)
    for bad in ("GOOG;DROP", "GOOG;ACME", "$GOOG", ",", '"GOOG'):
        env_p = tmp_path / "bad.env"
        env_p.write_text(f"BLACKOUT_PERIOD={bad}\n", encoding="utf-8")
        res = resolve_live_blackout(env_file=env_p, require_env_file=True)
        assert res.halted is True, bad
        assert res.halt_reason

    # Missing required .env also fails closed
    missing_res = resolve_live_blackout(env_file=tmp_path / "missing.env", require_env_file=True)
    assert missing_res.halted is True


def test_etf_concentration_and_staleness_rules(tmp_path: Path) -> None:
    now = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)
    db = StateDB(tmp_path / "state.db")
    universe = load_universe(REPO_CONFIG_DIR)
    env_p = tmp_path / ".env"

    # 1) BLACKOUT_PERIOD="GOOG": freezes GOOG+GOOGL, XLC stays allowed at 20.0%, SPY/QQQ/TLT/GLD allowed
    _seed_etf_holdings(db, now_utc=now, xlc_googl=0.1108, xlc_goog=0.0892)
    env_p.write_text('BLACKOUT_PERIOD="GOOG"\n', encoding="utf-8")
    res = resolve_live_blackout(env_file=env_p, require_env_file=True, db=db, universe=universe, now_utc=now)
    assert res.halted is False
    assert res.has_warnings is False
    assert res.frozen_symbols == frozenset({"GOOG", "GOOGL"})
    assert "XLC" not in res.frozen_symbols
    assert "SPY" not in res.frozen_symbols
    assert "QQQ" not in res.frozen_symbols
    assert "TLT" not in res.frozen_symbols
    assert "GLD" not in res.frozen_symbols
    assert any("XLC: 20.0% GOOG/GOOGL" in s for s in res.etf_weights_summary)

    # 2) Drift test: if XLC drifts to 24.0% (>= 23.5%), XLC is FROZEN
    _seed_etf_holdings(db, now_utc=now, xlc_googl=0.1300, xlc_goog=0.1100)
    res_drift = resolve_live_blackout(
        env_file=env_p, require_env_file=True, db=db, universe=universe, now_utc=now
    )
    assert "XLC" in res_drift.frozen_symbols
    assert "XLC" in res_drift.etf_frozen_symbols

    # 3) Conservative missing-class bound: GOOGL=16% and #10 holding=8% (sum=92%, unaccounted=8%), GOOG absent -> 16+8=24% >= 23.5% -> frozen
    db.upsert_etf_holdings(
        "XLC",
        {
            "GOOGL": 0.1600,
            "META": 0.1200,
            "NFLX": 0.0800,
            "DIS": 0.0800,
            "TMUS": 0.0800,
            "VZ": 0.0800,
            "T": 0.0800,
            "CMCSA": 0.0800,
            "EA": 0.0800,
            "TTWO": 0.0800,
        },
        is_non_equity_etf=False,
        updated_at=now,
    )
    res_bound = resolve_live_blackout(
        env_file=env_p, require_env_file=True, db=db, universe=universe, now_utc=now
    )
    assert "XLC" in res_bound.frozen_symbols

    # 4) Stale equity ETF (> 7 days) fails closed on that ETF with has_warnings=True, while TLT/GLD stay allowed
    _seed_etf_holdings(db, now_utc=now)
    db.upsert_etf_holdings(
        "XLK",
        {f"H{i}": 0.05 for i in range(1, 11)},
        is_non_equity_etf=False,
        updated_at=now - timedelta(days=8),
    )
    res_stale = resolve_live_blackout(
        env_file=env_p, require_env_file=True, db=db, universe=universe, now_utc=now
    )
    assert res_stale.has_warnings is True
    assert "XLK" in res_stale.frozen_symbols
    assert "TLT" not in res_stale.frozen_symbols
    assert "GLD" not in res_stale.frozen_symbols

    # 5) Missing equity ETF metadata fails closed on that ETF
    db2 = StateDB(tmp_path / "missing_etf.db")
    res_missing = resolve_live_blackout(
        env_file=env_p, require_env_file=True, db=db2, universe=universe, now_utc=now
    )
    assert res_missing.has_warnings is True
    assert "XLC" in res_missing.frozen_symbols
    assert "TLT" not in res_missing.frozen_symbols
    assert "GLD" not in res_missing.frozen_symbols

    # 6) ACME (not in universe.yaml) freezes ACME while leaving XLC/TLT/GLD allowed when fresh
    _seed_etf_holdings(db, now_utc=now)
    env_p.write_text('BLACKOUT_PERIOD="ACME, GOOG"\n', encoding="utf-8")
    res_acme = resolve_live_blackout(
        env_file=env_p, require_env_file=True, db=db, universe=universe, now_utc=now
    )
    assert res_acme.frozen_symbols == frozenset({"ACME", "GOOG", "GOOGL"})
    assert "XLC" not in res_acme.frozen_symbols


def test_yfinance_etf_holdings_provider_and_market_data_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeFundsData:
        def __init__(self, df: pd.DataFrame) -> None:
            self.top_holdings = df

    class _FakeTicker:
        def __init__(self, sym: str) -> None:
            if sym == "XLC":
                self.funds_data = _FakeFundsData(
                    pd.DataFrame(
                        {"Name": ["Meta", "Alphabet A", "Alphabet C"], "Holding Percent": [22.0, 11.08, 8.92]},
                        index=pd.Index(["META", "GOOGL", "GOOG"], name="Symbol"),
                    )
                )
            elif sym in {"TLT", "GLD"}:
                self.funds_data = _FakeFundsData(pd.DataFrame(columns=["Name", "Holding Percent"]))
            elif sym == "XLE":
                raise RuntimeError("transient error")
            else:
                self.funds_data = _FakeFundsData(
                    pd.DataFrame(
                        {"Name": ["Apple", "Microsoft"], "Holding Percent": [0.08, 0.07]},
                        index=pd.Index(["AAPL", "MSFT"], name="Symbol"),
                    )
                )

    monkeypatch.setattr("agy_trader.data.macro_and_calendar.yf.Ticker", _FakeTicker)

    prov = YFinanceEtfHoldingsProvider()
    fetched = prov.fetch_etf_holdings(["XLC", "SPY", "TLT", "GLD", "XLE"])
    assert pytest.approx(fetched["XLC"]["GOOGL"], abs=1e-6) == 0.1108
    assert pytest.approx(fetched["XLC"]["GOOG"], abs=1e-6) == 0.0892
    assert pytest.approx(fetched["SPY"]["AAPL"], abs=1e-6) == 0.08
    assert fetched["TLT"] == {}
    assert fetched["GLD"] == {}
    assert "XLE" not in fetched

    db = StateDB(tmp_path / "state.db")
    daemon = MarketDataDaemon(
        EnvSettings(agy_data_dir=tmp_path, agy_config_dir=REPO_CONFIG_DIR),
        tmp_path,
        REPO_CONFIG_DIR,
        db=db,
        etf_holdings_provider=prov,
    )
    now = datetime(2026, 11, 20, 21, 10, tzinfo=UTC)
    aux = daemon._refresh_auxiliary_data(["SPY", "XLC", "TLT", "GLD"], now, lookback_days=7)
    assert aux["etf_holdings_saved"] >= 4
    tlt_meta = db.get_etf_holdings_meta("TLT")
    assert tlt_meta is not None
    assert tlt_meta["is_non_equity_etf"] is True
    assert tlt_meta["holdings_count"] == 0
    xlc_holdings = db.get_etf_holdings("XLC")
    assert pytest.approx(xlc_holdings["GOOGL"] + xlc_holdings["GOOG"], abs=1e-6) == 0.20


def test_portfolio_constructor_locks_frozen_positions_and_respects_headroom() -> None:
    as_of = datetime(2026, 11, 20, 20, 40, tzinfo=UTC)
    meta = _make_universe_meta()
    ctor = PortfolioConstructor(
        StrategyConfig(top_k_long=4, top_k_short=0, waterfill_weights=True),
        RiskConfig(max_long_position_pct=0.10, max_sector_gross_pct=0.25, max_gross_exposure=1.0),
    )

    signals = [
        _make_signal("GOOGL", as_of, score=-0.80),  # Bearish! Must STILL NOT be sold or trimmed!
        _make_signal("META", as_of, score=0.90),
        _make_signal("DIS", as_of, score=0.85),
        _make_signal("NFLX", as_of, score=0.80),
        _make_signal("AAPL", as_of, score=0.75),
    ]
    prices = {
        "GOOGL": 180.0,
        "META": 500.0,
        "DIS": 100.0,
        "NFLX": 700.0,
        "AAPL": 220.0,
        "ACME": 500.0,
    }
    # Held: GOOGL (10% of $100k = $10k) and ACME (5% non-universe)
    current_positions = {"GOOGL": 10_000.0 / 180.0, "ACME": 5_000.0 / 500.0}
    frozen = frozenset({"GOOG", "GOOGL", "ACME"})

    weights = ctor.compute_target_weights(
        signals=signals,
        dials=ControlDials(pause_new_entries=True, exposure_multiplier=0.5),
        as_of=as_of,
        universe_meta=meta,
        current_weights={"GOOGL": 0.10, "ACME": 0.05},
        frozen_symbols=frozen,
    )
    assert pytest.approx(weights["GOOGL"], abs=1e-9) == 0.10
    assert pytest.approx(weights["ACME"], abs=1e-9) == 0.05

    # Check Communication Services headroom (GOOGL takes 0.10 of 0.25 cap, non-frozen <= 0.15)
    weights_active = ctor.compute_target_weights(
        signals=signals,
        dials=ControlDials(),
        as_of=as_of,
        universe_meta=meta,
        current_weights={"GOOGL": 0.10, "ACME": 0.05},
        frozen_symbols=frozen,
    )
    assert pytest.approx(weights_active["GOOGL"], abs=1e-9) == 0.10
    comm_gross = sum(
        abs(w) for sym, w in weights_active.items() if sym in meta and meta[sym].sector == "Communication Services"
    )
    assert comm_gross <= 0.25 + 1e-9
    assert sum(abs(w) for w in weights_active.values()) <= 1.0 + 1e-9

    orders = ctor.build_proposed_orders(
        signals=signals,
        current_positions=current_positions,
        prices=prices,
        equity=100_000.0,
        dials=ControlDials(),
        as_of=as_of,
        universe_meta=meta,
        frozen_symbols=frozen,
    )
    order_syms = {o.symbol for o in orders}
    assert "GOOGL" not in order_syms
    assert "GOOG" not in order_syms
    assert "ACME" not in order_syms
    assert "META" in order_syms


def test_hard_risk_engine_blocks_both_pass1_and_pass2_and_halt() -> None:
    as_of = datetime(2026, 11, 20, 20, 40, tzinfo=UTC)
    meta = _make_universe_meta()
    engine = HardRiskEngine(RiskConfig())
    state = PortfolioRiskState(
        as_of=as_of,
        equity=100_000.0,
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
        positions={"GOOGL": 50.0},
        prices={"GOOGL": 180.0, "AAPL": 220.0},
        latest_bar_age_minutes=1.0,
    )
    signals = {
        "GOOGL": _make_signal("GOOGL", as_of),
        "AAPL": _make_signal("AAPL", as_of),
    }

    orders = [
        ProposedOrder(
            symbol="GOOGL",
            side=OrderSide.sell,
            qty=50.0,
            is_risk_reducing=True,
            reference_price=180.0,
            reason="exit_to_flat",
        ),
        ProposedOrder(
            symbol="GOOGL",
            side=OrderSide.buy,
            qty=10.0,
            is_risk_reducing=False,
            reference_price=180.0,
            stop_loss_price=170.0,
            reason="add",
        ),
        ProposedOrder(
            symbol="AAPL",
            side=OrderSide.buy,
            qty=20.0,
            is_risk_reducing=False,
            reference_price=220.0,
            stop_loss_price=210.0,
            reason="entry",
        ),
    ]

    verdicts = engine.evaluate_batch(
        orders=orders,
        state=state,
        dials=ControlDials(),
        universe_meta=meta,
        signals_by_symbol=signals,
        frozen_symbols={"GOOG", "GOOGL"},
        blackout_halted=False,
    )
    assert verdicts[0].approved is False
    assert verdicts[0].adjusted_qty == 0.0
    assert verdicts[0].violations == ["BLACKOUT_PERIOD_FROZEN"]
    assert verdicts[1].approved is False
    assert verdicts[1].violations == ["BLACKOUT_PERIOD_FROZEN"]
    assert verdicts[2].approved is True

    # Fail-closed halt blocks ALL orders across the portfolio
    halted_verdicts = engine.evaluate_batch(
        orders=orders,
        state=state,
        dials=ControlDials(),
        universe_meta=meta,
        signals_by_symbol=signals,
        frozen_symbols=set(),
        blackout_halted=True,
    )
    for v in halted_verdicts:
        assert v.approved is False
        assert v.adjusted_qty == 0.0
        assert v.violations == ["BLACKOUT_PERIOD_HALT"]


def test_alpaca_broker_leaves_frozen_positions_and_resting_stops_untouched(tmp_path: Path) -> None:
    clock = _FakeClock()
    tc = _FakeTradingClient()
    db = StateDB(tmp_path / "state.db")
    now = clock.now()
    _seed_etf_holdings(db, now_utc=now)
    env_p = tmp_path / ".env"
    env_p.write_text('BLACKOUT_PERIOD="GOOG"\n', encoding="utf-8")

    # Seed broker positions: GOOGL (50 sh) + AAPL (20 sh), each with a resting GTC sell stop
    tc.positions = [
        _FakePosition("GOOGL", "50", "175.00", "180.00", "250.00"),
        _FakePosition("AAPL", "20", "210.00", "220.00", "200.00"),
    ]
    googl_stop = _FakeOrder(
        id="stp-googl-1",
        client_order_id="manual-googl-stop",
        symbol="GOOGL",
        side="sell",
        order_type="stop",
        qty="50",
        status="new",
        stop_price="165.00",
    )
    aapl_stop = _FakeOrder(
        id="stp-aapl-1",
        client_order_id="agy-aapl-stop",
        symbol="AAPL",
        side="sell",
        order_type="stop",
        qty="20",
        status="new",
        stop_price="200.00",
    )
    for o in (googl_stop, aapl_stop):
        tc.orders_by_id[o.id] = o
        tc.orders_by_cid[o.client_order_id] = o

    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
        env_file=env_p,
        config_dir=REPO_CONFIG_DIR,
    )

    # 1a) Last-millisecond Pass-1 blackout block: GOOGL Pass-1 exit is blocked_blackout AND
    # safely gates Pass-2 MSFT (preventing entries sized on unfreed Pass-1 margin/headroom).
    v_googl_sell = RiskVerdict(
        approved=True,
        adjusted_qty=50.0,
        order=ProposedOrder(
            symbol="GOOGL",
            side=OrderSide.sell,
            qty=50.0,
            is_risk_reducing=True,
            reference_price=180.0,
        ),
    )
    v_msft_buy = RiskVerdict(
        approved=True,
        adjusted_qty=10.0,
        stop_loss_price=400.0,
        order=ProposedOrder(
            symbol="MSFT",
            side=OrderSide.buy,
            qty=10.0,
            is_risk_reducing=False,
            reference_price=420.0,
            stop_loss_price=400.0,
        ),
    )
    rep_gate = broker.execute_verdicts(
        [v_googl_sell, v_msft_buy],
        session_date="2026-11-20",
        checkpoint="1535",
        run_id="r-exec-gate",
    )
    googl_gate_rec = next(r for r in rep_gate.records if r.symbol == "GOOGL")
    msft_gate_rec = next(r for r in rep_gate.records if r.symbol == "MSFT")
    assert googl_gate_rec.status == "blocked_blackout"
    assert msft_gate_rec.status == "skipped_pass1_gate"
    assert len(tc.submitted_requests) == 0

    # 1b) Steady-state blackout (Pass-1 GOOGL already rejected by HardRiskEngine, plus a Pass-2
    # GOOGL buy): GOOGL Pass-2 is blocked_blackout while MSFT Pass-2 executes and fills cleanly.
    v_googl_sell_rejected = RiskVerdict(
        approved=False,
        adjusted_qty=0.0,
        violations=["BLACKOUT_PERIOD_FROZEN"],
        order=v_googl_sell.order,
    )
    v_googl_buy = RiskVerdict(
        approved=True,
        adjusted_qty=5.0,
        stop_loss_price=165.0,
        order=ProposedOrder(
            symbol="GOOGL",
            side=OrderSide.buy,
            qty=5.0,
            is_risk_reducing=False,
            reference_price=180.0,
            stop_loss_price=165.0,
        ),
    )
    rep = broker.execute_verdicts(
        [v_googl_sell_rejected, v_googl_buy, v_msft_buy],
        session_date="2026-11-20",
        checkpoint="1540",
        run_id="r-exec",
    )
    googl_rec = next(r for r in rep.records if r.symbol == "GOOGL")
    msft_rec = next(r for r in rep.records if r.symbol == "MSFT")
    assert googl_rec.status == "blocked_blackout"
    assert msft_rec.status == "filled"
    assert all(getattr(req, "symbol", "") != "GOOGL" for req in tc.submitted_requests)

    # 2) sync_protective_stops even with tighter desired stop for GOOGL must leave stp-googl-1 UNTOUCHED
    tc.submitted_requests.clear()
    tc.canceled_ids.clear()
    sync_res = broker.sync_protective_stops(
        {"GOOGL": 175.0, "AAPL": 215.0},  # Both tighter than existing stops!
        session_date="2026-11-20",
        run_id="r-sync",
    )
    assert "stp-googl-1" not in tc.canceled_ids
    assert tc.orders_by_id["stp-googl-1"].status == "new"
    assert "GOOGL" in sync_res.untouched_blackout_positions
    assert "stp-googl-1" in sync_res.untouched_blackout_orders
    assert "stp-aapl-1" in tc.canceled_ids  # AAPL stop WAS tightened normally!
    assert "AAPL" in sync_res.submitted_stops

    # 3) reconcile_with_broker also leaves GOOGL untouched
    tc.canceled_ids.clear()
    recon = broker.reconcile_with_broker(daemon_name="trader")
    assert "stp-googl-1" not in tc.canceled_ids
    assert tc.orders_by_id["stp-googl-1"].status == "new"
    assert recon.untouched_blackout_positions == ["GOOGL"]
    assert "stp-googl-1" in recon.untouched_blackout_orders

    # 4) emergency_flatten skips GOOGL & leaves stp-googl-1 untouched unless override_blackout=True
    tc.submitted_requests.clear()
    tc.canceled_ids.clear()
    flat_res = broker.emergency_flatten(confirm=True, reason="risk test")
    assert tc.cancel_all_called is False
    assert "stp-googl-1" not in tc.canceled_ids
    assert tc.orders_by_id["stp-googl-1"].status == "new"
    assert flat_res["skipped_blackout_symbols"] == ["GOOGL"]
    assert "stp-googl-1" in flat_res["untouched_blackout_orders"]
    assert all(c["symbol"] != "GOOGL" for c in flat_res["close_orders"])
    assert any(c["symbol"] == "AAPL" for c in flat_res["close_orders"])

    # With override_blackout=True, GOOGL stop IS canceled and GOOGL IS closed
    flat_override = broker.emergency_flatten(
        confirm=True, reason="legal cleared", override_blackout=True
    )
    assert tc.cancel_all_called is True
    assert tc.orders_by_id["stp-googl-1"].status == "canceled"
    assert any(c["symbol"] == "GOOGL" for c in flat_override["close_orders"])


def test_frozen_blackout_hybrid_transition_preserves_software_stop_locally(
    tmp_path: Path,
) -> None:
    """HIGH #7: Frozen 5.75->0.75 transition preserves stop_order_id='software' and stop_price=95.00."""
    clock = _FakeClock()
    tc = _FakeTradingClient()
    db = StateDB(tmp_path / "state.db")
    now = clock.now()
    _seed_etf_holdings(db, now_utc=now)
    env_p = tmp_path / ".env"
    env_p.write_text('BLACKOUT_PERIOD="GOOG"\n', encoding="utf-8")

    # Pre-existing frozen GOOGL hybrid position: 5.75 shares, 5-share GTC stop filled -> 0.75 remains
    db.replace_live_positions(
        [
            {
                "symbol": "GOOGL",
                "qty": 5.75,
                "avg_entry_price": 100.0,
                "market_price": 94.5,
                "unrealized_pnl": -31.62,
                "stop_order_id": "stp-googl-5",
                "stop_price": 95.0,
                "updated_at": now,
            }
        ]
    )
    tc.positions = [_FakePosition("GOOGL", "0.75", "100.00", "94.50", "-4.12", side="long")]

    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
        env_file=env_p,
        config_dir=REPO_CONFIG_DIR,
    )

    rec = broker.reconcile_with_broker(daemon_name="trader", record_if_clean=False)
    assert rec.untouched_blackout_positions == ["GOOGL"]
    assert tc.submitted_requests == []
    assert tc.canceled_ids == []
    live1 = {p["symbol"]: p for p in db.get_live_positions()}
    assert live1["GOOGL"]["stop_order_id"] == "software"
    assert live1["GOOGL"]["stop_price"] == pytest.approx(95.00)

    sync_res = broker.sync_protective_stops({}, session_date="2026-11-20", run_id="frozen-sync")
    assert "GOOGL" in sync_res.untouched_blackout_positions
    assert tc.submitted_requests == []
    assert tc.canceled_ids == []
    live2 = {p["symbol"]: p for p in db.get_live_positions()}
    assert live2["GOOGL"]["stop_order_id"] == "software"
    assert live2["GOOGL"]["stop_price"] == pytest.approx(95.00)


def test_trader_daemon_hot_reload_and_deduplicated_discord_alert(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    env_p = tmp_path / ".env"
    env_p.write_text('BLACKOUT_PERIOD=""\n', encoding="utf-8")

    db = StateDB(data_dir / "state.db")
    bar_store = ParquetBarStore(data_dir / "parquet")
    universe = load_universe(REPO_CONFIG_DIR)
    generate_synthetic_dataset(universe=universe, bar_store=bar_store, db=db, days=320, seed=19)

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

    # Friday 2026-11-20 15:40 ET == 20:40 UTC
    rebal_utc = datetime(2026, 11, 20, 20, 40, tzinfo=UTC)
    db.upsert_macro(
        [
            MacroPoint(as_of_date=rebal_utc.date(), series_id="DGS3MO", value=4.25),
            MacroPoint(as_of_date=rebal_utc.date(), series_id="VIXCLS", value=16.0),
        ]
    )
    _seed_etf_holdings(db, now_utc=rebal_utc)

    tc = _FakeTradingClient()
    tc.positions = [_FakePosition("GOOGL", "25", "175.00", "180.00", "125.00")]
    googl_stop = _FakeOrder(
        id="stp-googl-live",
        client_order_id="manual-googl-live",
        symbol="GOOGL",
        side="sell",
        order_type="stop",
        qty="25",
        status="new",
        stop_price="165.00",
    )
    tc.orders_by_id[googl_stop.id] = googl_stop
    tc.orders_by_cid[googl_stop.client_order_id] = googl_stop

    prices = {u.symbol: 150.0 for u in universe}
    prices["GOOGL"] = 180.0
    snap_client = _FakeSnapshotClient(prices=prices, now_utc=rebal_utc)

    settings = EnvSettings(
        alpaca_paper=True,
        agy_data_dir=data_dir,
        agy_config_dir=REPO_CONFIG_DIR,
        discord_webhook_url=SAMPLE_WEBHOOK,
        nlp_backend="lexicon",
    )
    broker = AlpacaBroker(
        settings,
        db,
        trading_client=tc,
        quote_client=snap_client,
        env_file=env_p,
        config_dir=REPO_CONFIG_DIR,
        now_fn=lambda: rebal_utc,
    )

    with respx.mock(assert_all_called=False) as router:
        route = router.post(SAMPLE_WEBHOOK).mock(
            return_value=httpx.Response(204)
        )
        notifier = DiscordWebhookNotifier(SAMPLE_WEBHOOK, max_retries=0)
        daemon = TraderDaemon(
            settings,
            data_dir,
            REPO_CONFIG_DIR,
            db=db,
            bar_store=bar_store,
            broker=broker,
            snapshot_client=snap_client,
            network_check_fn=lambda: True,
            env_file=env_p,
            discord_notifier=notifier,
        )

        # Tick 1: BLACKOUT_PERIOD="" (before 15:40 ET rebalance window so rebalance hasn't run yet)
        pre_utc = datetime(2026, 11, 20, 20, 35, tzinfo=UTC)
        res1 = daemon.run_once(now_utc=pre_utc)
        assert res1["blackout"]["active"] is False
        assert route.call_count == 0

        # Hot-reload: edit .env on disk mid-run WITHOUT restarting TraderDaemon!
        env_p.write_text('BLACKOUT_PERIOD="GOOG"\n', encoding="utf-8")

        # Tick 2: 15:40 ET rebalance checkpoint runs with hot-reloaded BLACKOUT_PERIOD="GOOG"
        res2 = daemon.run_once(now_utc=rebal_utc)
        assert res2["blackout"]["active"] is True
        assert sorted(res2["blackout"]["frozen_symbols"]) == ["GOOG", "GOOGL"]
        assert res2["rebalance"] is not None
        assert all(getattr(req, "symbol", "") not in {"GOOG", "GOOGL"} for req in tc.submitted_requests)
        assert "stp-googl-live" not in tc.canceled_ids
        assert tc.orders_by_id["stp-googl-live"].status == "new"
        assert route.call_count == 1
        sent_payload = json.loads(route.calls[0].request.content.decode("utf-8"))
        assert "untouched" in sent_payload["embeds"][0]["description"].lower()

        # Tick 3: identical blackout state must NOT spam Discord again
        res3 = daemon.run_once(now_utc=rebal_utc + timedelta(seconds=30))
        assert res3["blackout"]["active"] is True
        assert route.call_count == 1

        # Tick 4: malformed .env mid-run fails closed (halted=True) & triggers halt alert
        env_p.write_text('BLACKOUT_PERIOD="GOOG;DROP"\n', encoding="utf-8")
        res4 = daemon.run_once(now_utc=rebal_utc + timedelta(seconds=60))
        assert res4["blackout"]["halted"] is True
        assert route.call_count == 2
        halt_payload = json.loads(route.calls[1].request.content.decode("utf-8"))
        assert "HALT" in halt_payload["embeds"][0]["title"]
        notifier.close()


def test_cli_reporting_discord_and_dashboard_surface_blackout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    now = datetime(2026, 11, 20, 14, 0, tzinfo=UTC)
    db = StateDB(data_dir / "state.db")
    bar_store = ParquetBarStore(data_dir / "parquet")
    universe = load_universe(REPO_CONFIG_DIR)
    generate_synthetic_dataset(universe=universe, bar_store=bar_store, db=db, days=320, seed=5)
    _seed_etf_holdings(db, now_utc=now)
    db.insert_portfolio_snapshot(
        timestamp=now,
        equity=100_000.0,
        cash=50_000.0,
        buying_power=150_000.0,
        gross_exposure=0.5,
        net_exposure=0.5,
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
        high_water_mark=100_000.0,
    )
    (tmp_path / ".env").write_text('BLACKOUT_PERIOD="GOOG"\n', encoding="utf-8")

    # 1) agy status --json
    res_status = runner.invoke(
        trader_cli_app,
        ["--data-dir", str(data_dir), "--config-dir", str(REPO_CONFIG_DIR), "status", "--json"],
    )
    assert res_status.exit_code == 0, res_status.output
    status_payload = json.loads(res_status.stdout)
    assert status_payload["blackout"]["active"] is True
    assert status_payload["blackout"]["frozen_symbols"] == ["GOOG", "GOOGL"]
    assert any("XLC: 20.0% GOOG/GOOGL" in s for s in status_payload["blackout"]["etf_weights_summary"])

    # 2) agy-ops reporting digests + Discord embeds
    readiness = build_readiness_report(data_dir, REPO_CONFIG_DIR, now_utc=now)
    morning = build_morning_plan(data_dir, REPO_CONFIG_DIR, now_utc=now)
    summary = build_daily_summary(data_dir, REPO_CONFIG_DIR, now_utc=now)
    assert readiness["blackout"]["frozen_symbols"] == ["GOOG", "GOOGL"]
    assert morning["blackout"]["frozen_symbols"] == ["GOOG", "GOOGL"]
    assert summary["blackout"]["frozen_symbols"] == ["GOOG", "GOOGL"]
    assert "BLACKOUT_PERIOD" in morning["markdown_summary"]
    assert "BLACKOUT_PERIOD" in summary["markdown_summary"]

    morning_embed = build_morning_discord_embed(readiness, morning)
    summary_embed = build_daily_summary_discord_embed(summary)
    alert_embed = build_blackout_alert_discord_embed(
        readiness["blackout"],
        affected_positions=["GOOGL"],
        affected_orders=["stp-1"],
    )
    assert any(f["name"] == "BLACKOUT_PERIOD Compliance Rail" for f in morning_embed["fields"])
    assert any(f["name"] == "BLACKOUT_PERIOD Compliance Rail" for f in summary_embed["fields"])
    assert "untouched" in alert_embed["description"].lower()

    # 3) FastAPI dashboard /api/overview
    dash_app = create_dashboard_app(data_dir, REPO_CONFIG_DIR)
    client = TestClient(dash_app, client=("127.0.0.1", 50000))
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    ov = resp.json()
    assert ov["blackout"]["active"] is True
    assert ov["blackout"]["frozen_symbols"] == ["GOOG", "GOOGL"]
    html = client.get("/").text
    assert "blackout-banner" in html


def test_critical_1_out_of_universe_broker_symbol_fails_closed(tmp_path: Path) -> None:
    """CRITICAL #1: Held/open-order out-of-universe symbol (e.g. VOX) freezes during blackout."""
    clock = _FakeClock()
    now = clock.now()
    db = StateDB(tmp_path / "state.db")
    _seed_etf_holdings(db, now_utc=now)
    env_p = tmp_path / ".env"
    env_p.write_text('BLACKOUT_PERIOD="GOOG"\n', encoding="utf-8")

    tc = _FakeTradingClient()
    tc.positions = [
        _FakePosition("VOX", "30", "130.00", "135.00", "150.00"),
        _FakePosition("AAPL", "10", "210.00", "220.00", "100.00"),
    ]
    vox_stop = _FakeOrder(
        id="stp-vox-1",
        client_order_id="manual-vox-stop",
        symbol="VOX",
        side="sell",
        order_type="stop",
        qty="30",
        status="new",
        stop_price="120.00",
    )
    tc.orders_by_id[vox_stop.id] = vox_stop
    tc.orders_by_cid[vox_stop.client_order_id] = vox_stop

    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
        env_file=env_p,
        config_dir=REPO_CONFIG_DIR,
    )

    res_direct = broker.resolve_blackout_state(now_utc=now, observed_broker_symbols={"VOX"})
    assert res_direct.has_warnings is True
    assert "VOX" in res_direct.frozen_symbols
    assert any("out_of_universe_unverified_during_blackout" in w for w in res_direct.warnings)

    sync_res = broker.sync_protective_stops(
        {"VOX": 130.0, "AAPL": 205.0}, session_date="2026-11-20", run_id="r-vox"
    )
    assert "VOX" in sync_res.untouched_blackout_positions
    assert "stp-vox-1" in sync_res.untouched_blackout_orders
    assert "stp-vox-1" not in tc.canceled_ids
    assert tc.orders_by_id["stp-vox-1"].status == "new"
    assert "AAPL" in sync_res.submitted_stops

    flat_res = broker.emergency_flatten(confirm=True, reason="vox rail test")
    assert "VOX" in flat_res["skipped_blackout_symbols"]
    assert "stp-vox-1" in flat_res["untouched_blackout_orders"]
    assert tc.orders_by_id["stp-vox-1"].status == "new"
    assert any(c["symbol"] == "AAPL" for c in flat_res["close_orders"])


def test_critical_2_share_class_hyphen_and_yfinance_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CRITICAL #2: Dot/hyphen share-class canonicalization, RangeIndex, IWM scaling, & weight sum checks."""
    assert {"BRK.A", "BRK.B", "BRK-A", "BRK-B"}.issubset(expand_share_class_symbols(("BRK.B",)))
    assert {"BF.A", "BF.B", "BF-A", "BF-B"}.issubset(expand_share_class_symbols(("BF-A",)))

    now = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)
    db = StateDB(tmp_path / "state.db")
    _seed_etf_holdings(db, now_utc=now)
    # BRK-B at 16% + 10th holding at 7% -> canonical missing count is 1 (BRK.A), conservative = 23% < 23.5%
    # (Before fix, 4-element group counted 3 missing classes -> 16% + 3*7% = 37%!)
    brk_holdings = {"BRK-B": 0.16, **{f"H{i}": 0.07 for i in range(1, 10)}}
    db.upsert_etf_holdings("XLF", brk_holdings, is_non_equity_etf=False, updated_at=now)
    env_p = tmp_path / ".env"
    env_p.write_text('BLACKOUT_PERIOD="BRK.B"\n', encoding="utf-8")
    res_brk = resolve_live_blackout(
        env_file=env_p,
        require_env_file=True,
        db=db,
        universe=load_universe(REPO_CONFIG_DIR),
        now_utc=now,
    )
    assert pytest.approx(res_brk.etf_details["XLF"].conservative_weight, abs=1e-6) == 0.23
    assert "XLF" not in res_brk.frozen_symbols

    # Corrupt weight sum > 1.05 rejected by StateDB
    with pytest.raises(ValueError, match="exceeds 1.05"):
        db.upsert_etf_holdings("SPY", {"AAPL": 0.60, "MSFT": 0.50}, updated_at=now)

    class _FakeFundsData:
        def __init__(self, df: pd.DataFrame) -> None:
            self.top_holdings = df

    class _FakeTicker:
        def __init__(self, sym: str) -> None:
            if sym == "IWM":
                # RangeIndex with Symbol column + 0-100 small weights (max=0.65, total=4.75)
                self.funds_data = _FakeFundsData(
                    pd.DataFrame(
                        {
                            "Symbol": [f"S{i}" for i in range(1, 11)],
                            "Holding Percent": [0.65, 0.55, 0.50, 0.50, 0.45, 0.45, 0.45, 0.40, 0.40, 0.40],
                        }
                    )
                )
            elif sym == "SPY":
                # Fractional 0-1 weights with single top holding 0.25 (total=0.85) -> MUST NOT divide by 100!
                self.funds_data = _FakeFundsData(
                    pd.DataFrame(
                        {"Holding Percent": [0.25, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]},
                        index=pd.Index(["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "LLY", "XOM"]),
                    )
                )
            elif sym == "XLK":
                # RangeIndex WITHOUT symbol column -> invalid ("0", "1"), must skip!
                self.funds_data = _FakeFundsData(pd.DataFrame({"Holding Percent": [0.20, 0.15]}))
            elif sym == "XLE":
                # Corrupt sum > 1.05 after normalization -> must skip!
                self.funds_data = _FakeFundsData(
                    pd.DataFrame({"Holding Percent": [0.60, 0.60]}, index=pd.Index(["XOM", "CVX"]))
                )
            else:
                self.funds_data = _FakeFundsData(pd.DataFrame())

    monkeypatch.setattr("agy_trader.data.macro_and_calendar.yf.Ticker", _FakeTicker)
    fetched = YFinanceEtfHoldingsProvider().fetch_etf_holdings(["IWM", "SPY", "XLK", "XLE", "TLT"])
    assert pytest.approx(fetched["IWM"]["S1"], abs=1e-6) == 0.0065
    assert pytest.approx(fetched["SPY"]["AAPL"], abs=1e-6) == 0.25
    assert "XLK" not in fetched
    assert "XLE" not in fetched
    assert fetched["TLT"] == {}


def test_critical_3_deleted_env_missing_universe_and_hardened_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CRITICAL #3: Deleted .env fails closed, missing universe.yaml fails closed, BOM/duplicates/typos caught."""
    monkeypatch.chdir(tmp_path)
    db = StateDB(tmp_path / "data" / "state.db")

    # 1) Hardened parser: BOM + inline comment on empty value
    env_bom = tmp_path / "bom.env"
    env_bom.write_bytes(b"\xef\xbb\xbfBLACKOUT_PERIOD= # market closed\n")
    ok_res = resolve_live_blackout(
        env_file=env_bom,
        require_env_file=True,
        db=db,
        universe=load_universe(REPO_CONFIG_DIR),
    )
    assert ok_res.halted is False
    assert ok_res.raw_symbols == ()

    # Duplicate, lowercase, colon, typo lines MUST fail closed
    for bad_line in (
        'BLACKOUT_PERIOD="GOOG"\nBLACKOUT_PERIOD="ACME"\n',
        'blackout_period="GOOG"\n',
        'BLACKOUT_PERIOD: "GOOG"\n',
        'BLACKOUT_PERIODS="GOOG"\n',
    ):
        bad_p = tmp_path / "bad_syntax.env"
        bad_p.write_text(bad_line, encoding="utf-8")
        bad_res = resolve_live_blackout(env_file=bad_p, require_env_file=True)
        assert bad_res.halted is True, bad_line

    # 2) Broker created with no .env initially -> creating .env is picked up, deleting .env fails closed!
    tc = _FakeTradingClient()
    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True, blackout_period=""),
        db,
        trading_client=tc,
        config_dir=REPO_CONFIG_DIR,
    )
    assert broker.resolve_blackout_state().halted is False

    dot_env = tmp_path / ".env"
    dot_env.write_text('BLACKOUT_PERIOD="GOOG"\n', encoding="utf-8")
    _seed_etf_holdings(db, now_utc=datetime.now(UTC))
    r_created = broker.resolve_blackout_state()
    assert "GOOGL" in r_created.frozen_symbols

    dot_env.unlink()
    r_deleted = broker.resolve_blackout_state()
    assert r_deleted.halted is True

    # 3) Missing universe.yaml when BLACKOUT_PERIOD is non-empty fails closed
    dot_env.write_text('BLACKOUT_PERIOD="GOOG"\n', encoding="utf-8")
    empty_cfg = tmp_path / "empty_config"
    empty_cfg.mkdir()
    broker_missing_uni = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        env_file=dot_env,
        config_dir=empty_cfg,
    )
    r_no_uni = broker_missing_uni.resolve_blackout_state()
    assert r_no_uni.halted is True
    assert "universe" in r_no_uni.halt_reason.lower()


def test_high_4_mid_poll_blackout_aborts_cancel_and_market_fallback(tmp_path: Path) -> None:
    """HIGH #4: Blackout activated mid-poll stops limit cancel and market fallback, and halts mid-batch."""
    clock = _FakeClock()
    now = clock.now()
    db = StateDB(tmp_path / "state.db")
    _seed_etf_holdings(db, now_utc=now)
    env_p = tmp_path / ".env"
    env_p.write_text('BLACKOUT_PERIOD=""\n', encoding="utf-8")

    class _PollingBlackoutClient(_FakeTradingClient):
        def submit_order(self, req: Any) -> _FakeOrder:
            self.submitted_requests.append(req)
            self._seq += 1
            oid = f"brk-{self._seq}"
            cid = str(getattr(req, "client_order_id", oid))
            sym = str(getattr(req, "symbol", "")).upper()
            side = str(getattr(getattr(req, "side", ""), "value", getattr(req, "side", ""))).lower()
            qty = float(getattr(req, "qty", 0.0))
            limit_px = getattr(req, "limit_price", None)
            if limit_px is not None:
                order = _FakeOrder(
                    id=oid,
                    client_order_id=cid,
                    symbol=sym,
                    side=side,
                    order_type="limit",
                    qty=str(qty),
                    status="new",
                    filled_qty="0",
                    limit_price=str(limit_px),
                )
            else:
                order = _FakeOrder(
                    id=oid,
                    client_order_id=cid,
                    symbol=sym,
                    side=side,
                    order_type="market",
                    qty=str(qty),
                    status="filled",
                    filled_qty=str(qty),
                    filled_avg_price="180.0",
                )
            self.orders_by_id[oid] = order
            self.orders_by_cid[cid] = order
            return order

        def get_order_by_id(self, order_id: str) -> _FakeOrder:
            # Simulate .env edited to BLACKOUT_PERIOD="GOOG" while limit order is polling!
            env_p.write_text('BLACKOUT_PERIOD="GOOG"\n', encoding="utf-8")
            return super().get_order_by_id(order_id)

    tc = _PollingBlackoutClient()
    broker = AlpacaBroker(
        EnvSettings(alpaca_paper=True),
        db,
        trading_client=tc,
        limit_timeout_seconds=0.0,
        poll_interval_seconds=0.0,
        sleep_fn=clock.sleep,
        now_fn=clock.now,
        env_file=env_p,
        config_dir=REPO_CONFIG_DIR,
    )

    v_googl = RiskVerdict(
        approved=True,
        adjusted_qty=5.0,
        stop_loss_price=160.0,
        order=ProposedOrder(
            symbol="GOOGL",
            side=OrderSide.buy,
            qty=5.0,
            is_risk_reducing=False,
            reference_price=180.0,
            stop_loss_price=160.0,
        ),
    )
    v_aapl = RiskVerdict(
        approved=True,
        adjusted_qty=5.0,
        stop_loss_price=200.0,
        order=ProposedOrder(
            symbol="AAPL",
            side=OrderSide.buy,
            qty=5.0,
            is_risk_reducing=False,
            reference_price=220.0,
            stop_loss_price=200.0,
        ),
    )
    rep = broker.execute_verdicts(
        [v_googl, v_aapl],
        session_date="2026-11-20",
        checkpoint="1540",
        run_id="r-mid",
    )
    assert rep.records[0].status == "blocked_blackout"
    assert "brk-1" not in tc.canceled_ids  # Resting GOOGL limit order was NOT canceled!
    assert tc.orders_by_id["brk-1"].status == "new"
    assert not any(
        getattr(req, "symbol", "") == "GOOGL" and getattr(req, "limit_price", None) is None
        for req in tc.submitted_requests
    )  # Zero GOOGL market fallback submitted!


def test_high_5_and_low_9_sparse_etf_bound_and_future_clock_skew(tmp_path: Path) -> None:
    """HIGH #5 & LOW #9: 9-row dropped-class bound, <8 sparse ETF fail-closed, & future updated_at."""
    now = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)
    db = StateDB(tmp_path / "state.db")
    universe = load_universe(REPO_CONFIG_DIR)
    _seed_etf_holdings(db, now_utc=now)
    env_p = tmp_path / ".env"
    env_p.write_text('BLACKOUT_PERIOD="GOOG"\n', encoding="utf-8")

    # 1) 9-row XLC where GOOG dropped out and GOOGL=16%, min_top=8%, sum=72% -> 16+min(8,28)=24% >= 23.5%
    db.upsert_etf_holdings(
        "XLC",
        {
            "GOOGL": 0.16,
            "META": 0.08,
            "NFLX": 0.08,
            "DIS": 0.08,
            "TMUS": 0.08,
            "VZ": 0.08,
            "T": 0.08,
            "CMCSA": 0.08,
            "EA": 0.08,
        },
        is_non_equity_etf=False,
        updated_at=now,
    )
    res_9 = resolve_live_blackout(
        env_file=env_p, require_env_file=True, db=db, universe=universe, now_utc=now
    )
    assert "XLC" in res_9.frozen_symbols
    assert pytest.approx(res_9.etf_details["XLC"].conservative_weight, abs=1e-6) == 0.24

    # 2) < 8 rows and sum < 0.85 fails closed as stale_or_missing
    _seed_etf_holdings(db, now_utc=now)
    db.upsert_etf_holdings(
        "XLK",
        {"AAPL": 0.20, "MSFT": 0.20, "NVDA": 0.10, "AVGO": 0.10},
        is_non_equity_etf=False,
        updated_at=now,
    )
    res_sparse = resolve_live_blackout(
        env_file=env_p, require_env_file=True, db=db, universe=universe, now_utc=now
    )
    assert "XLK" in res_sparse.frozen_symbols
    assert res_sparse.etf_details["XLK"].stale_or_missing is True

    # 3) LOW #9: Future timestamp > 0.05 days ahead fails closed as stale_or_missing
    _seed_etf_holdings(db, now_utc=now)
    db.upsert_etf_holdings(
        "XLK",
        {f"H{i}": 0.05 for i in range(1, 11)},
        is_non_equity_etf=False,
        updated_at=now + timedelta(hours=3),
    )
    res_future = resolve_live_blackout(
        env_file=env_p, require_env_file=True, db=db, universe=universe, now_utc=now
    )
    assert "XLK" in res_future.frozen_symbols
    assert res_future.etf_details["XLK"].stale_or_missing is True


def test_high_6_medium_7_8_held_goog_pricing_sector_and_discord_retry_and_reconcile_spam(
    tmp_path: Path,
) -> None:
    """HIGH #6, MEDIUM #7, MEDIUM #8: Held GOOG pricing/sector, Discord retry on 500, no reconcile spam."""
    data_dir = tmp_path / "data"
    env_p = tmp_path / ".env"
    env_p.write_text('BLACKOUT_PERIOD="GOOG"\n', encoding="utf-8")

    db = StateDB(data_dir / "state.db")
    bar_store = ParquetBarStore(data_dir / "parquet")
    universe = load_universe(REPO_CONFIG_DIR)
    generate_synthetic_dataset(universe=universe, bar_store=bar_store, db=db, days=320, seed=23)

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

    rebal_utc = datetime(2026, 11, 20, 20, 40, tzinfo=UTC)
    db.upsert_macro(
        [
            MacroPoint(as_of_date=rebal_utc.date(), series_id="DGS3MO", value=4.25),
            MacroPoint(as_of_date=rebal_utc.date(), series_id="VIXCLS", value=16.0),
        ]
    )
    _seed_etf_holdings(db, now_utc=rebal_utc)

    # Hold out-of-universe GOOG WITHOUT a protective stop!
    tc = _FakeTradingClient()
    tc.positions = [_FakePosition("GOOG", "40", "175.00", "180.00", "200.00")]

    prices = {u.symbol: 150.0 for u in universe}
    snap_client = _FakeSnapshotClient(prices=prices, now_utc=rebal_utc)
    settings = EnvSettings(
        alpaca_paper=True,
        agy_data_dir=data_dir,
        agy_config_dir=REPO_CONFIG_DIR,
        discord_webhook_url=SAMPLE_WEBHOOK,
        nlp_backend="lexicon",
    )
    broker = AlpacaBroker(
        settings,
        db,
        trading_client=tc,
        quote_client=snap_client,
        env_file=env_p,
        config_dir=REPO_CONFIG_DIR,
        now_fn=lambda: rebal_utc,
    )

    # MEDIUM #8: reconcile_with_broker(record_if_clean=False) logs initial position discovery once,
    # then does NOT spam daemon_events on subsequent ticks even though frozen GOOG lacks a protective stop.
    broker.reconcile_with_broker(daemon_name="trader", record_if_clean=False)
    assert len(db.get_daemon_events(daemon_name="trader", event_type="reconcile")) == 1
    broker.reconcile_with_broker(daemon_name="trader", record_if_clean=False)
    broker.reconcile_with_broker(daemon_name="trader", record_if_clean=False)
    assert len(db.get_daemon_events(daemon_name="trader", event_type="reconcile")) == 1

    with respx.mock(assert_all_called=False) as router:
        route = router.post(SAMPLE_WEBHOOK).mock(
            side_effect=[
                httpx.Response(500),  # Tick 1 fails -> must NOT permanently suppress retry!
                httpx.Response(204),  # Tick at +301s succeeds!
            ]
        )
        notifier = DiscordWebhookNotifier(SAMPLE_WEBHOOK, max_retries=0)
        daemon = TraderDaemon(
            settings,
            data_dir,
            REPO_CONFIG_DIR,
            db=db,
            bar_store=bar_store,
            broker=broker,
            snapshot_client=snap_client,
            network_check_fn=lambda: True,
            env_file=env_p,
            discord_notifier=notifier,
        )

        # Tick 1 (rebalance): GOOG has valid price & Communication Services metadata, so non-frozen entries proceed!
        res1 = daemon.run_once(now_utc=rebal_utc)
        assert res1["rebalance"] is not None
        assert res1["rebalance"]["proposed_count"] > 0
        assert res1["rebalance"]["approved_count"] > 0
        assert all(getattr(req, "symbol", "") != "GOOG" for req in tc.submitted_requests)
        assert route.call_count == 1  # Failed with 500
        assert len(db.get_daemon_events(daemon_name="trader", event_type="blackout_alert")) == 1

        # Within 300s cooldown: no retry yet
        daemon.run_once(now_utc=rebal_utc + timedelta(seconds=60))
        assert route.call_count == 1

        # Past 300s cooldown: retries Discord and succeeds, without duplicating DB event!
        daemon.run_once(now_utc=rebal_utc + timedelta(seconds=301))
        assert route.call_count == 2
        assert len(db.get_daemon_events(daemon_name="trader", event_type="blackout_alert")) == 1

        # Subsequent tick: deduplicated after success
        daemon.run_once(now_utc=rebal_utc + timedelta(seconds=605))
        assert route.call_count == 2
        notifier.close()


def test_low_9_backfill_refreshes_etf_holdings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LOW #9: Non-synthetic `agy backfill` refreshes etf_holdings via YFinanceEtfHoldingsProvider."""
    data_dir = tmp_path / "data"
    universe = load_universe(REPO_CONFIG_DIR)
    sample_bars = {
        u.symbol: pd.DataFrame(
            {
                "timestamp": [datetime(2026, 11, 20, 21, 0, tzinfo=UTC)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000.0],
                "vwap": [100.2],
            }
        )
        for u in universe
    }

    class _MockMarket:
        def __init__(self, **_: Any) -> None:
            pass

        def fetch_bars(self, **_: Any) -> dict[str, pd.DataFrame]:
            return sample_bars

    class _MockEarn:
        def fetch_earnings(self, symbols: list[str]) -> list[Any]:
            return []

    class _MockEtf:
        def fetch_etf_holdings(self, symbols: list[str]) -> dict[str, dict[str, float]]:
            return {s: ({} if s in {"TLT", "GLD"} else {"AAPL": 0.10}) for s in symbols}

    monkeypatch.setattr("agy_trader.cli.main.AlpacaMarketDataProvider", _MockMarket)
    monkeypatch.setattr("agy_trader.cli.main.YFinanceEarningsProvider", _MockEarn)
    monkeypatch.setattr("agy_trader.cli.main.YFinanceEtfHoldingsProvider", _MockEtf)
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    res = runner.invoke(
        trader_cli_app,
        [
            "backfill",
            "--days",
            "5",
            "--skip-news",
            "--skip-sec",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(REPO_CONFIG_DIR),
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["etf_holdings_saved"] == 15
    db = StateDB(data_dir / "state.db")
    assert db.get_etf_holdings_meta("TLT")["is_non_equity_etf"] is True  # type: ignore[index]
    assert db.get_etf_holdings("XLC") == {"AAPL": 0.10}
