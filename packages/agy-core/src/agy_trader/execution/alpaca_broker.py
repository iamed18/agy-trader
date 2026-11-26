from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import structlog
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopOrderRequest,
)
from pydantic import BaseModel, ConfigDict, Field

from agy_trader.core.blackout import (
    BlackoutResolution,
    EnvBlackoutTracker,
)
from agy_trader.core.config import EnvSettings, validate_trading_mode
from agy_trader.core.control_plane import ControlPlane
from agy_trader.core.models import OrderSide, RiskVerdict, UtcDatetime
from agy_trader.storage.db import StateDB

logger = structlog.get_logger(__name__)

_TERMINAL_ORDER_STATUSES = frozenset(
    {
        "filled",
        "canceled",
        "expired",
        "rejected",
        "done_for_day",
        "replaced",
        "suspended",
    }
)
_CANCEL_FALLBACK_ELIGIBLE_STATUSES = frozenset({"canceled", "expired", "done_for_day"})
_OPEN_ORDER_STATUSES = frozenset(
    {
        "new",
        "accepted",
        "partially_filled",
        "pending_new",
        "pending_cancel",
        "pending_replace",
        "held",
        "accepted_for_bidding",
        "stopped",
        "calculated",
    }
)
_STOP_PROTECTIVE_TOLERANCE = 0.0025  # 0.25% tolerance when checking existing stop protectiveness
_MAX_CLIENT_ORDER_ID_LEN = 48
_MIN_STOP_QTY = 1e-5
SOFTWARE_STOP_ORDER_ID = "software"


class _StrictExecutionModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, validate_assignment=True)


class OrderExecutionRecord(_StrictExecutionModel):
    client_order_id: str
    broker_order_id: str | None = None
    symbol: str
    side: OrderSide
    order_type: str
    requested_qty: float
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    reference_price: float
    limit_price: float | None = None
    realized_slippage_bps: float | None = None
    status: str
    is_risk_reducing: bool
    adopted_existing: bool = False
    fallback_to_market: bool = False
    fallback_client_order_id: str | None = None
    error: str | None = None


class ExecutionReport(_StrictExecutionModel):
    run_id: str
    session_date: str
    checkpoint: str
    started_at: UtcDatetime
    completed_at: UtcDatetime
    records: list[OrderExecutionRecord] = Field(default_factory=list)
    pass1_count: int = 0
    pass2_count: int = 0
    skipped_unapproved: int = 0
    errors: list[str] = Field(default_factory=list)


class StopSyncResult(_StrictExecutionModel):
    session_date: str
    run_id: str
    kept_stops: dict[str, str] = Field(default_factory=dict)
    submitted_stops: dict[str, str] = Field(default_factory=dict)
    software_stops: dict[str, float] = Field(default_factory=dict)
    canceled_orphans: list[str] = Field(default_factory=list)
    canceled_stale: list[str] = Field(default_factory=list)
    untouched_blackout_positions: list[str] = Field(default_factory=list)
    untouched_blackout_orders: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class SoftwareStopExecution(_StrictExecutionModel):
    symbol: str
    qty: float
    trigger_price: float
    stop_price: float
    client_order_id: str
    broker_order_id: str | None = None
    status: str
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    error: str | None = None


class SoftwareStopResult(_StrictExecutionModel):
    session_date: str
    run_id: str
    checked_symbols: list[str] = Field(default_factory=list)
    triggered: list[SoftwareStopExecution] = Field(default_factory=list)
    untouched_blackout_positions: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def has_open_orders(self) -> bool:
        return any(t.status not in _TERMINAL_ORDER_STATUSES for t in self.triggered)


class ReconciliationResult(_StrictExecutionModel):
    timestamp: UtcDatetime
    daemon_name: str
    broker_positions_count: int
    db_positions_count: int
    position_diffs: list[dict[str, Any]] = Field(default_factory=list)
    updated_orders: list[str] = Field(default_factory=list)
    missing_stop_symbols: list[str] = Field(default_factory=list)
    untouched_blackout_positions: list[str] = Field(default_factory=list)
    untouched_blackout_orders: list[str] = Field(default_factory=list)
    stop_sync_result: StopSyncResult | None = None


def build_client_order_id(
    session_date: str,
    checkpoint: str,
    symbol: str,
    side: OrderSide,
    leg: str = "main",
) -> str:
    """Build a deterministic, ASCII-safe client_order_id within Alpaca's 48-char limit."""
    date_compact = re.sub(r"[^0-9]", "", session_date)[:8] or "00000000"
    ckpt_compact = re.sub(r"[^A-Za-z0-9]", "", checkpoint)[:6] or "0000"
    sym_clean = re.sub(r"[^A-Za-z0-9._-]", "", symbol.strip().upper())[:10] or "UNK"
    side_code = "B" if side == OrderSide.buy else "S"
    leg_clean = re.sub(r"[^A-Za-z0-9_-]", "", leg.strip()) or "main"

    cid = f"agy:{date_compact}:{ckpt_compact}:{sym_clean}:{side_code}:{leg_clean}"
    if len(cid) <= _MAX_CLIENT_ORDER_ID_LEN:
        return cid

    digest = hashlib.sha1(cid.encode("ascii")).hexdigest()[:6]
    prefix = f"agy:{date_compact}:{ckpt_compact}:{sym_clean}:{side_code}:"
    avail = _MAX_CLIENT_ORDER_ID_LEN - len(prefix) - 7
    if avail >= 1:
        return f"{prefix}{leg_clean[:avail]}-{digest}"
    return f"agy:{date_compact}:{sym_clean[:6]}:{side_code}:{digest}"[:_MAX_CLIENT_ORDER_ID_LEN]


def _enum_str(val: Any) -> str:
    if val is None:
        return ""
    return str(getattr(val, "value", val)).strip().lower()


def _opt_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    f = float(val)
    return f if math.isfinite(f) else None


def _req_float(val: Any, default: float = 0.0) -> float:
    parsed = _opt_float(val)
    return parsed if parsed is not None else default


def _to_alpaca_side(side: OrderSide) -> AlpacaOrderSide:
    return AlpacaOrderSide.BUY if side == OrderSide.buy else AlpacaOrderSide.SELL


def compute_slippage_bps(
    side: OrderSide, reference_price: float, fill_price: float | None
) -> float | None:
    """Return signed slippage in bps (>0 unfavorable, <0 price improvement)."""
    if fill_price is None or fill_price <= 0.0 or reference_price <= 0.0:
        return None
    if side == OrderSide.buy:
        return ((fill_price - reference_price) / reference_price) * 10_000.0
    return ((reference_price - fill_price) / reference_price) * 10_000.0


def _weighted_fill_price(
    qty1: float,
    px1: float | None,
    qty2: float,
    px2: float | None,
) -> float | None:
    parts: list[tuple[float, float]] = []
    if qty1 > 0.0 and px1 is not None and px1 > 0.0:
        parts.append((qty1, px1))
    if qty2 > 0.0 and px2 is not None and px2 > 0.0:
        parts.append((qty2, px2))
    if not parts:
        return None
    total_qty = sum(q for q, _ in parts)
    if total_qty <= 0.0:
        return None
    return sum(q * p for q, p in parts) / total_qty


def _is_not_found_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    msg = str(exc).lower()
    return status_code == 404 or "404" in msg or "not found" in msg


def _open_order_remaining_qty(order: Any) -> float:
    return max(
        0.0,
        _req_float(getattr(order, "qty", 0.0)) - _req_float(getattr(order, "filled_qty", 0.0)),
    )


def _same_side_db_stop_price(
    db_pos: Mapping[str, Any] | None,
    signed_qty: float,
) -> float | None:
    if db_pos is None:
        return None
    raw_px = db_pos.get("stop_price")
    if raw_px is None:
        return None
    px = float(raw_px)
    if not math.isfinite(px) or px <= 0.0:
        return None
    db_qty = float(db_pos.get("qty", 0.0))
    if (db_qty > 1e-9 and signed_qty > 1e-9) or (db_qty < -1e-9 and signed_qty < -1e-9):
        return px
    return None


class AlpacaBroker:
    """Idempotent, risk-aware live/paper Alpaca execution adapter.

    Hybrid Protective Stop Architecture:
        Alpaca only supports server-side ``StopOrderRequest`` (``TimeInForce.GTC``) on
        whole-share quantities. For hybrid fractional longs (``N + f`` shares with ``N >= 1``),
        ``sync_protective_stops`` attaches a server-side ``GTC`` stop on ``N`` whole shares
        and tracks the fractional remainder ``f`` in ``software_stops`` for 30-second daemon
        monitoring via ``evaluate_and_execute_software_stops``. For sub-1.0 fractional longs
        (``0 < f < 1``), no broker ``StopOrderRequest`` is submitted; instead, the position is
        persisted with ``stop_order_id="software"`` and monitored by the daemon watcher.
        Shorts remain whole-share ``GTC`` only.
    """

    def __init__(
        self,
        settings: EnvSettings,
        db: StateDB,
        *,
        trading_client: Any = None,
        quote_client: Any = None,
        limit_collar_bps: float = 15.0,
        limit_timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 2.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], datetime] | None = None,
        env_file: Path | None = None,
        require_env_file: bool = False,
        config_dir: Path | None = None,
    ) -> None:
        validate_trading_mode(settings)
        self.settings = settings
        self.db = db
        self.limit_collar_bps = float(limit_collar_bps)
        self.limit_timeout_seconds = float(limit_timeout_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self._sleep = sleep_fn
        self._now = now_fn or (lambda: datetime.now(UTC))
        self._blackout_tracker = EnvBlackoutTracker(
            env_file=env_file,
            require_env_file=require_env_file,
            fallback_raw=settings.blackout_period,
        )
        self._config_dir: Path = Path(config_dir) if config_dir is not None else Path(settings.agy_config_dir)
        self._data_feed = DataFeed.SIP if settings.alpaca_data_feed == "sip" else DataFeed.IEX
        self._owns_trading_client = trading_client is None
        self._owns_quote_client = False

        if trading_client is not None:
            self.trading_client = trading_client
        else:
            if not settings.alpaca_api_key.strip() or not settings.alpaca_secret_key.strip():
                raise ValueError(
                    "ALPACA_API_KEY and ALPACA_SECRET_KEY are required to initialize TradingClient"
                )
            self.trading_client = TradingClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                paper=settings.alpaca_paper,
            )

        if quote_client is not None:
            self.quote_client = quote_client
        elif (
            trading_client is None
            and settings.alpaca_api_key.strip()
            and settings.alpaca_secret_key.strip()
        ):
            self.quote_client = StockHistoricalDataClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
            )
            self._owns_quote_client = True
        else:
            self.quote_client = None

    def close(self) -> None:
        """Close underlying HTTP sessions on owned Alpaca SDK clients."""
        for owned, client in (
            (self._owns_trading_client, self.trading_client),
            (self._owns_quote_client, self.quote_client),
        ):
            if not owned or client is None:
                continue
            session = getattr(client, "_session", None)
            if session is not None and hasattr(session, "close"):
                session.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @staticmethod
    def _collect_observed_symbols(
        positions: Mapping[str, Any] | None = None,
        open_orders: Sequence[Any] | None = None,
        extra_symbols: Iterable[str] = (),
    ) -> set[str]:
        observed: set[str] = set()
        if positions:
            observed.update(str(s).strip().upper() for s in positions if str(s).strip())
        if open_orders:
            for order in open_orders:
                sym = getattr(order, "symbol", None)
                if sym:
                    clean = str(sym).strip().upper()
                    if clean:
                        observed.add(clean)
        for sym in extra_symbols:
            clean = str(sym).strip().upper()
            if clean:
                observed.add(clean)
        return observed

    def resolve_blackout_state(
        self,
        now_utc: datetime | None = None,
        observed_broker_symbols: Iterable[str] = (),
    ) -> BlackoutResolution:
        """Hot-reload BLACKOUT_PERIOD from .env on disk and resolve frozen symbols."""
        return self._blackout_tracker.resolve(
            db=self.db,
            config_dir=self._config_dir,
            now_utc=now_utc or self._now(),
            observed_broker_symbols=observed_broker_symbols,
        )

    def _resolve_blackout_with_live_observed(
        self,
        *,
        now_utc: datetime | None = None,
        extra_symbols: Iterable[str] = (),
    ) -> tuple[BlackoutResolution, set[str]]:
        """Resolve blackout and inspect live broker positions/open orders if blackout is active."""
        extra_set = {str(s).strip().upper() for s in extra_symbols if str(s).strip()}
        prelim = self.resolve_blackout_state(
            now_utc=now_utc,
            observed_broker_symbols=extra_set,
        )
        if prelim.halted or not prelim.raw_symbols:
            return prelim, extra_set
        positions = self.get_broker_positions()
        open_orders = self._fetch_open_orders()
        observed = self._collect_observed_symbols(positions, open_orders, extra_set)
        resolved = self.resolve_blackout_state(
            now_utc=now_utc,
            observed_broker_symbols=observed,
        )
        return resolved, observed

    def get_account_state(self) -> dict[str, float]:
        account = self.trading_client.get_account()
        return {
            "equity": _req_float(getattr(account, "equity", 0.0)),
            "cash": _req_float(getattr(account, "cash", 0.0)),
            "buying_power": _req_float(getattr(account, "buying_power", 0.0)),
            "last_equity": _req_float(getattr(account, "last_equity", 0.0)),
        }

    def get_broker_positions(self) -> dict[str, dict[str, float]]:
        raw_positions = self.trading_client.get_all_positions() or []
        result: dict[str, dict[str, float]] = {}
        for pos in raw_positions:
            sym = str(getattr(pos, "symbol", "")).strip().upper()
            if not sym:
                continue
            raw_qty = _req_float(getattr(pos, "qty", 0.0))
            side_str = _enum_str(getattr(pos, "side", ""))
            signed_qty = -raw_qty if (side_str == "short" and raw_qty > 0.0) else raw_qty
            if abs(signed_qty) <= 1e-9:
                continue
            qty_avail = _opt_float(getattr(pos, "qty_available", None))
            result[sym] = {
                "qty": signed_qty,
                "qty_available": (
                    float(qty_avail)
                    if (qty_avail is not None and qty_avail >= 0.0)
                    else abs(signed_qty)
                ),
                "avg_entry_price": _req_float(getattr(pos, "avg_entry_price", 0.0)),
                "market_price": _req_float(
                    getattr(pos, "current_price", getattr(pos, "market_price", 0.0))
                ),
                "unrealized_pnl": _req_float(
                    getattr(pos, "unrealized_pl", getattr(pos, "unrealized_pnl", 0.0))
                ),
            }
        return result

    def get_latest_quote(self, symbol: str) -> tuple[float | None, float | None]:
        if self.quote_client is None:
            return None, None
        clean_sym = symbol.strip().upper()
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=clean_sym, feed=self._data_feed)
            resp = self.quote_client.get_stock_latest_quote(req)
            quote = resp.get(clean_sym) if isinstance(resp, Mapping) else resp
            if quote is None:
                return None, None
            bid = _opt_float(getattr(quote, "bid_price", None))
            ask = _opt_float(getattr(quote, "ask_price", None))
            valid_bid = bid if (bid is not None and bid > 0.0) else None
            valid_ask = ask if (ask is not None and ask > 0.0) else None
            return valid_bid, valid_ask
        except Exception as exc:  # noqa: BLE001 - best-effort quote fetch falls back to reference_price
            logger.warning("alpaca_quote_unavailable", symbol=clean_sym, error=str(exc))
            return None, None

    def _fetch_order_by_client_id(self, client_order_id: str) -> Any | None:
        try:
            return self.trading_client.get_order_by_client_id(client_order_id)
        except Exception as exc:
            if _is_not_found_error(exc):
                return None
            raise

    def _fetch_order_by_id(self, order_id: str) -> Any | None:
        try:
            return self.trading_client.get_order_by_id(order_id)
        except Exception as exc:
            if _is_not_found_error(exc):
                return None
            raise

    def _fetch_open_orders(self) -> list[Any]:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        return list(self.trading_client.get_orders(filter=req) or [])

    def _compute_limit_price(self, symbol: str, side: OrderSide, reference_price: float) -> float:
        bid, ask = self.get_latest_quote(symbol)
        collar = self.limit_collar_bps / 10_000.0
        if side == OrderSide.buy:
            base = ask if ask is not None else reference_price
            raw_px = base * (1.0 + collar)
        else:
            base = bid if bid is not None else reference_price
            raw_px = base * (1.0 - collar)
        return max(0.01, round(raw_px, 2))

    def _persist_broker_order(
        self,
        order: Any,
        *,
        client_order_id: str,
        run_id: str,
        symbol: str,
        side: OrderSide,
        order_type: str,
        qty: float,
        reference_price: float,
        is_risk_reducing: bool,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> tuple[str | None, str, float, float | None, float | None]:
        broker_id_raw = getattr(order, "id", None)
        broker_id = str(broker_id_raw) if broker_id_raw is not None else None
        status = _enum_str(getattr(order, "status", "new")) or "new"
        filled_qty = _req_float(getattr(order, "filled_qty", 0.0))
        filled_avg_price = _opt_float(getattr(order, "filled_avg_price", None))
        slippage_bps = compute_slippage_bps(side, reference_price, filled_avg_price)
        eff_limit = _opt_float(getattr(order, "limit_price", limit_price))
        eff_stop = _opt_float(getattr(order, "stop_price", stop_price))
        now = self._now()

        self.db.upsert_live_order(
            client_order_id=client_order_id,
            broker_order_id=broker_id,
            run_id=run_id,
            symbol=symbol,
            side=side.value,
            order_type=order_type,
            qty=qty,
            limit_price=eff_limit,
            stop_price=eff_stop,
            status=status,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
            reference_price=reference_price,
            realized_slippage_bps=slippage_bps,
            is_risk_reducing=is_risk_reducing,
            submitted_at=now,
            updated_at=now,
        )
        return broker_id, status, filled_qty, filled_avg_price, slippage_bps

    def _poll_order(
        self,
        *,
        broker_order_id: str | None,
        client_order_id: str,
        timeout_seconds: float,
    ) -> Any | None:
        started = self._now()
        latest: Any | None = None
        while True:
            if broker_order_id is not None:
                latest = self._fetch_order_by_id(broker_order_id)
            if latest is None and client_order_id:
                latest = self._fetch_order_by_client_id(client_order_id)
            if latest is not None:
                status = _enum_str(getattr(latest, "status", ""))
                if status in _TERMINAL_ORDER_STATUSES:
                    return latest
            elapsed = (self._now() - started).total_seconds()
            if elapsed >= timeout_seconds:
                return latest
            self._sleep(self.poll_interval_seconds)

    def _adopt_existing_order(
        self,
        *,
        cid: str,
        mkt_cid: str,
        verdict: RiskVerdict,
        run_id: str,
        db_order: dict[str, Any] | None,
        broker_order: Any | None,
    ) -> OrderExecutionRecord:
        symbol = verdict.order.symbol
        side = verdict.order.side
        req_qty = float(verdict.adjusted_qty)
        ref_px = float(verdict.order.reference_price)
        is_rr = bool(verdict.order.is_risk_reducing)

        if broker_order is not None:
            order_type = (
                _enum_str(
                    getattr(broker_order, "order_type", None)
                    or getattr(broker_order, "type", "limit")
                )
                or "limit"
            )
            b_id, status, f_qty, f_px, slip = self._persist_broker_order(
                broker_order,
                client_order_id=cid,
                run_id=run_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                qty=req_qty,
                reference_price=ref_px,
                is_risk_reducing=is_rr,
            )
            limit_px = _opt_float(getattr(broker_order, "limit_price", None))
        else:
            assert db_order is not None
            b_id = db_order["broker_order_id"]
            status = str(db_order["status"])
            order_type = str(db_order["order_type"])
            f_qty = float(db_order["filled_qty"])
            f_px = db_order["filled_avg_price"]
            slip = db_order["realized_slippage_bps"]
            limit_px = db_order["limit_price"]

        # Also check if a fallback market leg was already submitted for this order
        mkt_broker = self._fetch_order_by_client_id(mkt_cid)
        mkt_db = self.db.get_live_order(mkt_cid)
        fallback_used = False
        fallback_cid: str | None = None
        if mkt_broker is not None:
            fallback_used = True
            fallback_cid = mkt_cid
            m_bid, m_status, m_qty, m_px, _ = self._persist_broker_order(
                mkt_broker,
                client_order_id=mkt_cid,
                run_id=run_id,
                symbol=symbol,
                side=side,
                order_type="market",
                qty=max(0.0, req_qty - f_qty),
                reference_price=ref_px,
                is_risk_reducing=is_rr,
            )
            if m_status not in _TERMINAL_ORDER_STATUSES:
                m_polled = self._poll_order(
                    broker_order_id=m_bid,
                    client_order_id=mkt_cid,
                    timeout_seconds=self.limit_timeout_seconds,
                )
                if m_polled is not None:
                    _, m_status, m_qty, m_px, _ = self._persist_broker_order(
                        m_polled,
                        client_order_id=mkt_cid,
                        run_id=run_id,
                        symbol=symbol,
                        side=side,
                        order_type="market",
                        qty=max(0.0, req_qty - f_qty),
                        reference_price=ref_px,
                        is_risk_reducing=is_rr,
                    )
            f_px = _weighted_fill_price(f_qty, f_px, m_qty, m_px)
            f_qty = f_qty + m_qty
            slip = compute_slippage_bps(side, ref_px, f_px)
            status = m_status
        elif mkt_db is not None:
            fallback_used = True
            fallback_cid = mkt_cid
            m_qty = float(mkt_db["filled_qty"])
            m_px = mkt_db["filled_avg_price"]
            f_px = _weighted_fill_price(f_qty, f_px, m_qty, m_px)
            f_qty = f_qty + m_qty
            slip = compute_slippage_bps(side, ref_px, f_px)
            status = str(mkt_db["status"])
        elif broker_order is not None and status in _OPEN_ORDER_STATUSES:
            if order_type == "market":
                polled_mkt = self._poll_order(
                    broker_order_id=b_id,
                    client_order_id=cid,
                    timeout_seconds=self.limit_timeout_seconds,
                )
                if polled_mkt is not None:
                    b_id, status, f_qty, f_px, slip = self._persist_broker_order(
                        polled_mkt,
                        client_order_id=cid,
                        run_id=run_id,
                        symbol=symbol,
                        side=side,
                        order_type=order_type,
                        qty=req_qty,
                        reference_price=ref_px,
                        is_risk_reducing=is_rr,
                        limit_price=limit_px,
                    )
            else:
                return self._complete_limit_and_fallback(
                    cid=cid,
                    mkt_cid=mkt_cid,
                    verdict=verdict,
                    run_id=run_id,
                    b_id=b_id,
                    status=status,
                    f_qty=f_qty,
                    f_px=f_px,
                    slip=slip,
                    limit_px=limit_px,
                    adopted_existing=True,
                )

        return OrderExecutionRecord(
            client_order_id=cid,
            broker_order_id=b_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            requested_qty=req_qty,
            filled_qty=f_qty,
            filled_avg_price=f_px,
            reference_price=ref_px,
            limit_price=limit_px,
            realized_slippage_bps=slip,
            status=status,
            is_risk_reducing=is_rr,
            adopted_existing=True,
            fallback_to_market=fallback_used,
            fallback_client_order_id=fallback_cid,
        )

    def _complete_limit_and_fallback(
        self,
        *,
        cid: str,
        mkt_cid: str,
        verdict: RiskVerdict,
        run_id: str,
        b_id: str | None,
        status: str,
        f_qty: float,
        f_px: float | None,
        slip: float | None,
        limit_px: float | None,
        adopted_existing: bool,
    ) -> OrderExecutionRecord:
        """Poll an open limit order, cancel on timeout, and submit terminal-gated market fallback."""
        symbol = verdict.order.symbol
        side = verdict.order.side
        req_qty = float(verdict.adjusted_qty)
        ref_px = float(verdict.order.reference_price)
        is_rr = bool(verdict.order.is_risk_reducing)

        if status not in _TERMINAL_ORDER_STATUSES:
            polled = self._poll_order(
                broker_order_id=b_id,
                client_order_id=cid,
                timeout_seconds=self.limit_timeout_seconds,
            )
            if polled is not None:
                b_id, status, f_qty, f_px, slip = self._persist_broker_order(
                    polled,
                    client_order_id=cid,
                    run_id=run_id,
                    symbol=symbol,
                    side=side,
                    order_type="limit",
                    qty=req_qty,
                    reference_price=ref_px,
                    is_risk_reducing=is_rr,
                    limit_price=limit_px,
                )

        if status == "filled" or f_qty >= req_qty - 1e-6:
            return OrderExecutionRecord(
                client_order_id=cid,
                broker_order_id=b_id,
                symbol=symbol,
                side=side,
                order_type="limit",
                requested_qty=req_qty,
                filled_qty=f_qty,
                filled_avg_price=f_px,
                reference_price=ref_px,
                limit_price=limit_px,
                realized_slippage_bps=slip,
                status=status,
                is_risk_reducing=is_rr,
                adopted_existing=adopted_existing,
            )

        # HIGH #4: Re-check blackout state after limit poll BEFORE canceling resting limit order
        blackout_mid = self.resolve_blackout_state(
            now_utc=self._now(),
            observed_broker_symbols=(symbol,),
        )
        if blackout_mid.halted or symbol in blackout_mid.frozen_symbols:
            mid_err = (
                f"{symbol}: BLACKOUT_PERIOD activated mid-poll "
                f"({blackout_mid.halt_reason or 'frozen'}); leaving resting order {cid} untouched"
            )
            logger.warning(
                "alpaca_blackout_mid_poll_abort",
                symbol=symbol,
                client_order_id=cid,
                halted=blackout_mid.halted,
            )
            return OrderExecutionRecord(
                client_order_id=cid,
                broker_order_id=b_id,
                symbol=symbol,
                side=side,
                order_type="limit",
                requested_qty=req_qty,
                filled_qty=f_qty,
                filled_avg_price=f_px,
                reference_price=ref_px,
                limit_price=limit_px,
                realized_slippage_bps=slip,
                status="blocked_blackout",
                is_risk_reducing=is_rr,
                adopted_existing=adopted_existing,
                error=mid_err,
            )

        # Unfilled or partially filled after limit timeout: cancel open limit before market fallback
        if status in _OPEN_ORDER_STATUSES and b_id is not None:
            try:
                self.trading_client.cancel_order_by_id(b_id)
            except Exception as exc:  # noqa: BLE001 - cancel may race with fill; poll checks final state
                logger.warning(
                    "alpaca_limit_cancel_error",
                    symbol=symbol,
                    broker_order_id=b_id,
                    error=str(exc),
                )
            cancel_ack_timeout = min(10.0, max(self.poll_interval_seconds * 3.0, 1.0))
            acked = self._poll_order(
                broker_order_id=b_id,
                client_order_id=cid,
                timeout_seconds=cancel_ack_timeout,
            )
            if acked is not None:
                b_id, status, f_qty, f_px, slip = self._persist_broker_order(
                    acked,
                    client_order_id=cid,
                    run_id=run_id,
                    symbol=symbol,
                    side=side,
                    order_type="limit",
                    qty=req_qty,
                    reference_price=ref_px,
                    is_risk_reducing=is_rr,
                    limit_price=limit_px,
                )

        remaining_qty = round(max(0.0, req_qty - f_qty), 6)
        if remaining_qty <= 1e-6 or status == "filled":
            return OrderExecutionRecord(
                client_order_id=cid,
                broker_order_id=b_id,
                symbol=symbol,
                side=side,
                order_type="limit",
                requested_qty=req_qty,
                filled_qty=f_qty,
                filled_avg_price=f_px,
                reference_price=ref_px,
                limit_price=limit_px,
                realized_slippage_bps=slip,
                status=status,
                is_risk_reducing=is_rr,
                adopted_existing=adopted_existing,
            )

        # CRITICAL #2: Only submit fallback market order when limit cancel reached a safe terminal
        # state (canceled/expired/done_for_day). Never race a non-terminal pending_cancel order!
        if status not in _CANCEL_FALLBACK_ELIGIBLE_STATUSES:
            err_msg = (
                None
                if status in {"rejected", "suspended", "replaced"}
                else (
                    f"{symbol}: limit order {cid} remained non-terminal ({status}) after cancel "
                    "timeout; skipping market fallback to prevent double-fill"
                )
            )
            if err_msg is not None:
                logger.warning(
                    "alpaca_cancel_ack_timeout_skipping_fallback",
                    symbol=symbol,
                    client_order_id=cid,
                    status=status,
                )
            return OrderExecutionRecord(
                client_order_id=cid,
                broker_order_id=b_id,
                symbol=symbol,
                side=side,
                order_type="limit",
                requested_qty=req_qty,
                filled_qty=f_qty,
                filled_avg_price=f_px,
                reference_price=ref_px,
                limit_price=limit_px,
                realized_slippage_bps=slip,
                status=status,
                is_risk_reducing=is_rr,
                adopted_existing=adopted_existing,
                error=err_msg,
            )

        # Idempotency check on fallback market order too
        existing_mkt_broker = self._fetch_order_by_client_id(mkt_cid)
        existing_mkt_db = self.db.get_live_order(mkt_cid)
        if existing_mkt_broker is not None:
            mkt_order = existing_mkt_broker
        elif existing_mkt_db is not None and existing_mkt_db["status"] not in {"rejected", "error"}:
            m_qty = float(existing_mkt_db["filled_qty"])
            m_px = existing_mkt_db["filled_avg_price"]
            agg_px = _weighted_fill_price(f_qty, f_px, m_qty, m_px)
            agg_qty = f_qty + m_qty
            return OrderExecutionRecord(
                client_order_id=cid,
                broker_order_id=b_id,
                symbol=symbol,
                side=side,
                order_type="market",
                requested_qty=req_qty,
                filled_qty=agg_qty,
                filled_avg_price=agg_px,
                reference_price=ref_px,
                limit_price=limit_px,
                realized_slippage_bps=compute_slippage_bps(side, ref_px, agg_px),
                status=str(existing_mkt_db["status"]),
                is_risk_reducing=is_rr,
                adopted_existing=True,
                fallback_to_market=True,
                fallback_client_order_id=mkt_cid,
            )
        else:
            # HIGH #4: Final blackout check immediately before submitting market fallback
            blackout_pre_mkt = self.resolve_blackout_state(
                now_utc=self._now(),
                observed_broker_symbols=(symbol,),
            )
            if blackout_pre_mkt.halted or symbol in blackout_pre_mkt.frozen_symbols:
                mkt_blk_err = (
                    f"{symbol}: BLACKOUT_PERIOD active before market fallback "
                    f"({blackout_pre_mkt.halt_reason or 'frozen'}); skipping market submit"
                )
                return OrderExecutionRecord(
                    client_order_id=cid,
                    broker_order_id=b_id,
                    symbol=symbol,
                    side=side,
                    order_type="limit",
                    requested_qty=req_qty,
                    filled_qty=f_qty,
                    filled_avg_price=f_px,
                    reference_price=ref_px,
                    limit_price=limit_px,
                    realized_slippage_bps=slip,
                    status="blocked_blackout",
                    is_risk_reducing=is_rr,
                    adopted_existing=adopted_existing,
                    error=mkt_blk_err,
                )
            mkt_req = MarketOrderRequest(
                symbol=symbol,
                qty=remaining_qty,
                side=_to_alpaca_side(side),
                time_in_force=TimeInForce.DAY,
                client_order_id=mkt_cid,
            )
            mkt_order = self.trading_client.submit_order(mkt_req)

        m_bid, m_status, m_qty, m_px, _ = self._persist_broker_order(
            mkt_order,
            client_order_id=mkt_cid,
            run_id=run_id,
            symbol=symbol,
            side=side,
            order_type="market",
            qty=remaining_qty,
            reference_price=ref_px,
            is_risk_reducing=is_rr,
        )
        if m_status not in _TERMINAL_ORDER_STATUSES:
            m_polled = self._poll_order(
                broker_order_id=m_bid,
                client_order_id=mkt_cid,
                timeout_seconds=self.limit_timeout_seconds,
            )
            if m_polled is not None:
                m_bid, m_status, m_qty, m_px, _ = self._persist_broker_order(
                    m_polled,
                    client_order_id=mkt_cid,
                    run_id=run_id,
                    symbol=symbol,
                    side=side,
                    order_type="market",
                    qty=remaining_qty,
                    reference_price=ref_px,
                    is_risk_reducing=is_rr,
                )

        agg_qty = f_qty + m_qty
        agg_px = _weighted_fill_price(f_qty, f_px, m_qty, m_px)
        return OrderExecutionRecord(
            client_order_id=cid,
            broker_order_id=m_bid or b_id,
            symbol=symbol,
            side=side,
            order_type="market",
            requested_qty=req_qty,
            filled_qty=agg_qty,
            filled_avg_price=agg_px,
            reference_price=ref_px,
            limit_price=limit_px,
            realized_slippage_bps=compute_slippage_bps(side, ref_px, agg_px),
            status=m_status,
            is_risk_reducing=is_rr,
            adopted_existing=adopted_existing,
            fallback_to_market=True,
            fallback_client_order_id=mkt_cid,
        )

    def _execute_single_verdict(
        self,
        verdict: RiskVerdict,
        *,
        session_date: str,
        checkpoint: str,
        run_id: str,
        leg_main: str = "main",
        leg_mkt: str = "mkt",
    ) -> OrderExecutionRecord:
        symbol = verdict.order.symbol
        side = verdict.order.side
        req_qty = float(verdict.adjusted_qty)
        ref_px = float(verdict.order.reference_price)
        is_rr = bool(verdict.order.is_risk_reducing)

        cid = build_client_order_id(session_date, checkpoint, symbol, side, leg=leg_main)
        mkt_cid = build_client_order_id(session_date, checkpoint, symbol, side, leg=leg_mkt)

        db_order = self.db.get_live_order(cid)
        broker_order = self._fetch_order_by_client_id(cid)
        if broker_order is not None or (
            db_order is not None and db_order["status"] not in {"rejected", "error"}
        ):
            return self._adopt_existing_order(
                cid=cid,
                mkt_cid=mkt_cid,
                verdict=verdict,
                run_id=run_id,
                db_order=db_order,
                broker_order=broker_order,
            )

        limit_px = self._compute_limit_price(symbol, side, ref_px)
        limit_req = LimitOrderRequest(
            symbol=symbol,
            qty=req_qty,
            side=_to_alpaca_side(side),
            time_in_force=TimeInForce.DAY,
            limit_price=limit_px,
            client_order_id=cid,
        )
        submitted = self.trading_client.submit_order(limit_req)
        b_id, status, f_qty, f_px, slip = self._persist_broker_order(
            submitted,
            client_order_id=cid,
            run_id=run_id,
            symbol=symbol,
            side=side,
            order_type="limit",
            qty=req_qty,
            reference_price=ref_px,
            is_risk_reducing=is_rr,
            limit_price=limit_px,
        )
        return self._complete_limit_and_fallback(
            cid=cid,
            mkt_cid=mkt_cid,
            verdict=verdict,
            run_id=run_id,
            b_id=b_id,
            status=status,
            f_qty=f_qty,
            f_px=f_px,
            slip=slip,
            limit_px=limit_px,
            adopted_existing=False,
        )

    def execute_verdicts(
        self,
        verdicts: Sequence[RiskVerdict],
        *,
        session_date: str,
        checkpoint: str,
        run_id: str,
    ) -> ExecutionReport:
        """Execute HardRiskEngine-approved verdicts: Pass 1 (risk-reducing) before Pass 2."""
        validate_trading_mode(self.settings)
        started_at = self._now()

        approved: list[RiskVerdict] = []
        skipped_unapproved = 0
        for v in verdicts:
            if v.approved and v.adjusted_qty > 0.0:
                approved.append(v)
            else:
                skipped_unapproved += 1

        verdict_symbols = {v.order.symbol for v in approved}
        blackout, observed_symbols = self._resolve_blackout_with_live_observed(
            now_utc=started_at,
            extra_symbols=verdict_symbols,
        )

        records: list[OrderExecutionRecord] = []
        errors: list[str] = []
        occ_counts: dict[tuple[int, str, OrderSide], int] = defaultdict(int)

        def _next_legs(pass_num: int, verdict: RiskVerdict) -> tuple[str, str]:
            key = (pass_num, verdict.order.symbol, verdict.order.side)
            occ = occ_counts[key]
            occ_counts[key] = occ + 1
            return f"p{pass_num}m{occ}", f"p{pass_num}k{occ}"

        if blackout.halted:
            halt_err = f"BLACKOUT_PERIOD halt active ({blackout.halt_reason}); all orders blocked"
            errors.append(halt_err)
            for v in approved:
                pass_num = 1 if v.order.is_risk_reducing else 2
                leg_main, _ = _next_legs(pass_num, v)
                cid = build_client_order_id(
                    session_date, checkpoint, v.order.symbol, v.order.side, leg=leg_main
                )
                records.append(
                    OrderExecutionRecord(
                        client_order_id=cid,
                        symbol=v.order.symbol,
                        side=v.order.side,
                        order_type="limit",
                        requested_qty=float(v.adjusted_qty),
                        reference_price=float(v.order.reference_price),
                        status="blocked_blackout",
                        is_risk_reducing=bool(v.order.is_risk_reducing),
                        error=halt_err,
                    )
                )
            completed_at = self._now()
            return ExecutionReport(
                run_id=run_id,
                session_date=session_date,
                checkpoint=checkpoint,
                started_at=started_at,
                completed_at=completed_at,
                records=records,
                pass1_count=0,
                pass2_count=0,
                skipped_unapproved=skipped_unapproved,
                errors=errors,
            )

        failed_pass1_symbols: set[str] = set()
        any_pass1_complete_failure = False

        executable_approved: list[RiskVerdict] = []
        for v in approved:
            if v.order.symbol in blackout.frozen_symbols:
                pass_num = 1 if v.order.is_risk_reducing else 2
                if v.order.is_risk_reducing:
                    failed_pass1_symbols.add(v.order.symbol)
                    any_pass1_complete_failure = True
                leg_main, _ = _next_legs(pass_num, v)
                cid = build_client_order_id(
                    session_date, checkpoint, v.order.symbol, v.order.side, leg=leg_main
                )
                blk_err = f"{v.order.symbol}: blocked by BLACKOUT_PERIOD compliance rail"
                errors.append(blk_err)
                records.append(
                    OrderExecutionRecord(
                        client_order_id=cid,
                        symbol=v.order.symbol,
                        side=v.order.side,
                        order_type="limit",
                        requested_qty=float(v.adjusted_qty),
                        reference_price=float(v.order.reference_price),
                        status="blocked_blackout",
                        is_risk_reducing=bool(v.order.is_risk_reducing),
                        error=blk_err,
                    )
                )
            else:
                executable_approved.append(v)

        pass1 = [v for v in executable_approved if v.order.is_risk_reducing]
        pass2 = [v for v in executable_approved if not v.order.is_risk_reducing]

        def _handle_exception(
            verdict: RiskVerdict, cid: str, exc: Exception
        ) -> OrderExecutionRecord:
            err_msg = f"{verdict.order.symbol}: {exc}"
            errors.append(err_msg)
            logger.exception(
                "alpaca_order_execution_error",
                symbol=verdict.order.symbol,
                client_order_id=cid,
            )
            # MEDIUM #10: Preserve already-submitted broker or DB order state if exception occurred
            # during polling/handling rather than overwriting to "error".
            existing_broker: Any | None = None
            try:
                existing_broker = self._fetch_order_by_client_id(cid)
            except Exception:  # noqa: BLE001 - best-effort lookup on error path
                existing_broker = None

            if existing_broker is not None:
                b_id, status, f_qty, f_px, slip = self._persist_broker_order(
                    existing_broker,
                    client_order_id=cid,
                    run_id=run_id,
                    symbol=verdict.order.symbol,
                    side=verdict.order.side,
                    order_type=(
                        _enum_str(
                            getattr(existing_broker, "order_type", None)
                            or getattr(existing_broker, "type", "limit")
                        )
                        or "limit"
                    ),
                    qty=verdict.adjusted_qty,
                    reference_price=verdict.order.reference_price,
                    is_risk_reducing=verdict.order.is_risk_reducing,
                )
                return OrderExecutionRecord(
                    client_order_id=cid,
                    broker_order_id=b_id,
                    symbol=verdict.order.symbol,
                    side=verdict.order.side,
                    order_type="limit",
                    requested_qty=verdict.adjusted_qty,
                    filled_qty=f_qty,
                    filled_avg_price=f_px,
                    reference_price=verdict.order.reference_price,
                    limit_price=_opt_float(getattr(existing_broker, "limit_price", None)),
                    realized_slippage_bps=slip,
                    status=status,
                    is_risk_reducing=verdict.order.is_risk_reducing,
                    error=str(exc),
                )

            existing_db = self.db.get_live_order(cid)
            if existing_db is not None and existing_db["status"] != "error":
                return OrderExecutionRecord(
                    client_order_id=cid,
                    broker_order_id=existing_db["broker_order_id"],
                    symbol=verdict.order.symbol,
                    side=verdict.order.side,
                    order_type=str(existing_db["order_type"]),
                    requested_qty=verdict.adjusted_qty,
                    filled_qty=float(existing_db["filled_qty"]),
                    filled_avg_price=existing_db["filled_avg_price"],
                    reference_price=verdict.order.reference_price,
                    limit_price=existing_db["limit_price"],
                    realized_slippage_bps=existing_db["realized_slippage_bps"],
                    status=str(existing_db["status"]),
                    is_risk_reducing=verdict.order.is_risk_reducing,
                    error=str(exc),
                )

            now = self._now()
            self.db.upsert_live_order(
                client_order_id=cid,
                run_id=run_id,
                symbol=verdict.order.symbol,
                side=verdict.order.side.value,
                order_type="limit",
                qty=verdict.adjusted_qty,
                status="error",
                reference_price=verdict.order.reference_price,
                is_risk_reducing=verdict.order.is_risk_reducing,
                submitted_at=now,
                updated_at=now,
            )
            return OrderExecutionRecord(
                client_order_id=cid,
                symbol=verdict.order.symbol,
                side=verdict.order.side,
                order_type="limit",
                requested_qty=verdict.adjusted_qty,
                reference_price=verdict.order.reference_price,
                status="error",
                is_risk_reducing=verdict.order.is_risk_reducing,
                error=str(exc),
            )

        # Pass 1: risk-reducing exits/trims
        for verdict in pass1:
            leg_main, leg_mkt = _next_legs(1, verdict)
            cid = build_client_order_id(
                session_date, checkpoint, verdict.order.symbol, verdict.order.side, leg=leg_main
            )
            blackout_now = self.resolve_blackout_state(
                now_utc=self._now(),
                observed_broker_symbols=observed_symbols,
            )
            if blackout_now.halted or verdict.order.symbol in blackout_now.frozen_symbols:
                blk_err = (
                    f"BLACKOUT_PERIOD halt active ({blackout_now.halt_reason}); all orders blocked"
                    if blackout_now.halted
                    else f"{verdict.order.symbol}: blocked by BLACKOUT_PERIOD compliance rail"
                )
                errors.append(blk_err)
                records.append(
                    OrderExecutionRecord(
                        client_order_id=cid,
                        symbol=verdict.order.symbol,
                        side=verdict.order.side,
                        order_type="limit",
                        requested_qty=float(verdict.adjusted_qty),
                        reference_price=float(verdict.order.reference_price),
                        status="blocked_blackout",
                        is_risk_reducing=True,
                        error=blk_err,
                    )
                )
                failed_pass1_symbols.add(verdict.order.symbol)
                any_pass1_complete_failure = True
                continue
            try:
                rec = self._execute_single_verdict(
                    verdict,
                    session_date=session_date,
                    checkpoint=checkpoint,
                    run_id=run_id,
                    leg_main=leg_main,
                    leg_mkt=leg_mkt,
                )
                records.append(rec)
                if rec.status == "rejected":
                    errors.append(f"{rec.symbol}: order {rec.client_order_id} rejected")
                elif rec.error is not None:
                    errors.append(rec.error)

                if rec.status in {"rejected", "error"} or rec.filled_qty <= 1e-6:
                    failed_pass1_symbols.add(rec.symbol)
                    any_pass1_complete_failure = True
                elif rec.filled_qty < rec.requested_qty - 1e-6 or rec.status != "filled":
                    failed_pass1_symbols.add(rec.symbol)
            except Exception as exc:  # noqa: BLE001 - isolate per-order SDK/network exceptions
                rec = _handle_exception(verdict, cid, exc)
                records.append(rec)
                failed_pass1_symbols.add(verdict.order.symbol)
                if rec.filled_qty <= 1e-6:
                    any_pass1_complete_failure = True

        # Pass 2: risk-increasing entries/adds (gated on Pass-1 health)
        for verdict in pass2:
            leg_main, leg_mkt = _next_legs(2, verdict)
            cid = build_client_order_id(
                session_date, checkpoint, verdict.order.symbol, verdict.order.side, leg=leg_main
            )
            if any_pass1_complete_failure or verdict.order.symbol in failed_pass1_symbols:
                gate_err = (
                    f"{verdict.order.symbol}: Pass-2 entry gated due to Pass-1 failure/under-fill"
                )
                errors.append(gate_err)
                records.append(
                    OrderExecutionRecord(
                        client_order_id=cid,
                        symbol=verdict.order.symbol,
                        side=verdict.order.side,
                        order_type="limit",
                        requested_qty=verdict.adjusted_qty,
                        reference_price=verdict.order.reference_price,
                        status="skipped_pass1_gate",
                        is_risk_reducing=False,
                        error=gate_err,
                    )
                )
                continue

            blackout_now = self.resolve_blackout_state(
                now_utc=self._now(),
                observed_broker_symbols=observed_symbols,
            )
            if blackout_now.halted or verdict.order.symbol in blackout_now.frozen_symbols:
                blk_err = (
                    f"BLACKOUT_PERIOD halt active ({blackout_now.halt_reason}); all orders blocked"
                    if blackout_now.halted
                    else f"{verdict.order.symbol}: blocked by BLACKOUT_PERIOD compliance rail"
                )
                errors.append(blk_err)
                records.append(
                    OrderExecutionRecord(
                        client_order_id=cid,
                        symbol=verdict.order.symbol,
                        side=verdict.order.side,
                        order_type="limit",
                        requested_qty=float(verdict.adjusted_qty),
                        reference_price=float(verdict.order.reference_price),
                        status="blocked_blackout",
                        is_risk_reducing=False,
                        error=blk_err,
                    )
                )
                continue

            try:
                rec = self._execute_single_verdict(
                    verdict,
                    session_date=session_date,
                    checkpoint=checkpoint,
                    run_id=run_id,
                    leg_main=leg_main,
                    leg_mkt=leg_mkt,
                )
                records.append(rec)
                if rec.status == "rejected":
                    errors.append(f"{rec.symbol}: order {rec.client_order_id} rejected")
                elif rec.error is not None:
                    errors.append(rec.error)
            except Exception as exc:  # noqa: BLE001 - isolate per-order SDK/network exceptions
                records.append(_handle_exception(verdict, cid, exc))

        completed_at = self._now()
        return ExecutionReport(
            run_id=run_id,
            session_date=session_date,
            checkpoint=checkpoint,
            started_at=started_at,
            completed_at=completed_at,
            records=records,
            pass1_count=len(pass1),
            pass2_count=len(pass2),
            skipped_unapproved=skipped_unapproved,
            errors=errors,
        )

    @staticmethod
    def _split_hybrid_stop_quantities(signed_qty: float) -> tuple[float, float]:
        """Split signed position qty into ``(broker_gtc_qty, software_frac_qty)``."""
        abs_qty = abs(signed_qty)
        if abs_qty < _MIN_STOP_QTY:
            return (0.0, 0.0)

        if signed_qty < 0.0:
            # Shorts are always whole-share GTC (floor safely instead of rounding up)
            return (float(math.floor(abs_qty + 1e-9)), 0.0)

        whole = math.floor(abs_qty + 1e-9)
        frac = round(max(0.0, abs_qty - whole), 6)
        if frac < _MIN_STOP_QTY:
            return (float(whole), 0.0)
        return (float(whole), frac)

    @staticmethod
    def _resolve_stop_qty_and_tif(
        symbol: str, signed_qty: float, warnings: list[str]
    ) -> tuple[float, TimeInForce]:
        abs_qty = abs(signed_qty)
        whole, frac = AlpacaBroker._split_hybrid_stop_quantities(signed_qty)
        if whole == 0.0 and frac == 0.0:
            warnings.append(
                f"{symbol}: dust position qty {abs_qty:.8f} < {_MIN_STOP_QTY:.0e}; skipping protective stop order"
            )
            return 0.0, TimeInForce.GTC

        if signed_qty < 0.0:
            return whole, TimeInForce.GTC

        if frac == 0.0:
            return whole, TimeInForce.GTC

        if whole >= 1.0:
            warnings.append(
                f"{symbol}: fractional long qty {abs_qty:.6f} floored to {int(whole)} whole shares "
                f"for Alpaca GTC stop; remaining {frac:.6f} fractional share protected via software stop"
            )
            return whole, TimeInForce.GTC

        warnings.append(
            f"{symbol}: fractional long qty {abs_qty:.6f} < 1.0 cannot use Alpaca GTC stop; "
            "protected via software stop"
        )
        return 0.0, TimeInForce.GTC

    @staticmethod
    def _is_stop_at_least_as_protective(
        *, is_long: bool, existing_stop: float, desired_stop: float
    ) -> bool:
        if is_long:
            return existing_stop >= desired_stop * (1.0 - _STOP_PROTECTIVE_TOLERANCE)
        return existing_stop <= desired_stop * (1.0 + _STOP_PROTECTIVE_TOLERANCE)

    def _cancel_and_confirm(
        self, order_id: str, *, timeout_seconds: float = 10.0
    ) -> bool:
        """Cancel an order and poll until it reaches a terminal state."""
        existing = self._fetch_order_by_id(order_id)
        if existing is not None:
            status = _enum_str(getattr(existing, "status", ""))
            if status in _TERMINAL_ORDER_STATUSES:
                return True

        try:
            self.trading_client.cancel_order_by_id(order_id)
        except Exception as exc:  # noqa: BLE001 - check if cancel raced with terminal state
            post_err = self._fetch_order_by_id(order_id)
            if post_err is not None and (
                _enum_str(getattr(post_err, "status", "")) in _TERMINAL_ORDER_STATUSES
            ):
                return True
            logger.warning("alpaca_cancel_order_failed", order_id=order_id, error=str(exc))
            return False

        polled = self._poll_order(
            broker_order_id=order_id,
            client_order_id="",
            timeout_seconds=timeout_seconds,
        )
        if polled is None:
            return False
        return _enum_str(getattr(polled, "status", "")) in _TERMINAL_ORDER_STATUSES

    def sync_protective_stops(
        self,
        desired_stops: Mapping[str, float],
        *,
        session_date: str,
        run_id: str,
    ) -> StopSyncResult:
        """Synchronize resting broker-side protective stop orders with live positions.

        Cancels orphan/wrong-side stop orders, keeps existing stops that are at least as
        protective (never widening a stop), and submits missing or tightened stops.
        Frozen BLACKOUT_PERIOD symbols are left strictly untouched (zero submits, zero cancels).
        """
        validate_trading_mode(self.settings)
        now = self._now()
        desired_map = {k.strip().upper(): float(v) for k, v in desired_stops.items()}
        existing_db_positions = {p["symbol"]: p for p in self.db.get_live_positions()}
        positions = self.get_broker_positions()
        open_orders = self._fetch_open_orders()
        observed_symbols = self._collect_observed_symbols(positions, open_orders)
        blackout = self.resolve_blackout_state(
            now_utc=now,
            observed_broker_symbols=observed_symbols,
        )
        cancel_ack_timeout = max(10.0, self.poll_interval_seconds * 3.0)

        kept_stops: dict[str, str] = {}
        submitted_stops: dict[str, str] = {}
        software_stops: dict[str, float] = {}
        canceled_orphans: list[str] = []
        canceled_stale: list[str] = []
        untouched_blackout_positions: list[str] = []
        untouched_blackout_orders: list[str] = []
        warnings: list[str] = []
        errors: list[str] = []
        cancel_failed_symbols: set[str] = set()

        frozen_stops_by_symbol: dict[str, list[Any]] = {}
        valid_stops_by_symbol: dict[str, list[Any]] = {}
        for order in open_orders:
            sym = str(getattr(order, "symbol", "")).strip().upper()
            oid = str(getattr(order, "id", ""))
            otype = _enum_str(getattr(order, "order_type", None) or getattr(order, "type", None))

            if blackout.halted or sym in blackout.frozen_symbols:
                if oid and oid not in untouched_blackout_orders:
                    untouched_blackout_orders.append(oid)
                if otype in {"stop", "stop_limit"}:
                    frozen_stops_by_symbol.setdefault(sym, []).append(order)
                continue

            if otype not in {"stop", "stop_limit"}:
                continue
            oside = _enum_str(getattr(order, "side", ""))
            pos = positions.get(sym)
            if pos is None:
                if oid:
                    if self._cancel_and_confirm(oid, timeout_seconds=cancel_ack_timeout):
                        canceled_orphans.append(oid)
                    else:
                        errors.append(f"{sym}: failed to confirm cancel of orphan stop {oid}")
                continue
            expected_side = "sell" if pos["qty"] > 0.0 else "buy"
            if oside != expected_side:
                if oid:
                    if self._cancel_and_confirm(oid, timeout_seconds=cancel_ack_timeout):
                        canceled_orphans.append(oid)
                    else:
                        errors.append(f"{sym}: failed to confirm cancel of wrong-side stop {oid}")
                        cancel_failed_symbols.add(sym)
                continue
            valid_stops_by_symbol.setdefault(sym, []).append(order)

        live_pos_rows: list[dict[str, Any]] = []
        if blackout.halted:
            warnings.append(
                f"BLACKOUT_PERIOD halt active ({blackout.halt_reason}); all protective stop operations skipped"
            )

        for sym, pos in sorted(positions.items()):
            signed_qty = pos["qty"]
            if blackout.halted or sym in blackout.frozen_symbols:
                untouched_blackout_positions.append(sym)
                existing_frozen_stops = frozen_stops_by_symbol.get(sym, [])
                expected_side = "sell" if signed_qty > 0.0 else "buy"
                matching_frozen = [
                    o for o in existing_frozen_stops if _enum_str(getattr(o, "side", "")) == expected_side
                ]
                chosen_frozen = matching_frozen[0] if matching_frozen else (
                    existing_frozen_stops[0] if existing_frozen_stops else None
                )
                active_id = str(getattr(chosen_frozen, "id", "")) if chosen_frozen is not None else None
                active_px = (
                    _opt_float(getattr(chosen_frozen, "stop_price", None))
                    if chosen_frozen is not None
                    else None
                )
                if active_id:
                    kept_stops[sym] = active_id
                else:
                    _, frozen_sw_frac = self._split_hybrid_stop_quantities(signed_qty)
                    prev_frozen = existing_db_positions.get(sym)
                    prev_frozen_px = _same_side_db_stop_price(prev_frozen, signed_qty)
                    if prev_frozen_px is not None and (
                        (
                            prev_frozen is not None
                            and prev_frozen.get("stop_order_id") == SOFTWARE_STOP_ORDER_ID
                        )
                        or (
                            math.floor(abs(signed_qty) + 1e-9) <= 0
                            and frozen_sw_frac > 0.0
                        )
                    ):
                        active_id = SOFTWARE_STOP_ORDER_ID
                        active_px = prev_frozen_px
                if not blackout.halted:
                    warnings.append(
                        f"{sym}: frozen by BLACKOUT_PERIOD; resting orders left untouched"
                    )
                live_pos_rows.append(
                    {
                        "symbol": sym,
                        "qty": signed_qty,
                        "avg_entry_price": pos["avg_entry_price"],
                        "market_price": pos["market_price"],
                        "unrealized_pnl": pos["unrealized_pnl"],
                        "stop_order_id": active_id or None,
                        "stop_price": active_px,
                        "updated_at": now,
                    }
                )
                continue
            is_long = signed_qty > 0.0
            protective_side = OrderSide.sell if is_long else OrderSide.buy
            _, software_frac_qty = self._split_hybrid_stop_quantities(signed_qty)
            target_qty, tif = self._resolve_stop_qty_and_tif(sym, signed_qty, warnings)
            desired_px = desired_map.get(sym)
            existing_list = valid_stops_by_symbol.get(sym, [])

            if target_qty <= 0.0 and software_frac_qty > 0.0:
                for cand in existing_list:
                    cand_id = str(getattr(cand, "id", ""))
                    if cand_id:
                        if self._cancel_and_confirm(
                            cand_id, timeout_seconds=cancel_ack_timeout
                        ):
                            canceled_stale.append(cand_id)
                        else:
                            errors.append(
                                f"{sym}: failed to confirm cancel of stray stop {cand_id}"
                            )
                prev_db_stop = _same_side_db_stop_price(
                    existing_db_positions.get(sym), signed_qty
                )
                if desired_px is not None and desired_px > 0.0:
                    if prev_db_stop is not None:
                        raw_stop = max(prev_db_stop, float(desired_px))
                    else:
                        raw_stop = float(desired_px)
                elif prev_db_stop is not None:
                    raw_stop = prev_db_stop
                else:
                    ref = pos["market_price"] if pos["market_price"] > 0 else pos["avg_entry_price"]
                    raw_stop = ref * 0.90
                    warnings.append(
                        f"{sym}: missing desired stop price; synthesized conservative fallback software stop {raw_stop:.2f}"
                    )
                rounded_sw_stop = max(0.01, round(raw_stop, 2))
                software_stops[sym] = rounded_sw_stop
                live_pos_rows.append(
                    {
                        "symbol": sym,
                        "qty": signed_qty,
                        "avg_entry_price": pos["avg_entry_price"],
                        "market_price": pos["market_price"],
                        "unrealized_pnl": pos["unrealized_pnl"],
                        "stop_order_id": SOFTWARE_STOP_ORDER_ID,
                        "stop_price": rounded_sw_stop,
                        "updated_at": now,
                    }
                )
                continue

            if target_qty <= 0.0:
                for cand in existing_list:
                    cand_id = str(getattr(cand, "id", ""))
                    if cand_id and self._cancel_and_confirm(
                        cand_id, timeout_seconds=cancel_ack_timeout
                    ):
                        canceled_stale.append(cand_id)
                live_pos_rows.append(
                    {
                        "symbol": sym,
                        "qty": signed_qty,
                        "avg_entry_price": pos["avg_entry_price"],
                        "market_price": pos["market_price"],
                        "unrealized_pnl": pos["unrealized_pnl"],
                        "stop_order_id": None,
                        "stop_price": None,
                        "updated_at": now,
                    }
                )
                continue

            matched_order: Any | None = None
            for cand in existing_list:
                cand_qty = _open_order_remaining_qty(cand)
                cand_stop = _opt_float(getattr(cand, "stop_price", None))
                if cand_stop is None or cand_stop <= 0.0:
                    continue
                if abs(cand_qty - target_qty) > 1e-6:
                    continue
                if desired_px is None or self._is_stop_at_least_as_protective(
                    is_long=is_long,
                    existing_stop=cand_stop,
                    desired_stop=desired_px,
                ):
                    if matched_order is None:
                        matched_order = cand
                    else:
                        # Keep the tighter of two matching stops, cancel duplicate
                        matched_stop = _req_float(getattr(matched_order, "stop_price", 0.0))
                        cand_is_tighter = (
                            cand_stop > matched_stop if is_long else cand_stop < matched_stop
                        )
                        if cand_is_tighter:
                            stale_id = str(getattr(matched_order, "id", ""))
                            if stale_id:
                                if self._cancel_and_confirm(
                                    stale_id, timeout_seconds=cancel_ack_timeout
                                ):
                                    canceled_stale.append(stale_id)
                                else:
                                    errors.append(
                                        f"{sym}: failed to confirm cancel of duplicate stop {stale_id}"
                                    )
                            matched_order = cand
                        else:
                            cand_id = str(getattr(cand, "id", ""))
                            if cand_id:
                                if self._cancel_and_confirm(
                                    cand_id, timeout_seconds=cancel_ack_timeout
                                ):
                                    canceled_stale.append(cand_id)
                                else:
                                    errors.append(
                                        f"{sym}: failed to confirm cancel of duplicate stop {cand_id}"
                                    )

            # Cancel any existing stop orders that didn't qualify (confirm terminal first!)
            for cand in existing_list:
                if matched_order is not None and cand is matched_order:
                    continue
                cand_id = str(getattr(cand, "id", ""))
                if not cand_id or cand_id in canceled_stale:
                    continue
                if self._cancel_and_confirm(cand_id, timeout_seconds=cancel_ack_timeout):
                    canceled_stale.append(cand_id)
                else:
                    cancel_failed_symbols.add(sym)
                    errors.append(
                        f"{sym}: failed to confirm cancel of stale stop {cand_id}; aborting replacement"
                    )

            active_stop_id: str | None = None
            active_stop_px: float | None = None

            if matched_order is not None:
                active_stop_id = str(getattr(matched_order, "id", ""))
                active_stop_px = _opt_float(getattr(matched_order, "stop_price", None))
                kept_stops[sym] = active_stop_id
            elif sym in cancel_failed_symbols:
                # Stale/wrong-side stop did not confirm cancel; skip replacement to prevent collision
                pass
            else:
                if desired_px is None:
                    ref = pos["market_price"] if pos["market_price"] > 0 else pos["avg_entry_price"]
                    raw_stop = ref * (0.90 if is_long else 1.10)
                    warnings.append(
                        f"{sym}: missing desired stop price; synthesized conservative fallback stop {raw_stop:.2f}"
                    )
                else:
                    raw_stop = desired_px
                prev_db_stop = _same_side_db_stop_price(
                    existing_db_positions.get(sym), signed_qty
                )
                if prev_db_stop is not None:
                    raw_stop = (
                        max(prev_db_stop, raw_stop) if is_long else min(prev_db_stop, raw_stop)
                    )
                rounded_stop = max(0.01, round(raw_stop, 2))
                leg_tag = hashlib.sha1(
                    f"{run_id}:{rounded_stop:.2f}:{target_qty:.6f}".encode()
                ).hexdigest()[:6]

                # HIGH #4: Never adopt a terminal order; disambiguate client_order_id if prior
                # order with same id is already terminal.
                chosen_stop_cid: str | None = None
                sub_stop: Any | None = None
                for attempt in range(20):
                    leg_name = f"stp{leg_tag}" if attempt == 0 else f"stp{leg_tag}s{attempt}"
                    cand_cid = build_client_order_id(
                        session_date,
                        "stop",
                        sym,
                        protective_side,
                        leg=leg_name,
                    )
                    existing_stop_order = self._fetch_order_by_client_id(cand_cid)
                    if existing_stop_order is not None:
                        existing_status = _enum_str(getattr(existing_stop_order, "status", ""))
                        if existing_status in _OPEN_ORDER_STATUSES:
                            chosen_stop_cid = cand_cid
                            sub_stop = existing_stop_order
                            break
                        continue
                    chosen_stop_cid = cand_cid
                    break

                if chosen_stop_cid is None:
                    errors.append(
                        f"{sym}: exhausted client_order_id disambiguation attempts for stop order"
                    )
                else:
                    try:
                        if sub_stop is None:
                            stop_req = StopOrderRequest(
                                symbol=sym,
                                qty=target_qty,
                                side=_to_alpaca_side(protective_side),
                                time_in_force=tif,
                                stop_price=rounded_stop,
                                client_order_id=chosen_stop_cid,
                            )
                            sub_stop = self.trading_client.submit_order(stop_req)
                        active_stop_id = str(getattr(sub_stop, "id", ""))
                        active_stop_px = _opt_float(getattr(sub_stop, "stop_price", rounded_stop))
                        submitted_stops[sym] = active_stop_id
                        ref_px = (
                            pos["market_price"]
                            if pos["market_price"] > 0
                            else pos["avg_entry_price"]
                        ) or rounded_stop
                        self._persist_broker_order(
                            sub_stop,
                            client_order_id=chosen_stop_cid,
                            run_id=run_id,
                            symbol=sym,
                            side=protective_side,
                            order_type="stop",
                            qty=target_qty,
                            reference_price=ref_px,
                            is_risk_reducing=True,
                            stop_price=rounded_stop,
                        )
                    except Exception as exc:  # noqa: BLE001 - record per-symbol stop failure without aborting batch
                        err = f"{sym}: failed to submit {tif.value.upper()} stop order: {exc}"
                        errors.append(err)
                        logger.warning("alpaca_stop_submit_failed", symbol=sym, error=str(exc))

            persisted_stop_px = active_stop_px
            if persisted_stop_px is None or persisted_stop_px <= 0.0:
                stop_candidates: list[float] = []
                if desired_px is not None and math.isfinite(desired_px) and desired_px > 0.0:
                    stop_candidates.append(float(desired_px))
                prev_db_stop = _same_side_db_stop_price(
                    existing_db_positions.get(sym), signed_qty
                )
                if prev_db_stop is not None:
                    stop_candidates.append(prev_db_stop)
                if stop_candidates:
                    persisted_stop_px = max(stop_candidates) if is_long else min(stop_candidates)
            if (
                persisted_stop_px is not None
                and math.isfinite(float(persisted_stop_px))
                and float(persisted_stop_px) > 0.0
            ):
                persisted_stop_px = max(0.01, round(float(persisted_stop_px), 2))
            else:
                persisted_stop_px = None

            if software_frac_qty > 0.0 and persisted_stop_px is not None:
                software_stops[sym] = persisted_stop_px

            live_pos_rows.append(
                {
                    "symbol": sym,
                    "qty": signed_qty,
                    "avg_entry_price": pos["avg_entry_price"],
                    "market_price": pos["market_price"],
                    "unrealized_pnl": pos["unrealized_pnl"],
                    "stop_order_id": active_stop_id,
                    "stop_price": persisted_stop_px,
                    "updated_at": now,
                }
            )

        self.db.replace_live_positions(live_pos_rows)
        return StopSyncResult(
            session_date=session_date,
            run_id=run_id,
            kept_stops=kept_stops,
            submitted_stops=submitted_stops,
            software_stops=software_stops,
            canceled_orphans=canceled_orphans,
            canceled_stale=canceled_stale,
            untouched_blackout_positions=sorted(untouched_blackout_positions),
            untouched_blackout_orders=sorted(untouched_blackout_orders),
            warnings=warnings,
            errors=errors,
        )

    def reconcile_with_broker(
        self,
        desired_stops: Mapping[str, float] | None = None,
        *,
        daemon_name: str = "agy-trader",
        repair_missing_stops: bool = True,
        fallback_stop_pct: float = 0.10,
        record_if_clean: bool = False,
    ) -> ReconciliationResult:
        """Diff broker ground truth against SQLite, update open orders, and repair stops."""
        validate_trading_mode(self.settings)
        now = self._now()
        broker_positions = self.get_broker_positions()
        db_positions = {p["symbol"]: p for p in self.db.get_live_positions()}
        open_orders = self._fetch_open_orders()
        observed_symbols = self._collect_observed_symbols(broker_positions, open_orders)
        blackout = self.resolve_blackout_state(
            now_utc=now,
            observed_broker_symbols=observed_symbols,
        )

        untouched_blackout_positions = (
            sorted(broker_positions.keys())
            if blackout.halted
            else sorted(sym for sym in broker_positions if sym in blackout.frozen_symbols)
        )
        untouched_blackout_orders = sorted(
            {
                str(getattr(order, "id", ""))
                for order in open_orders
                if getattr(order, "id", None)
                and (
                    blackout.halted
                    or str(getattr(order, "symbol", "")).strip().upper() in blackout.frozen_symbols
                )
            }
        )

        # 1. Diff positions between SQLite and broker ground truth
        position_diffs: list[dict[str, Any]] = []
        all_syms = sorted(set(broker_positions) | set(db_positions))
        for sym in all_syms:
            b_pos = broker_positions.get(sym)
            d_pos = db_positions.get(sym)
            if b_pos is None and d_pos is not None:
                position_diffs.append(
                    {"symbol": sym, "kind": "closed_on_broker", "db_qty": d_pos["qty"], "broker_qty": 0.0}
                )
            elif b_pos is not None and d_pos is None:
                position_diffs.append(
                    {"symbol": sym, "kind": "opened_on_broker", "db_qty": 0.0, "broker_qty": b_pos["qty"]}
                )
            elif b_pos is not None and d_pos is not None and abs(b_pos["qty"] - d_pos["qty"]) > 1e-6:
                position_diffs.append(
                    {
                        "symbol": sym,
                        "kind": "qty_mismatch",
                        "db_qty": d_pos["qty"],
                        "broker_qty": b_pos["qty"],
                    }
                )

        # 2. Update any open live_orders statuses from broker
        updated_orders: list[str] = []
        open_by_cid = {
            str(getattr(o, "client_order_id", "")): o
            for o in open_orders
            if getattr(o, "client_order_id", None)
        }
        for db_ord in self.db.get_open_live_orders():
            cid = db_ord["client_order_id"]
            broker_ord = open_by_cid.get(cid)
            if broker_ord is None:
                broker_ord = self._fetch_order_by_client_id(cid)
            if broker_ord is None:
                continue
            new_status = _enum_str(getattr(broker_ord, "status", ""))
            new_fqty = _req_float(getattr(broker_ord, "filled_qty", 0.0))
            new_fpx = _opt_float(getattr(broker_ord, "filled_avg_price", None))
            if (
                new_status != db_ord["status"]
                or abs(new_fqty - float(db_ord["filled_qty"])) > 1e-6
                or new_fpx != db_ord["filled_avg_price"]
            ):
                side = OrderSide(db_ord["side"])
                self._persist_broker_order(
                    broker_ord,
                    client_order_id=cid,
                    run_id=str(db_ord["run_id"]),
                    symbol=str(db_ord["symbol"]),
                    side=side,
                    order_type=str(db_ord["order_type"]),
                    qty=float(db_ord["qty"]),
                    reference_price=float(db_ord["reference_price"]),
                    is_risk_reducing=bool(db_ord["is_risk_reducing"]),
                )
                updated_orders.append(cid)

        # 3. Inspect resting stops for missing, under-sized, orphan, wrong-side, or duplicate stops
        same_side_stops: dict[str, list[Any]] = {}
        orphan_stop_symbols: list[str] = []
        wrong_side_stop_symbols: list[str] = []
        for order in open_orders:
            otype = _enum_str(getattr(order, "order_type", None) or getattr(order, "type", None))
            if otype not in {"stop", "stop_limit"}:
                continue
            sym = str(getattr(order, "symbol", "")).strip().upper()
            pos = broker_positions.get(sym)
            if pos is None:
                orphan_stop_symbols.append(sym)
                continue
            expected_side = "sell" if pos["qty"] > 0.0 else "buy"
            if _enum_str(getattr(order, "side", "")) != expected_side:
                wrong_side_stop_symbols.append(sym)
                continue
            same_side_stops.setdefault(sym, []).append(order)

        missing_stop_symbols: list[str] = []
        undersized_stop_symbols: list[str] = []
        duplicate_stop_symbols: list[str] = []
        resting_stops: dict[str, Any] = {}
        for sym, pos in sorted(broker_positions.items()):
            target_qty, software_frac_qty = self._split_hybrid_stop_quantities(pos["qty"])
            if target_qty <= 0.0 and software_frac_qty > 0.0:
                sym_stops = same_side_stops.get(sym, [])
                if sym_stops:
                    undersized_stop_symbols.append(sym)
                    resting_stops[sym] = sym_stops[0]
                if _same_side_db_stop_price(db_positions.get(sym), pos["qty"]) is None:
                    missing_stop_symbols.append(sym)
                continue
            if target_qty <= 0.0:
                continue
            sym_stops = same_side_stops.get(sym, [])
            if not sym_stops:
                missing_stop_symbols.append(sym)
                continue
            if len(sym_stops) > 1:
                duplicate_stop_symbols.append(sym)
            matching_qty_stops = [
                s
                for s in sym_stops
                if abs(_open_order_remaining_qty(s) - target_qty) <= 1e-6
            ]
            if not matching_qty_stops:
                undersized_stop_symbols.append(sym)
                resting_stops[sym] = sym_stops[0]
            else:
                resting_stops[sym] = matching_qty_stops[0]

        # 4. Sync/repair protective stops or overwrite live_positions with broker ground truth
        stop_sync_result: StopSyncResult | None = None
        session_date = now.date().isoformat()
        run_id = f"reconcile-{now.strftime('%Y%m%dT%H%M%SZ')}"
        non_frozen_missing = [
            sym
            for sym in missing_stop_symbols
            if not blackout.halted and sym not in blackout.frozen_symbols
        ]
        non_frozen_undersized = [
            sym
            for sym in undersized_stop_symbols
            if not blackout.halted and sym not in blackout.frozen_symbols
        ]
        non_frozen_anomalies = any(
            not blackout.halted and sym not in blackout.frozen_symbols
            for sym in (
                orphan_stop_symbols
                + wrong_side_stop_symbols
                + undersized_stop_symbols
                + duplicate_stop_symbols
            )
        )
        if desired_stops is not None:
            stop_sync_result = self.sync_protective_stops(
                desired_stops, session_date=session_date, run_id=run_id
            )
        elif (
            repair_missing_stops
            and not blackout.halted
            and (non_frozen_missing or non_frozen_anomalies)
        ):
            repair_stops: dict[str, float] = {}
            for sym in non_frozen_missing + non_frozen_undersized:
                existing_sym_stops = same_side_stops.get(sym, [])
                existing_stop_px = (
                    _opt_float(getattr(existing_sym_stops[0], "stop_price", None))
                    if existing_sym_stops
                    else None
                )
                if existing_stop_px is not None and existing_stop_px > 0.0:
                    repair_stops[sym] = existing_stop_px
                    continue
                pos = broker_positions[sym]
                same_side_prev_px = _same_side_db_stop_price(
                    db_positions.get(sym), pos["qty"]
                )
                if same_side_prev_px is not None:
                    repair_stops[sym] = same_side_prev_px
                    continue
                ref = pos["market_price"] if pos["market_price"] > 0 else pos["avg_entry_price"]
                if ref > 0.0:
                    mult = (
                        (1.0 - fallback_stop_pct)
                        if pos["qty"] > 0.0
                        else (1.0 + fallback_stop_pct)
                    )
                    repair_stops[sym] = max(0.01, round(ref * mult, 2))
            stop_sync_result = self.sync_protective_stops(
                repair_stops, session_date=session_date, run_id=run_id
            )

        if stop_sync_result is None:
            live_pos_rows = []
            for sym, pos in sorted(broker_positions.items()):
                row_stop_id: str | None = None
                row_stop_px: float | None = None
                if sym in resting_stops:
                    row_stop_id = str(getattr(resting_stops[sym], "id", "")) or None
                    row_stop_px = _opt_float(getattr(resting_stops[sym], "stop_price", None))
                else:
                    target_qty, software_frac_qty = self._split_hybrid_stop_quantities(pos["qty"])
                    d_pos = db_positions.get(sym)
                    same_side_px = _same_side_db_stop_price(d_pos, pos["qty"])
                    is_frozen = blackout.halted or sym in blackout.frozen_symbols
                    if same_side_px is not None and (
                        (not is_frozen and target_qty <= 0.0 and software_frac_qty > 0.0)
                        or (
                            is_frozen
                            and (
                                (
                                    d_pos is not None
                                    and d_pos.get("stop_order_id") == SOFTWARE_STOP_ORDER_ID
                                )
                                or (target_qty <= 0.0 and software_frac_qty > 0.0)
                            )
                        )
                    ):
                        row_stop_id = SOFTWARE_STOP_ORDER_ID
                        row_stop_px = same_side_px
                live_pos_rows.append(
                    {
                        "symbol": sym,
                        "qty": pos["qty"],
                        "avg_entry_price": pos["avg_entry_price"],
                        "market_price": pos["market_price"],
                        "unrealized_pnl": pos["unrealized_pnl"],
                        "stop_order_id": row_stop_id,
                        "stop_price": row_stop_px,
                        "updated_at": now,
                    }
                )
            self.db.replace_live_positions(live_pos_rows)

        has_changes = bool(
            position_diffs
            or updated_orders
            or non_frozen_missing
            or non_frozen_anomalies
            or (
                stop_sync_result is not None
                and (
                    stop_sync_result.submitted_stops
                    or stop_sync_result.canceled_orphans
                    or stop_sync_result.canceled_stale
                    or stop_sync_result.errors
                )
            )
        )
        if has_changes or record_if_clean:
            self.db.record_daemon_event(
                daemon_name=daemon_name,
                event_type="reconcile",
                details={
                    "broker_positions_count": len(broker_positions),
                    "db_positions_count": len(db_positions),
                    "position_diffs": position_diffs,
                    "updated_orders": updated_orders,
                    "missing_stop_symbols": missing_stop_symbols,
                    "undersized_stop_symbols": sorted(set(undersized_stop_symbols)),
                    "orphan_stop_symbols": sorted(set(orphan_stop_symbols)),
                    "wrong_side_stop_symbols": sorted(set(wrong_side_stop_symbols)),
                    "duplicate_stop_symbols": sorted(set(duplicate_stop_symbols)),
                    "repaired_stops": (
                        sorted(stop_sync_result.submitted_stops) if stop_sync_result else []
                    ),
                    "software_stops": (
                        sorted(stop_sync_result.software_stops) if stop_sync_result else []
                    ),
                    "untouched_blackout_positions": untouched_blackout_positions,
                    "untouched_blackout_orders": untouched_blackout_orders,
                },
                timestamp=now,
            )

        return ReconciliationResult(
            timestamp=now,
            daemon_name=daemon_name,
            broker_positions_count=len(broker_positions),
            db_positions_count=len(db_positions),
            position_diffs=position_diffs,
            updated_orders=updated_orders,
            missing_stop_symbols=missing_stop_symbols,
            untouched_blackout_positions=untouched_blackout_positions,
            untouched_blackout_orders=untouched_blackout_orders,
            stop_sync_result=stop_sync_result,
        )

    def evaluate_and_execute_software_stops(
        self,
        live_marks: Mapping[str, float],
        *,
        session_date: str,
        run_id: str,
        market_open: bool = False,
        symbols: Iterable[str] | None = None,
    ) -> SoftwareStopResult:
        """Evaluate and execute daemon-managed software stops on unprotected fractional longs.

        Protects sub-1.0 fractional long positions, the fractional remainder of ``N + f``
        hybrid longs, and any long position with a failed/missing broker stop that still has
        a persisted ``stop_price`` during regular market hours while respecting
        ``BLACKOUT_PERIOD`` freezes, open sell/stop quantities, ``qty_available``, and
        cross-tick idempotent ``client_order_id`` disambiguation.
        """
        validate_trading_mode(self.settings)
        if not market_open:
            return SoftwareStopResult(session_date=session_date, run_id=run_id)

        # CRITICAL #1: Fetch open_orders BEFORE broker_positions to avoid TOCTOU double-sell
        open_orders = self._fetch_open_orders()
        broker_positions = self.get_broker_positions()
        db_positions = {p["symbol"]: p for p in self.db.get_live_positions()}
        observed_symbols = self._collect_observed_symbols(broker_positions, open_orders)
        blackout = self.resolve_blackout_state(
            now_utc=self._now(),
            observed_broker_symbols=observed_symbols,
        )
        if blackout.halted:
            return SoftwareStopResult(
                session_date=session_date,
                run_id=run_id,
                untouched_blackout_positions=sorted(broker_positions.keys()),
            )

        target_symbol_set = (
            {str(s).strip().upper() for s in symbols if str(s).strip()}
            if symbols is not None
            else None
        )

        open_sell_qty_by_sym: dict[str, float] = defaultdict(float)
        open_sell_stop_px_by_sym: dict[str, float] = {}
        for order in open_orders:
            sym = str(getattr(order, "symbol", "")).strip().upper()
            if not sym:
                continue
            if _enum_str(getattr(order, "side", "")) != "sell":
                continue
            rem = _open_order_remaining_qty(order)
            open_sell_qty_by_sym[sym] += rem
            otype = _enum_str(getattr(order, "order_type", None) or getattr(order, "type", None))
            if otype in {"stop", "stop_limit"}:
                spx = _opt_float(getattr(order, "stop_price", None))
                if spx is not None and spx > 0.0:
                    prev_spx = open_sell_stop_px_by_sym.get(sym)
                    if prev_spx is None or spx > prev_spx:
                        open_sell_stop_px_by_sym[sym] = spx

        normalized_marks: dict[str, float] = {}
        for k, v in live_marks.items():
            clean_k = str(k).strip().upper()
            parsed_v = _opt_float(v)
            if clean_k and parsed_v is not None and parsed_v > 0.0:
                normalized_marks[clean_k] = parsed_v

        checked_symbols: list[str] = []
        triggered: list[SoftwareStopExecution] = []
        untouched_blackout_positions: list[str] = []
        errors: list[str] = []

        for sym, pos in sorted(broker_positions.items()):
            if target_symbol_set is not None and sym not in target_symbol_set:
                continue
            signed_qty = float(pos["qty"])
            if signed_qty <= _MIN_STOP_QTY:
                continue

            _, software_frac_qty = self._split_hybrid_stop_quantities(signed_qty)
            db_pos = db_positions.get(sym)
            same_side_db_stop = _same_side_db_stop_price(db_pos, signed_qty)
            is_watched = (
                software_frac_qty > 0.0
                or (db_pos is not None and db_pos.get("stop_order_id") == SOFTWARE_STOP_ORDER_ID)
                or (
                    db_pos is not None
                    and db_pos.get("stop_order_id") is None
                    and same_side_db_stop is not None
                )
            )
            if not is_watched:
                continue

            if sym in blackout.frozen_symbols:
                untouched_blackout_positions.append(sym)
                continue

            checked_symbols.append(sym)

            stop_candidates: list[float] = []
            open_spx = open_sell_stop_px_by_sym.get(sym)
            if open_spx is not None and open_spx > 0.0:
                stop_candidates.append(open_spx)
            if same_side_db_stop is not None:
                stop_candidates.append(same_side_db_stop)
            if not stop_candidates:
                continue
            stop_px = max(stop_candidates)

            mark_px = normalized_marks.get(sym)
            if mark_px is None:
                pos_mkt = _opt_float(pos.get("market_price"))
                if pos_mkt is not None and pos_mkt > 0.0:
                    mark_px = pos_mkt
            if mark_px is None:
                continue

            if mark_px > stop_px:
                continue

            qty_available = float(pos.get("qty_available", signed_qty))
            unprotected_qty = round(
                min(
                    max(0.0, signed_qty - open_sell_qty_by_sym.get(sym, 0.0)),
                    max(0.0, qty_available),
                ),
                6,
            )
            if unprotected_qty <= _MIN_STOP_QTY:
                continue

            # HIGH #8: Skip < $1 partial fractional remainders while whole shares still rest via GTC
            if (
                unprotected_qty < signed_qty - _MIN_STOP_QTY
                and unprotected_qty * mark_px < 1.0
            ):
                logger.info(
                    "alpaca_software_stop_sub_dollar_partial_skip",
                    symbol=sym,
                    unprotected_qty=unprotected_qty,
                    mark_price=mark_px,
                    notional=round(unprotected_qty * mark_px, 4),
                )
                continue

            # Re-check blackout immediately before submitting (defense-in-depth)
            blackout_pre = self.resolve_blackout_state(
                now_utc=self._now(),
                observed_broker_symbols=observed_symbols,
            )
            if blackout_pre.halted or sym in blackout_pre.frozen_symbols:
                untouched_blackout_positions.append(sym)
                continue

            # HIGH #8: Key leg_tag on stop_px and unprotected_qty (NOT run_id) for cross-tick idempotency
            leg_tag = hashlib.sha1(
                f"{stop_px:.2f}:{unprotected_qty:.6f}".encode()
            ).hexdigest()[:6]
            chosen_cid: str | None = None
            sub_order: Any | None = None
            for attempt in range(20):
                leg_name = f"sw{leg_tag}" if attempt == 0 else f"sw{leg_tag}s{attempt}"
                cand_cid = build_client_order_id(
                    session_date,
                    "swstp",
                    sym,
                    OrderSide.sell,
                    leg=leg_name,
                )
                existing_order = self._fetch_order_by_client_id(cand_cid)
                if existing_order is not None:
                    existing_status = _enum_str(getattr(existing_order, "status", ""))
                    if existing_status in _OPEN_ORDER_STATUSES:
                        chosen_cid = cand_cid
                        sub_order = existing_order
                        break
                    continue
                chosen_cid = cand_cid
                break

            if chosen_cid is None:
                errors.append(
                    f"{sym}: exhausted client_order_id disambiguation attempts for software stop"
                )
                continue

            try:
                if sub_order is None:
                    mkt_req = MarketOrderRequest(
                        symbol=sym,
                        qty=unprotected_qty,
                        side=AlpacaOrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                        client_order_id=chosen_cid,
                    )
                    sub_order = self.trading_client.submit_order(mkt_req)
                b_id, status, f_qty, f_px, _ = self._persist_broker_order(
                    sub_order,
                    client_order_id=chosen_cid,
                    run_id=run_id,
                    symbol=sym,
                    side=OrderSide.sell,
                    order_type="market",
                    qty=unprotected_qty,
                    reference_price=mark_px,
                    is_risk_reducing=True,
                    stop_price=stop_px,
                )
                if status not in _TERMINAL_ORDER_STATUSES:
                    polled = self._poll_order(
                        broker_order_id=b_id,
                        client_order_id=chosen_cid,
                        timeout_seconds=min(5.0, self.limit_timeout_seconds),
                    )
                    if polled is not None:
                        b_id, status, f_qty, f_px, _ = self._persist_broker_order(
                            polled,
                            client_order_id=chosen_cid,
                            run_id=run_id,
                            symbol=sym,
                            side=OrderSide.sell,
                            order_type="market",
                            qty=unprotected_qty,
                            reference_price=mark_px,
                            is_risk_reducing=True,
                            stop_price=stop_px,
                        )
                triggered.append(
                    SoftwareStopExecution(
                        symbol=sym,
                        qty=unprotected_qty,
                        trigger_price=mark_px,
                        stop_price=stop_px,
                        client_order_id=chosen_cid,
                        broker_order_id=b_id,
                        status=status,
                        filled_qty=f_qty,
                        filled_avg_price=f_px,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - isolate per-symbol software stop error
                err_msg = f"{sym}: failed to execute software stop market sell: {exc}"
                errors.append(err_msg)
                logger.warning("alpaca_software_stop_failed", symbol=sym, error=str(exc))

        return SoftwareStopResult(
            session_date=session_date,
            run_id=run_id,
            checked_symbols=checked_symbols,
            triggered=triggered,
            untouched_blackout_positions=sorted(set(untouched_blackout_positions)),
            errors=errors,
        )

    def emergency_flatten(
        self,
        *,
        confirm: bool,
        reason: str,
        actor: str = "operator",
        override_blackout: bool = False,
    ) -> dict[str, Any]:
        """Pause new entries, cancel open orders, and market-close open positions.

        Respects BLACKOUT_PERIOD by leaving frozen positions and resting orders untouched
        unless override_blackout=True is explicitly supplied.
        """
        if not confirm:
            raise ValueError("emergency_flatten requires confirm=True")
        clean_reason = reason.strip()
        if not clean_reason:
            raise ValueError("A non-empty reason is required for emergency_flatten")

        validate_trading_mode(self.settings)
        now = self._now()

        # Pause new entries immediately so daemons do not re-enter after flatten
        dials = ControlPlane(self.db).set_pause(
            True,
            actor=actor,
            reason=f"[EMERGENCY_FLATTEN] {clean_reason}",
            now=now,
        )

        open_orders = self._fetch_open_orders()
        positions = self.get_broker_positions()
        observed_symbols = self._collect_observed_symbols(positions, open_orders)
        blackout = self.resolve_blackout_state(
            now_utc=now,
            observed_broker_symbols=observed_symbols,
        )

        errors: list[str] = []
        skipped_blackout_symbols: list[str] = []
        untouched_blackout_orders: list[str] = []
        canceled_count = 0
        close_orders: list[dict[str, Any]] = []

        if not override_blackout and blackout.halted:
            skipped_blackout_symbols = sorted(positions.keys())
            untouched_blackout_orders = sorted(
                {str(getattr(o, "id", "")) for o in open_orders if getattr(o, "id", None)}
            )
            errors.append(
                f"BLACKOUT_PERIOD halt active ({blackout.halt_reason}); "
                "refusing emergency_flatten without override_blackout=True"
            )
        else:
            frozen_set = (
                frozenset() if override_blackout else blackout.frozen_symbols
            )
            cancel_ack_timeout = max(10.0, self.poll_interval_seconds * 3.0)

            if frozen_set:
                non_frozen_open_ids: list[str] = []
                for order in open_orders:
                    sym = str(getattr(order, "symbol", "")).strip().upper()
                    oid = str(getattr(order, "id", ""))
                    if not oid:
                        continue
                    if sym in frozen_set:
                        untouched_blackout_orders.append(oid)
                    else:
                        non_frozen_open_ids.append(oid)

                for oid in non_frozen_open_ids:
                    if self._cancel_and_confirm(oid, timeout_seconds=cancel_ack_timeout):
                        canceled_count += 1
                    else:
                        errors.append(f"failed to confirm cancel of non-blackout order {oid}")
                untouched_blackout_orders = sorted(set(untouched_blackout_orders))
            else:
                canceled = self.trading_client.cancel_orders()
                canceled_count = len(canceled) if isinstance(canceled, Sequence) else 0

                # HIGH #3: Poll open orders until cancel acknowledgement drains before submitting closes
                cancel_started = self._now()
                while True:
                    remaining_open = self._fetch_open_orders()
                    if not remaining_open:
                        break
                    elapsed = (self._now() - cancel_started).total_seconds()
                    if elapsed >= cancel_ack_timeout:
                        open_ids = [str(getattr(o, "id", "")) for o in remaining_open]
                        errors.append(
                            f"cancel_orders timeout: {len(remaining_open)} open order(s) remained active ({open_ids})"
                        )
                        logger.warning(
                            "alpaca_emergency_flatten_cancel_timeout", open_order_ids=open_ids
                        )
                        break
                    self._sleep(self.poll_interval_seconds)

            positions = self.get_broker_positions()
            run_id = f"flatten-{now.strftime('%Y%m%dT%H%M%SZ')}"
            session_date = now.date().isoformat()
            checkpoint = now.strftime("%H%M")

            for sym, pos in sorted(positions.items()):
                if sym in frozen_set:
                    skipped_blackout_symbols.append(sym)
                    continue
                signed_qty = pos["qty"]
                close_side = OrderSide.sell if signed_qty > 0.0 else OrderSide.buy
                close_qty = abs(signed_qty)
                ref_px = pos["market_price"] if pos["market_price"] > 0 else pos["avg_entry_price"]
                if ref_px <= 0.0:
                    ref_px = 1.0
                cid = build_client_order_id(
                    session_date,
                    checkpoint,
                    sym,
                    close_side,
                    leg=f"f{uuid.uuid4().hex[:6]}",
                )
                try:
                    req = MarketOrderRequest(
                        symbol=sym,
                        qty=close_qty,
                        side=_to_alpaca_side(close_side),
                        time_in_force=TimeInForce.DAY,
                        client_order_id=cid,
                    )
                    submitted = self.trading_client.submit_order(req)
                    b_id, status, f_qty, f_px, _ = self._persist_broker_order(
                        submitted,
                        client_order_id=cid,
                        run_id=run_id,
                        symbol=sym,
                        side=close_side,
                        order_type="market",
                        qty=close_qty,
                        reference_price=ref_px,
                        is_risk_reducing=True,
                    )
                    close_orders.append(
                        {
                            "symbol": sym,
                            "side": close_side.value,
                            "qty": close_qty,
                            "client_order_id": cid,
                            "broker_order_id": b_id,
                            "status": status,
                            "filled_qty": f_qty,
                            "filled_avg_price": f_px,
                        }
                    )
                except Exception as exc:
                    errors.append(f"{sym}: {exc}")
                    logger.exception("alpaca_emergency_flatten_close_error", symbol=sym)

        self.db.record_daemon_event(
            daemon_name="cli",
            event_type="emergency_flatten",
            details={
                "actor": actor,
                "reason": clean_reason,
                "override_blackout": override_blackout,
                "canceled_orders": canceled_count,
                "close_orders": close_orders,
                "skipped_blackout_symbols": skipped_blackout_symbols,
                "untouched_blackout_orders": untouched_blackout_orders,
                "errors": errors,
            },
            timestamp=now,
        )

        return {
            "ok": len(errors) == 0,
            "paper_mode": self.settings.alpaca_paper,
            "paused_new_entries": dials.pause_new_entries,
            "override_blackout": override_blackout,
            "canceled_orders": canceled_count,
            "close_orders": close_orders,
            "skipped_blackout_symbols": skipped_blackout_symbols,
            "untouched_blackout_orders": untouched_blackout_orders,
            "errors": errors,
            "actor": actor,
            "reason": clean_reason,
            "timestamp": now.isoformat(),
        }
