from __future__ import annotations

import math
import socket
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockSnapshotRequest

from agy_ops.notifications import (
    DEFAULT_DISCORD_FOOTER,
    DiscordWebhookNotifier,
    build_blackout_alert_discord_embed,
)
from agy_trader.core.blackout import (
    ISSUER_SHARE_CLASS_GROUPS,
    BlackoutResolution,
    EnvBlackoutTracker,
)
from agy_trader.core.calendar import is_nyse_trading_day, nyse_trading_days_apart
from agy_trader.core.config import (
    AppConfig,
    EnvSettings,
    load_app_config,
    load_universe,
    validate_trading_mode,
)
from agy_trader.core.control_plane import ControlPlane
from agy_trader.core.models import (
    AssetRiskMetadata,
    AssetType,
    ControlDials,
    MacroPoint,
    NewsArticle,
    OrderSide,
    UniverseSymbol,
    ensure_utc,
)
from agy_trader.data.macro_and_calendar import FRED_SERIES_IDS, fetch_yfinance_latest_mark
from agy_trader.execution.alpaca_broker import AlpacaBroker
from agy_trader.risk.engine import HardRiskEngine, PortfolioRiskState
from agy_trader.storage.db import StateDB
from agy_trader.storage.parquet_store import ParquetBarStore
from agy_trader.strategy.features import (
    SAFE_HAVEN_SECTORS,
    SAFE_HAVEN_SYMBOLS,
    FeatureEngine,
)
from agy_trader.strategy.portfolio import PortfolioConstructor

from .common import (
    NY_TZ,
    DaemonRunner,
    check_market_open,
    get_market_clock_info,
    release_memory,
    wait_for_network,
)

logger = structlog.get_logger(__name__)
_TRAILING_BAR_ROWS = 360


def active_scorer_model(settings: EnvSettings) -> str:
    return "ProsusAI/finbert" if settings.nlp_backend == "finbert" else "lexicon-v1"


@dataclass(frozen=True, slots=True)
class _IntradaySnapshot:
    price: float
    timestamp: datetime | None = None
    daily_open: float | None = None
    daily_high: float | None = None
    daily_low: float | None = None
    daily_volume: float | None = None
    daily_vwap: float | None = None


def _opt_pos_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    f = float(val)
    return f if math.isfinite(f) and f > 0.0 else None


def _opt_nonneg_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    f = float(val)
    return f if math.isfinite(f) and f >= 0.0 else None


def _parse_ts_any(raw: Any) -> datetime | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, datetime):
        return ensure_utc(raw)
    try:
        return ensure_utc(pd.Timestamp(raw).to_pydatetime())
    except Exception:  # noqa: BLE001
        return None


def _get_field(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj and obj[name] is not None:
                return obj[name]
        else:
            val = getattr(obj, name, None)
            if val is not None:
                return val
    return None


def _extract_intraday_snapshot(snap: Any) -> _IntradaySnapshot | None:
    if snap is None:
        return None

    candidates: list[tuple[int, float, datetime | None]] = []

    trade = _get_field(snap, "latest_trade")
    if trade is not None:
        px = _opt_pos_float(_get_field(trade, "price", "p"))
        if px is not None:
            candidates.append((3, px, _parse_ts_any(_get_field(trade, "timestamp", "t"))))

    quote = _get_field(snap, "latest_quote")
    if quote is not None:
        bid = _opt_pos_float(_get_field(quote, "bid_price", "bp"))
        ask = _opt_pos_float(_get_field(quote, "ask_price", "ap"))
        q_px: float | None = None
        if bid is not None and ask is not None:
            q_px = (bid + ask) / 2.0
        elif ask is not None or bid is not None:
            q_px = ask or bid
        if q_px is not None:
            candidates.append((2, q_px, _parse_ts_any(_get_field(quote, "timestamp", "t"))))

    mbar = _get_field(snap, "minute_bar")
    if mbar is not None:
        px = _opt_pos_float(_get_field(mbar, "close", "c", "price"))
        if px is not None:
            candidates.append((1, px, _parse_ts_any(_get_field(mbar, "timestamp", "t"))))

    dbar = _get_field(snap, "daily_bar")
    d_open: float | None = None
    d_high: float | None = None
    d_low: float | None = None
    d_vol: float | None = None
    d_vwap: float | None = None
    if dbar is not None:
        d_open = _opt_pos_float(_get_field(dbar, "open", "o"))
        d_high = _opt_pos_float(_get_field(dbar, "high", "h"))
        d_low = _opt_pos_float(_get_field(dbar, "low", "l"))
        d_vol = _opt_nonneg_float(_get_field(dbar, "volume", "v"))
        d_vwap = _opt_pos_float(_get_field(dbar, "vwap", "vw"))
        d_close = _opt_pos_float(_get_field(dbar, "close", "c", "price"))
        if d_close is not None and not candidates:
            candidates.append((0, d_close, _parse_ts_any(_get_field(dbar, "timestamp", "t"))))

    if not candidates:
        return None

    best_px: float
    best_ts: datetime | None
    with_ts = [(prio, px, ts) for prio, px, ts in candidates if ts is not None]
    if with_ts:
        _, best_px, best_ts = max(with_ts, key=lambda item: (item[2], item[0]))
    else:
        _, best_px, best_ts = max(candidates, key=lambda item: item[0])

    return _IntradaySnapshot(
        price=best_px,
        timestamp=best_ts,
        daily_open=d_open,
        daily_high=d_high,
        daily_low=d_low,
        daily_volume=d_vol,
        daily_vwap=d_vwap,
    )


class TraderDaemon(DaemonRunner):
    """Always-on trading loop: 30s broker sync/stops/kill-switches + 15:40 ET rebalance."""

    def __init__(
        self,
        settings: EnvSettings,
        data_dir: Path,
        config_dir: Path,
        *,
        db: StateDB | None = None,
        bar_store: ParquetBarStore | None = None,
        broker: AlpacaBroker | None = None,
        snapshot_client: Any = None,
        network_check_fn: Callable[[], bool] | None = None,
        env_file: Path | None = None,
        require_env_file: bool = False,
        discord_notifier: DiscordWebhookNotifier | None = None,
        poll_interval_open_seconds: float = 30.0,
        poll_interval_closed_seconds: float = 300.0,
        snapshot_interval_open_seconds: float = 300.0,
        snapshot_interval_closed_seconds: float = 3600.0,
        max_network_wait_seconds: float = 120.0,
        stop_event: threading.Event | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        validate_trading_mode(settings)
        self.settings = settings
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)
        resolved_db = db or StateDB(self.data_dir / "state.db")
        super().__init__(
            daemon_name="trader",
            db=resolved_db,
            poll_interval_seconds=poll_interval_open_seconds,
            stop_event=stop_event,
            sleep_fn=sleep_fn,
        )
        self.bar_store = bar_store or ParquetBarStore(self.data_dir / "parquet")
        self._injected_broker = broker
        self.broker: AlpacaBroker | None = broker
        self._snapshot_client = snapshot_client
        self._network_check_fn = network_check_fn
        self._explicit_env_file: Path | None = Path(env_file) if env_file is not None else None
        self._require_env_file: bool = bool(require_env_file) or (env_file is not None)
        self._blackout_tracker = EnvBlackoutTracker(
            env_file=env_file,
            require_env_file=require_env_file,
            fallback_raw=settings.blackout_period,
        )
        self._discord_notifier = discord_notifier
        self._last_blackout_db_sig: tuple[Any, ...] | None = None
        self._last_blackout_discord_sig: tuple[Any, ...] | None = None
        self._last_blackout_discord_attempt_sig: tuple[Any, ...] | None = None
        self._last_blackout_discord_attempt_at: datetime | None = None

        self.poll_interval_open_seconds = float(poll_interval_open_seconds)
        self.poll_interval_closed_seconds = float(poll_interval_closed_seconds)
        self.snapshot_interval_open_seconds = float(snapshot_interval_open_seconds)
        self.snapshot_interval_closed_seconds = float(snapshot_interval_closed_seconds)
        self.max_network_wait_seconds = float(max_network_wait_seconds)

        self._high_water_mark: float = 0.0
        self._last_snapshot_at: datetime | None = None
        self._ew50_base_prices: dict[str, float] = {}
        self._last_noncompleted_rebalance_at: datetime | None = None
        self._last_noncompleted_rebalance_reason: str | None = None

    def _get_broker(self) -> AlpacaBroker:
        if self.broker is None:
            self.broker = AlpacaBroker(
                self.settings,
                self.db,
                env_file=self._explicit_env_file,
                require_env_file=self._require_env_file,
                config_dir=self.config_dir,
            )
        return self.broker

    def _resolve_live_blackout_state(
        self,
        now_utc: datetime,
        observed_broker_symbols: Iterable[str] = (),
    ) -> BlackoutResolution:
        return self._blackout_tracker.resolve(
            db=self.db,
            config_dir=self.config_dir,
            now_utc=now_utc,
            observed_broker_symbols=observed_broker_symbols,
        )

    def _maybe_send_blackout_alert(
        self,
        blackout: BlackoutResolution,
        *,
        affected_positions: list[str],
        affected_orders: list[str],
        now_utc: datetime,
    ) -> bool:
        should_alert = (
            blackout.halted
            or blackout.has_warnings
            or (
                bool(blackout.frozen_symbols)
                and (bool(affected_positions) or bool(affected_orders))
            )
        )
        if not should_alert:
            self._last_blackout_db_sig = None
            self._last_blackout_discord_sig = None
            self._last_blackout_discord_attempt_sig = None
            self._last_blackout_discord_attempt_at = None
            return False

        sig = (
            blackout.halted,
            blackout.halt_reason,
            tuple(sorted(blackout.frozen_symbols)),
            tuple(sorted(blackout.warnings)),
            tuple(affected_positions),
            tuple(affected_orders),
        )
        blackout_dict = blackout.to_status_dict()
        if sig != self._last_blackout_db_sig:
            self.db.record_daemon_event(
                self.daemon_name,
                "blackout_alert",
                {
                    **blackout_dict,
                    "affected_positions": affected_positions,
                    "affected_orders": affected_orders,
                },
                timestamp=now_utc,
            )
            self._last_blackout_db_sig = sig

        webhook_url = self.settings.discord_webhook_url.strip()
        if self._discord_notifier is None and not webhook_url:
            return False
        if sig == self._last_blackout_discord_sig:
            return False
        if (
            sig == self._last_blackout_discord_attempt_sig
            and self._last_blackout_discord_attempt_at is not None
            and (now_utc - self._last_blackout_discord_attempt_at).total_seconds() < 300.0
        ):
            return False

        self._last_blackout_discord_attempt_sig = sig
        self._last_blackout_discord_attempt_at = now_utc

        embed = build_blackout_alert_discord_embed(
            blackout_dict,
            affected_positions=affected_positions,
            affected_orders=affected_orders,
        )
        try:
            if self._discord_notifier is not None:
                self._discord_notifier.send_embed(
                    title=embed["title"],
                    description=embed["description"],
                    color=embed["color"],
                    fields=embed.get("fields", ()),
                    footer=embed.get("footer", DEFAULT_DISCORD_FOOTER),
                )
            else:
                with DiscordWebhookNotifier(webhook_url) as notifier:
                    notifier.send_embed(
                        title=embed["title"],
                        description=embed["description"],
                        color=embed["color"],
                        fields=embed.get("fields", ()),
                        footer=embed.get("footer", DEFAULT_DISCORD_FOOTER),
                    )
        except Exception as exc:  # noqa: BLE001 - retry on next eligible cooldown tick
            logger.warning("blackout_discord_alert_failed", error=str(exc))
            return False

        self._last_blackout_discord_sig = sig
        return True

    def _get_snapshot_client(self, broker: AlpacaBroker) -> Any:
        if self._snapshot_client is not None:
            return self._snapshot_client
        return broker.quote_client

    def _default_network_check(self) -> bool:
        if self._injected_broker is not None:
            return True
        if not self.settings.alpaca_api_key.strip():
            return True
        host = "paper-api.alpaca.markets" if self.settings.alpaca_paper else "api.alpaca.markets"
        try:
            socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            return True
        except OSError:
            return False

    def _check_network(self) -> bool:
        fn = self._network_check_fn or self._default_network_check
        return bool(fn())

    def _seed_state_from_db(self) -> None:
        latest_snap = self.db.get_latest_portfolio_snapshot()
        if latest_snap is not None:
            self._high_water_mark = max(
                self._high_water_mark,
                float(latest_snap["high_water_mark"]),
                float(latest_snap["equity"]),
            )
            self._last_snapshot_at = ensure_utc(datetime.fromisoformat(latest_snap["timestamp"]))

    def get_sleep_seconds(self, now_utc: datetime | None = None) -> float:
        ref = ensure_utc(now_utc) if now_utc is not None else datetime.now(UTC)
        trading_client = self.broker.trading_client if self.broker is not None else None
        if check_market_open(ref, trading_client):
            return self.poll_interval_open_seconds
        return self.poll_interval_closed_seconds

    def startup(self, now_utc: datetime | None = None) -> None:
        validate_trading_mode(self.settings)
        now = ensure_utc(now_utc) if now_utc is not None else datetime.now(UTC)
        self.verify_wal_integrity(now_utc=now)
        net_ok = wait_for_network(
            self._check_network,
            max_wait_seconds=self.max_network_wait_seconds,
            stop_event=self.stop_event,
            sleep_fn=self._sleep_fn,
        )
        self._seed_state_from_db()
        broker = self._get_broker()
        now_et = now.astimezone(NY_TZ)
        session_date = now_et.date().isoformat()

        recon = broker.reconcile_with_broker(daemon_name=self.daemon_name, record_if_clean=True)
        stop_sync = recon.stop_sync_result
        if stop_sync is None:
            stop_sync = broker.sync_protective_stops(
                {},
                session_date=session_date,
                run_id=f"boot-{now.strftime('%Y%m%dT%H%M%SZ')}",
            )

        acct = broker.get_account_state()
        self._high_water_mark = max(self._high_water_mark, acct["equity"])

        boot_details = {
            "network_ok": net_ok,
            "paper_mode": self.settings.alpaca_paper,
            "broker_positions_count": recon.broker_positions_count,
            "position_diffs_count": len(recon.position_diffs),
            "missing_stop_symbols": recon.missing_stop_symbols,
            "submitted_stops": sorted(stop_sync.submitted_stops.keys()),
            "equity": round(acct["equity"], 2),
        }
        self.db.record_daemon_event(
            self.daemon_name,
            "boot",
            boot_details,
            timestamp=now,
        )
        self.write_heartbeat("healthy" if net_ok else "degraded", boot_details, now_utc=now)

    def _fetch_live_marks(
        self,
        symbols: list[str],
        broker: AlpacaBroker,
        broker_positions: Mapping[str, Mapping[str, float]],
        now_utc: datetime,
    ) -> tuple[dict[str, float], dict[str, datetime], dict[str, _IntradaySnapshot]]:
        marks: dict[str, float] = {}
        mark_ts: dict[str, datetime] = {}
        snapshots: dict[str, _IntradaySnapshot] = {}

        snap_client = self._get_snapshot_client(broker)
        if snap_client is not None and hasattr(snap_client, "get_stock_snapshot"):
            feed = DataFeed.SIP if self.settings.alpaca_data_feed == "sip" else DataFeed.IEX
            for idx in range(0, len(symbols), 25):
                batch = symbols[idx : idx + 25]
                try:
                    req = StockSnapshotRequest(symbol_or_symbols=batch, feed=feed)
                    resp = snap_client.get_stock_snapshot(req)
                    if isinstance(resp, Mapping):
                        for sym, snap in resp.items():
                            parsed = _extract_intraday_snapshot(snap)
                            if parsed is not None:
                                clean_sym = str(sym).strip().upper()
                                snapshots[clean_sym] = parsed
                                marks[clean_sym] = parsed.price
                                mark_ts[clean_sym] = parsed.timestamp or now_utc
                except Exception as exc:  # noqa: BLE001
                    logger.warning("trader_snapshot_fetch_failed", error=str(exc))

        # Fall back to quote_client or position marks for any missing symbols
        for sym in symbols:
            if sym in marks:
                continue
            if broker.quote_client is not None:
                bid, ask = broker.get_latest_quote(sym)
                if bid is not None and ask is not None:
                    marks[sym] = (bid + ask) / 2.0
                    mark_ts[sym] = now_utc
                    continue
                if ask is not None or bid is not None:
                    marks[sym] = float(ask or bid or 0.0)
                    mark_ts[sym] = now_utc
                    continue
            pos = broker_positions.get(sym)
            if pos is not None and pos["market_price"] > 0.0:
                marks[sym] = float(pos["market_price"])
                mark_ts[sym] = now_utc

        return marks, mark_ts, snapshots

    def _resolve_symbol_mark(
        self,
        symbol: str,
        live_marks: Mapping[str, float],
        *,
        prefer_consolidated_iex_fallback: bool = False,
    ) -> float | None:
        if (
            prefer_consolidated_iex_fallback
            and self._injected_broker is None
            and self.settings.alpaca_data_feed == "iex"
        ):
            yf_px = fetch_yfinance_latest_mark(symbol)
            if yf_px is not None:
                return yf_px
        live_px = live_marks.get(symbol)
        if live_px is not None:
            val = float(live_px)
            if math.isfinite(val) and val > 0.0:
                return val
        df = self.bar_store.load_bars(symbol, timeframe="1d")
        if not df.empty:
            val = float(df["close"].iloc[-1])
            if math.isfinite(val) and val > 0.0:
                return val
        return None

    def _compute_baselines(
        self,
        universe: list[UniverseSymbol],
        benchmark_symbol: str,
        live_marks: Mapping[str, float],
    ) -> tuple[float | None, float | None]:
        uni_syms = [u.symbol for u in universe]

        # Seed deterministic base closes once from Parquet (<= 50 floats) using O(1) first snapshot
        if not self._ew50_base_prices and uni_syms:
            first_snap = self.db.get_first_portfolio_snapshot()
            anchor_date = (
                ensure_utc(datetime.fromisoformat(first_snap["timestamp"])).date()
                if first_snap is not None
                else None
            )
            for sym in uni_syms:
                df = self.bar_store.load_bars(sym, timeframe="1d")
                if df.empty:
                    continue
                if anchor_date is not None:
                    prior = df[df["timestamp"].dt.date <= anchor_date]
                    base_px = (
                        float(prior["close"].iloc[-1])
                        if not prior.empty
                        else float(df["close"].iloc[0])
                    )
                else:
                    base_px = float(df["close"].iloc[-1])
                if math.isfinite(base_px) and base_px > 0.0:
                    self._ew50_base_prices[sym] = base_px

        spy_close = self._resolve_symbol_mark(benchmark_symbol, live_marks)

        ratios: list[float] = []
        for sym in uni_syms:
            anchor_px = self._ew50_base_prices.get(sym)
            if anchor_px is None or anchor_px <= 0.0:
                continue
            cur_px = live_marks.get(sym)
            if cur_px is None:
                df = self.bar_store.load_bars(sym, timeframe="1d")
                if not df.empty:
                    cur_px = float(df["close"].iloc[-1])
            if cur_px is not None and math.isfinite(cur_px) and cur_px > 0.0:
                ratios.append(cur_px / anchor_px)

        if ratios:
            ew50_index = round(100.0 * (sum(ratios) / len(ratios)), 6)
        elif live_marks:
            ew50_index = 100.0
        else:
            ew50_index = None

        return spy_close, ew50_index

    def _build_bars_with_intraday_proxy(
        self,
        *,
        symbols: list[str],
        universe_symbols: set[str],
        benchmark_symbol: str,
        max_staleness_minutes: float,
        live_marks: Mapping[str, float],
        mark_ts: Mapping[str, datetime],
        snapshots: Mapping[str, _IntradaySnapshot],
        now_utc: datetime,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, float], float | None]:
        bars_by_symbol: dict[str, pd.DataFrame] = {}
        all_prices: dict[str, float] = {}
        now_et_date = now_utc.astimezone(NY_TZ).date()
        fresh_ages: list[float] = []

        for sym in symbols:
            df = self.bar_store.load_bars(sym, timeframe="1d")
            if len(df) > _TRAILING_BAR_ROWS:
                df = df.tail(_TRAILING_BAR_ROWS).reset_index(drop=True)

            live_px = live_marks.get(sym)
            snap = snapshots.get(sym)
            age_min: float | None = None

            if live_px is not None and math.isfinite(live_px) and live_px > 0.0:
                all_prices[sym] = float(live_px)
                ts = mark_ts.get(sym, now_utc)
                age_min = max(0.0, (now_utc - ts).total_seconds() / 60.0)
            elif not df.empty:
                last_close = float(df["close"].iloc[-1])
                if math.isfinite(last_close) and last_close > 0.0:
                    all_prices[sym] = last_close
                last_ts = ensure_utc(df["timestamp"].iloc[-1].to_pydatetime())
                age_min = max(0.0, (now_utc - last_ts).total_seconds() / 60.0)

            is_fresh = age_min is not None and age_min <= max_staleness_minutes
            if not is_fresh:
                continue

            assert age_min is not None
            fresh_ages.append(age_min)

            if live_px is not None and math.isfinite(live_px) and live_px > 0.0:
                d_open = snap.daily_open if (snap is not None and snap.daily_open is not None) else live_px
                d_high = max(
                    snap.daily_high if (snap is not None and snap.daily_high is not None) else live_px,
                    live_px,
                )
                d_low = min(
                    snap.daily_low if (snap is not None and snap.daily_low is not None) else live_px,
                    live_px,
                )
                d_vol = snap.daily_volume if (snap is not None and snap.daily_volume is not None) else 0.0
                d_vwap = (
                    snap.daily_vwap
                    if (snap is not None and snap.daily_vwap is not None)
                    else float("nan")
                )

                if not df.empty:
                    last_bar_et_date = (
                        ensure_utc(df["timestamp"].iloc[-1].to_pydatetime())
                        .astimezone(NY_TZ)
                        .date()
                    )
                    if last_bar_et_date == now_et_date:
                        df = df.copy()
                        idx = df.index[-1]
                        if snap is not None and snap.daily_open is not None:
                            df.at[idx, "open"] = d_open
                        df.at[idx, "close"] = live_px
                        df.at[idx, "high"] = max(float(df.at[idx, "high"]), d_high)
                        df.at[idx, "low"] = min(float(df.at[idx, "low"]), d_low)
                        if snap is not None and snap.daily_volume is not None:
                            df.at[idx, "volume"] = d_vol
                        if snap is not None and snap.daily_vwap is not None:
                            df.at[idx, "vwap"] = d_vwap
                        df.at[idx, "timestamp"] = pd.Timestamp(now_utc)
                    else:
                        proxy_row = pd.DataFrame(
                            [
                                {
                                    "timestamp": pd.Timestamp(now_utc),
                                    "open": d_open,
                                    "high": d_high,
                                    "low": d_low,
                                    "close": live_px,
                                    "volume": d_vol,
                                    "vwap": d_vwap,
                                }
                            ]
                        )
                        df = pd.concat([df, proxy_row], ignore_index=True)
                else:
                    df = pd.DataFrame(
                        [
                            {
                                "timestamp": pd.Timestamp(now_utc),
                                "open": d_open,
                                "high": d_high,
                                "low": d_low,
                                "close": live_px,
                                "volume": d_vol,
                                "vwap": d_vwap,
                            }
                        ]
                    )

            bars_by_symbol[sym] = df

        fresh_universe_count = sum(1 for s in universe_symbols if s in bars_by_symbol)
        benchmark_fresh = benchmark_symbol in bars_by_symbol
        min_coverage = 0.80 * len(universe_symbols)

        if not benchmark_fresh or fresh_universe_count < min_coverage or not fresh_ages:
            latest_age: float | None = float(max_staleness_minutes) + 1.0
        else:
            latest_age = max(fresh_ages)

        return bars_by_symbol, all_prices, latest_age

    def _has_completed_rebalance_for_month(self, target_date: date) -> bool:
        return self.db.has_month_rebalance_completed(
            target_date.year,
            target_date.month,
            daemon_name=self.daemon_name,
            exclude_session_date=target_date.isoformat(),
        )

    def _active_rebalance_checkpoint(
        self,
        *,
        now_utc: datetime,
        now_et: datetime,
        next_close_utc: datetime | None,
        app_cfg: AppConfig,
        has_open_positions: bool | None = None,
    ) -> str | None:
        """Return matching configured/half-day checkpoint string if within window, else None."""
        today = now_et.date()
        if not is_nyse_trading_day(today):
            return None

        schedule = app_cfg.strategy.rebalance_schedule
        if schedule == "monthly_first":
            probe = today - timedelta(days=1)
            is_first_trading_day = True
            while probe.month == today.month:
                if is_nyse_trading_day(probe):
                    is_first_trading_day = False
                    break
                probe -= timedelta(days=1)
            if not is_first_trading_day and (
                has_open_positions is None or self._has_completed_rebalance_for_month(today)
            ):
                return None
        elif schedule == "monthly_last":
            probe = today + timedelta(days=1)
            is_last_trading_day = True
            while probe.month == today.month:
                if is_nyse_trading_day(probe):
                    is_last_trading_day = False
                    break
                probe += timedelta(days=1)
            if not is_last_trading_day and (
                has_open_positions is None
                or has_open_positions
                or self._has_completed_rebalance_for_month(today)
            ):
                return None
        elif app_cfg.strategy.rebalance_every_n_days > 1:
            recent = self.db.get_daemon_events(
                daemon_name=self.daemon_name, event_type="rebalance", limit=1
            )
            if recent:
                last_session = str(recent[0]["details"].get("session_date", "")).strip()
                if last_session:
                    try:
                        last_dt = datetime.fromisoformat(f"{last_session}T00:00:00").date()
                        if (
                            last_dt != now_et.date()
                            and nyse_trading_days_apart(last_dt, now_et.date())
                            < app_cfg.strategy.rebalance_every_n_days
                        ):
                            return None
                    except ValueError:
                        pass

        # Early-close half-day support via Alpaca /v2/clock next_close
        if next_close_utc is not None:
            next_close_et = next_close_utc.astimezone(NY_TZ)
            if next_close_et.date() == now_et.date() and next_close_et.time() < dt_time(16, 0):
                win_start = next_close_et - timedelta(minutes=20)
                win_end = next_close_et - timedelta(minutes=2)
                if win_start <= now_et < win_end:
                    return win_start.strftime("%H:%M")
                return None

        cur_t = now_et.time()
        for ckpt in app_cfg.strategy.rebalance_checkpoints_et:
            hh, mm = (int(part) for part in ckpt.split(":"))
            ckpt_start = datetime.combine(now_et.date(), dt_time(hh, mm), tzinfo=NY_TZ)
            ckpt_end = min(
                ckpt_start + timedelta(minutes=18),
                datetime.combine(now_et.date(), dt_time(15, 58), tzinfo=NY_TZ),
            )
            if ckpt_start.time() <= cur_t < ckpt_end.time():
                return ckpt
        return None

    def _run_rebalance_checkpoint(
        self,
        *,
        now_utc: datetime,
        session_date: str,
        checkpoint: str,
        app_cfg: AppConfig,
        universe: list[UniverseSymbol],
        broker: AlpacaBroker,
        broker_positions: Mapping[str, Mapping[str, float]],
        equity: float,
        daily_pnl_pct: float,
        drawdown_pct: float,
        dials: ControlDials,
        live_marks: Mapping[str, float],
        mark_ts: Mapping[str, datetime],
        snapshots: Mapping[str, _IntradaySnapshot],
        blackout: BlackoutResolution | None = None,
    ) -> dict[str, Any]:
        symbols = list(
            dict.fromkeys([u.symbol for u in universe] + [app_cfg.backtest.benchmark_symbol])
        )
        universe_sym_set = {u.symbol for u in universe}
        bars_with_proxy, prices, latest_bar_age_minutes = self._build_bars_with_intraday_proxy(
            symbols=symbols,
            universe_symbols=universe_sym_set,
            benchmark_symbol=app_cfg.backtest.benchmark_symbol,
            max_staleness_minutes=float(app_cfg.risk.max_data_staleness_minutes),
            live_marks=live_marks,
            mark_ts=mark_ts,
            snapshots=snapshots,
            now_utc=now_utc,
        )

        scorer_model = active_scorer_model(self.settings)
        scored_news = self.db.get_all_scored_news(
            start_utc=now_utc - timedelta(days=35),
            scorer_model=scorer_model,
            sources=app_cfg.strategy.model_news_sources,
            include_forward_only=False,
        )
        news_by_symbol: dict[str, list[NewsArticle]] = defaultdict(list)
        for art in scored_news:
            for sym in art.symbols:
                news_by_symbol[sym].append(art)

        macro_points: list[MacroPoint] = []
        for sid in FRED_SERIES_IDS:
            macro_points.extend(self.db.get_macro_series(sid))

        earnings_by_symbol = self.db.get_all_earnings_map()
        sec_filings_by_symbol = self.db.get_sec_filings_map()

        universe_meta: dict[str, AssetRiskMetadata] = {
            u.symbol: AssetRiskMetadata(
                symbol=u.symbol,
                asset_type=u.asset_type,
                sector=u.sector,
                shortable=u.shortable_default,
                easy_to_borrow=u.shortable_default,
            )
            for u in universe
        }
        # HIGH #6: Propagate sector/asset metadata across share-class siblings (e.g. GOOGL -> GOOG)
        for grp in ISSUER_SHARE_CLASS_GROUPS:
            template_meta = next((universe_meta[s] for s in grp if s in universe_meta), None)
            if template_meta is not None:
                for alias in grp:
                    if alias not in universe_meta:
                        universe_meta[alias] = template_meta.model_copy(update={"symbol": alias})

        # HIGH #6: Ensure every held broker position (including out-of-universe GOOG/ACME/BRK.B/VOX)
        # has a valid price so frozen positions count toward gross/net/sector headroom and do not
        # fail Pass-2 entries with MISSING_OR_INVALID_PORTFOLIO_PRICE.
        for sym, pos in broker_positions.items():
            cur_px = prices.get(sym)
            if cur_px is None or not math.isfinite(cur_px) or cur_px <= 0.0:
                mark_px = live_marks.get(sym)
                if mark_px is not None and math.isfinite(mark_px) and mark_px > 0.0:
                    prices[sym] = float(mark_px)
                else:
                    mkt_px = float(pos.get("market_price", 0.0))
                    entry_px = float(pos.get("avg_entry_price", 0.0))
                    fallback_px = mkt_px if (math.isfinite(mkt_px) and mkt_px > 0.0) else entry_px
                    if math.isfinite(fallback_px) and fallback_px > 0.0:
                        prices[sym] = float(fallback_px)

        signal_universe_syms: set[str] = set(universe_sym_set)
        if app_cfg.strategy.exclude_etfs_from_equity_slots:
            signal_universe_syms = {
                u.symbol for u in universe if u.asset_type != AssetType.etf
            }
            if app_cfg.strategy.safe_haven_rotation_fraction > 0.0:
                signal_universe_syms |= {
                    u.symbol for u in universe if u.symbol in SAFE_HAVEN_SYMBOLS
                }
        elif (
            app_cfg.strategy.exclude_safe_havens_from_equity_slots
            and app_cfg.strategy.safe_haven_rotation_fraction <= 0.0
        ):
            signal_universe_syms = {
                u.symbol
                for u in universe
                if not (u.sector in SAFE_HAVEN_SECTORS or u.symbol in SAFE_HAVEN_SYMBOLS)
            }

        feature_engine = FeatureEngine(app_cfg.strategy)
        signals = feature_engine.build_signals(
            as_of=now_utc,
            bars_by_symbol=bars_with_proxy,
            news_by_symbol=news_by_symbol,
            macro_points=macro_points,
            benchmark_symbol=app_cfg.backtest.benchmark_symbol,
            fallback_rf_apr=app_cfg.backtest.risk_free_rate_apr,
            sec_filings_by_symbol=sec_filings_by_symbol,
            restatement_blackout_days=app_cfg.risk.restatement_blackout_days,
            universe_symbols=signal_universe_syms,
        )

        current_positions = {sym: float(pos["qty"]) for sym, pos in broker_positions.items()}
        active_blackout = blackout or self._resolve_live_blackout_state(
            now_utc,
            observed_broker_symbols=broker_positions.keys(),
        )

        portfolio_ctor = PortfolioConstructor(app_cfg.strategy, app_cfg.risk)
        proposed = portfolio_ctor.build_proposed_orders(
            signals=signals,
            current_positions=current_positions,
            prices=prices,
            equity=equity,
            dials=dials,
            as_of=now_utc,
            universe_meta=universe_meta,
            frozen_symbols=active_blackout.frozen_symbols,
        )

        risk_state = PortfolioRiskState(
            as_of=now_utc,
            equity=equity,
            daily_pnl_pct=daily_pnl_pct,
            drawdown_pct=drawdown_pct,
            positions=current_positions,
            prices=prices,
            latest_bar_age_minutes=latest_bar_age_minutes,
        )
        signals_by_symbol = {s.symbol: s for s in signals}
        risk_engine = HardRiskEngine(
            app_cfg.risk,
            allow_fractional_longs=app_cfg.strategy.allow_fractional_longs,
        )
        verdicts = risk_engine.evaluate_batch(
            orders=proposed,
            state=risk_state,
            dials=dials,
            universe_meta=universe_meta,
            signals_by_symbol=signals_by_symbol,
            earnings_by_symbol=earnings_by_symbol,
            frozen_symbols=active_blackout.frozen_symbols,
            blackout_halted=active_blackout.halted,
        )

        ckpt_compact = checkpoint.replace(":", "")
        run_id = f"rebal-{session_date}-{ckpt_compact}-{now_utc.strftime('%H%M%S')}"
        exec_report = broker.execute_verdicts(
            verdicts,
            session_date=session_date,
            checkpoint=ckpt_compact,
            run_id=run_id,
        )

        existing_db_stops = {
            p["symbol"]: p["stop_price"]
            for p in self.db.get_live_positions()
            if p.get("stop_price") is not None
        }
        desired_stops: dict[str, float] = {}
        for v in verdicts:
            if v.approved and v.adjusted_qty > 0.0 and v.stop_loss_price is not None:
                sym = v.order.symbol
                cand_stop = float(v.stop_loss_price)
                cur_q = current_positions.get(sym, 0.0)
                prev_stop = existing_db_stops.get(sym)
                ref_px = prices.get(sym, v.order.reference_price)
                if cur_q > 1e-9 and v.order.side == OrderSide.buy and prev_stop is not None:
                    tightened = max(cand_stop, float(prev_stop))
                    desired_stops[sym] = (
                        tightened if (ref_px > 0.0 and tightened < ref_px) else cand_stop
                    )
                elif cur_q < -1e-9 and v.order.side == OrderSide.sell and prev_stop is not None:
                    tightened = min(cand_stop, float(prev_stop))
                    desired_stops[sym] = (
                        tightened if (ref_px > 0.0 and tightened > ref_px) else cand_stop
                    )
                else:
                    desired_stops[sym] = cand_stop

        for sym, pos in broker.get_broker_positions().items():
            if active_blackout.halted or sym in active_blackout.frozen_symbols:
                continue
            if sym in desired_stops:
                continue
            if sym in existing_db_stops and existing_db_stops[sym] is not None:
                desired_stops[sym] = float(existing_db_stops[sym])
            elif sym in signals_by_symbol:
                sig = signals_by_symbol[sym]
                ref = prices.get(sym, pos["market_price"] or pos["avg_entry_price"])
                if ref > 0.0:
                    if pos["qty"] > 0.0:
                        desired_stops[sym] = round(
                            ref * max(0.05, 1.0 - app_cfg.risk.stop_loss_atr_mult_long * sig.atr_pct),
                            4,
                        )
                    else:
                        desired_stops[sym] = round(
                            ref * (1.0 + app_cfg.risk.stop_loss_atr_mult_short * sig.atr_pct),
                            4,
                        )

        stop_sync = broker.sync_protective_stops(
            desired_stops,
            session_date=session_date,
            run_id=run_id,
        )

        approved_count = sum(1 for v in verdicts if v.approved and v.adjusted_qty > 0.0)
        successful_exec_count = sum(
            1
            for r in exec_report.records
            if r.filled_qty > 0.0
            or (
                r.error is None
                and r.status not in {"rejected", "canceled", "expired", "blocked_blackout"}
            )
        )
        if active_blackout.halted:
            rebalance_status = "halted_blackout"
        elif dials.pause_new_entries:
            rebalance_status = "paused"
        elif len(signals) == 0:
            rebalance_status = "skipped_no_signals"
        elif len(proposed) > 0 and approved_count == 0:
            rebalance_status = "rejected_by_risk"
        elif approved_count > 0 and successful_exec_count == 0:
            rebalance_status = "failed_execution"
        else:
            rebalance_status = "completed"

        if rebalance_status == "completed":
            self._last_noncompleted_rebalance_at = None
            self._last_noncompleted_rebalance_reason = None
        else:
            self._last_noncompleted_rebalance_at = now_utc
            self._last_noncompleted_rebalance_reason = rebalance_status

        rebalance_details = {
            "session_date": session_date,
            "checkpoint": checkpoint,
            "status": rebalance_status,
            "paused": dials.pause_new_entries,
            "run_id": run_id,
            "signals_count": len(signals),
            "fresh_symbols_count": len(bars_with_proxy),
            "latest_bar_age_minutes": latest_bar_age_minutes,
            "proposed_count": len(proposed),
            "approved_count": approved_count,
            "successful_exec_count": successful_exec_count,
            "executed_count": len(exec_report.records),
            "submitted_stops": sorted(stop_sync.submitted_stops.keys()),
            "untouched_blackout_positions": stop_sync.untouched_blackout_positions,
            "untouched_blackout_orders": stop_sync.untouched_blackout_orders,
            "blackout": active_blackout.to_status_dict(),
            "errors": exec_report.errors + stop_sync.errors,
        }
        self.db.record_daemon_event(
            self.daemon_name,
            "rebalance",
            rebalance_details,
            timestamp=now_utc,
        )
        del bars_with_proxy, scored_news
        release_memory()
        return rebalance_details

    def run_once(self, now_utc: datetime | None = None) -> dict[str, Any]:
        validate_trading_mode(self.settings)
        now = ensure_utc(now_utc) if now_utc is not None else datetime.now(UTC)
        now_et = now.astimezone(NY_TZ)
        session_date = now_et.date().isoformat()
        broker = self._get_broker()
        market_open, next_close_utc = get_market_clock_info(now, broker.trading_client)

        app_cfg = load_app_config(self.config_dir)
        universe = load_universe(self.config_dir)
        all_symbols = list(
            dict.fromkeys([u.symbol for u in universe] + [app_cfg.backtest.benchmark_symbol])
        )

        # 1. Fast cycle broker reconciliation + stop verification + live BLACKOUT_PERIOD check
        recon = broker.reconcile_with_broker(daemon_name=self.daemon_name, record_if_clean=False)
        acct = broker.get_account_state()
        positions = broker.get_broker_positions()
        open_orders = broker._fetch_open_orders()
        observed_broker_symbols = set(positions.keys()) | {
            str(getattr(o, "symbol", "")).strip().upper()
            for o in open_orders
            if getattr(o, "symbol", None)
        }
        blackout = self._resolve_live_blackout_state(
            now,
            observed_broker_symbols=observed_broker_symbols,
        )
        affected_positions = (
            sorted(positions.keys())
            if blackout.halted
            else sorted(sym for sym in positions if sym in blackout.frozen_symbols)
        )
        self._maybe_send_blackout_alert(
            blackout,
            affected_positions=affected_positions,
            affected_orders=recon.untouched_blackout_orders,
            now_utc=now,
        )

        equity = acct["equity"]
        cash = acct["cash"]
        buying_power = acct["buying_power"]
        last_equity = acct["last_equity"]
        daily_pnl_pct = (equity - last_equity) / last_equity if last_equity > 1e-9 else 0.0

        self._high_water_mark = max(self._high_water_mark, equity)
        drawdown_pct = (
            min(0.0, (equity - self._high_water_mark) / self._high_water_mark)
            if self._high_water_mark > 1e-9
            else 0.0
        )

        # 2. Kill-switch auto-pause checks
        cp = ControlPlane(self.db)
        dials = cp.get_dials()
        kill_switch_triggered: str | None = None
        if daily_pnl_pct <= -app_cfg.risk.daily_loss_halt_pct:
            kill_switch_triggered = (
                f"Daily loss kill-switch breached ({daily_pnl_pct:.2%} <= "
                f"-{app_cfg.risk.daily_loss_halt_pct:.2%})"
            )
        elif drawdown_pct <= -app_cfg.risk.max_drawdown_halt_pct:
            kill_switch_triggered = (
                f"Max drawdown kill-switch breached ({drawdown_pct:.2%} <= "
                f"-{app_cfg.risk.max_drawdown_halt_pct:.2%})"
            )

        if kill_switch_triggered is not None and not dials.pause_new_entries:
            dials = cp.set_pause(
                True,
                actor="agy-trader",
                reason=kill_switch_triggered,
                now=now,
            )
            self.db.record_daemon_event(
                self.daemon_name,
                "error",
                {"kind": "kill_switch", "reason": kill_switch_triggered},
                timestamp=now,
            )

        # 3. Scheduled idempotent rebalance checkpoint (strictly gated on market_open == True!)
        rebalance_result: dict[str, Any] | None = None
        live_marks: dict[str, float] = {}
        mark_ts: dict[str, datetime] = {}
        snapshots: dict[str, _IntradaySnapshot] = {}

        active_ckpt = (
            self._active_rebalance_checkpoint(
                now_utc=now,
                now_et=now_et,
                next_close_utc=next_close_utc,
                app_cfg=app_cfg,
                has_open_positions=bool(positions),
            )
            if market_open
            else None
        )
        should_rebalance = (
            market_open
            and active_ckpt is not None
            and not self.db.has_session_rebalance_completed(session_date, active_ckpt)
        )
        if should_rebalance and self._last_noncompleted_rebalance_at is not None:
            blocker_cleared = (
                self._last_noncompleted_rebalance_reason == "paused" and not dials.pause_new_entries
            ) or (
                self._last_noncompleted_rebalance_reason == "halted_blackout"
                and not blackout.halted
            )
            if (
                not blocker_cleared
                and (now - self._last_noncompleted_rebalance_at).total_seconds() < 300.0
            ):
                should_rebalance = False

        snap_cadence = (
            self.snapshot_interval_open_seconds
            if market_open
            else self.snapshot_interval_closed_seconds
        )
        due_for_snapshot = (
            self._last_snapshot_at is None
            or (now - self._last_snapshot_at).total_seconds() >= snap_cadence
        )

        if should_rebalance or due_for_snapshot:
            comp_benchmarks = [
                s.strip().upper()
                for s in app_cfg.backtest.comparison_benchmarks
                if s.strip()
            ]
            mark_symbols = list(
                dict.fromkeys(all_symbols + list(positions.keys()) + comp_benchmarks)
            )
            live_marks, mark_ts, snapshots = self._fetch_live_marks(
                mark_symbols, broker, positions, now
            )

        if should_rebalance:
            assert active_ckpt is not None
            rebalance_result = self._run_rebalance_checkpoint(
                now_utc=now,
                session_date=session_date,
                checkpoint=active_ckpt,
                app_cfg=app_cfg,
                universe=universe,
                broker=broker,
                broker_positions=positions,
                equity=equity,
                daily_pnl_pct=daily_pnl_pct,
                drawdown_pct=drawdown_pct,
                dials=dials,
                live_marks=live_marks,
                mark_ts=mark_ts,
                snapshots=snapshots,
                blackout=blackout,
            )
            acct = broker.get_account_state()
            positions = broker.get_broker_positions()
            equity = acct["equity"]
            cash = acct["cash"]
            buying_power = acct["buying_power"]
            last_equity = acct["last_equity"]
            daily_pnl_pct = (equity - last_equity) / last_equity if last_equity > 1e-9 else 0.0
            self._high_water_mark = max(self._high_water_mark, equity)
            drawdown_pct = (
                min(0.0, (equity - self._high_water_mark) / self._high_water_mark)
                if self._high_water_mark > 1e-9
                else 0.0
            )
            due_for_snapshot = True

        # 4. Bounded portfolio snapshot recording
        snapshot_recorded = False
        if due_for_snapshot:
            gross_notional = 0.0
            net_notional = 0.0
            for pos in positions.values():
                px = pos["market_price"] if pos["market_price"] > 0.0 else pos["avg_entry_price"]
                notional = pos["qty"] * px
                gross_notional += abs(notional)
                net_notional += notional
            gross_exposure = gross_notional / equity if equity > 1e-9 else 0.0
            net_exposure = net_notional / equity if equity > 1e-9 else 0.0
            spy_close, ew50_index = self._compute_baselines(
                universe, app_cfg.backtest.benchmark_symbol, live_marks
            )
            spmo_close = self._resolve_symbol_mark(
                "SPMO", live_marks, prefer_consolidated_iex_fallback=True
            )
            qmom_close = self._resolve_symbol_mark(
                "QMOM", live_marks, prefer_consolidated_iex_fallback=True
            )
            self.db.insert_portfolio_snapshot(
                timestamp=now,
                equity=round(equity, 2),
                cash=round(cash, 2),
                buying_power=round(buying_power, 2),
                gross_exposure=round(gross_exposure, 4),
                net_exposure=round(net_exposure, 4),
                daily_pnl_pct=round(daily_pnl_pct, 6),
                drawdown_pct=round(drawdown_pct, 6),
                high_water_mark=round(self._high_water_mark, 2),
                spy_close=spy_close,
                ew50_index=ew50_index,
                spmo_close=spmo_close,
                qmom_close=qmom_close,
            )
            self._last_snapshot_at = now
            snapshot_recorded = True

        hb_status = "healthy" if market_open else "idle_market_closed"
        details = {
            "market_open": market_open,
            "open_positions": len(positions),
            "equity": round(equity, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 6),
            "drawdown_pct": round(drawdown_pct, 6),
            "paused": dials.pause_new_entries,
            "blackout": blackout.to_status_dict(),
            "rebalance_ran": rebalance_result is not None,
            "snapshot_recorded": snapshot_recorded,
        }
        self.write_heartbeat(hb_status, details, now_utc=now)
        return {"status": hb_status, "rebalance": rebalance_result, **details}

    def shutdown(self) -> None:
        if self.broker is not None:
            self.broker.close()
