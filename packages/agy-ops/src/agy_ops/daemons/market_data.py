from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Any

import httpx
import structlog
import yaml
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest

from agy_trader.core.blackout import NON_EQUITY_ETF_SECTORS
from agy_trader.core.calendar import is_nyse_trading_day
from agy_trader.core.config import EnvSettings, load_app_config, load_universe
from agy_trader.core.models import AssetType, ensure_utc
from agy_trader.data.alpaca_market import AlpacaMarketDataProvider
from agy_trader.data.macro_and_calendar import (
    FREDMacroProvider,
    YFinanceEarningsProvider,
    YFinanceEtfHoldingsProvider,
    fetch_yfinance_consolidated_bars,
)
from agy_trader.data.providers import (
    EarningsProvider,
    EtfHoldingsProvider,
    MacroProvider,
    MarketDataProvider,
    SecFilingsProvider,
)
from agy_trader.data.sec_edgar import SecEdgarProvider
from agy_trader.storage.db import StateDB
from agy_trader.storage.parquet_store import CorporateActionMismatchError, ParquetBarStore

from .common import (
    NY_TZ,
    DaemonRunner,
    check_market_open,
    close_sdk_client,
    release_memory,
    wait_for_network,
)

logger = structlog.get_logger(__name__)


def _extract_snapshot_price(snap: Any) -> float | None:
    if isinstance(snap, dict):
        for key in ("latest_trade", "minute_bar", "daily_bar"):
            sub = snap.get(key)
            if isinstance(sub, dict):
                px = sub.get("price") or sub.get("close") or sub.get("c") or sub.get("p")
                if px is not None and float(px) > 0.0:
                    return float(px)
        return None
    for attr in ("latest_trade", "minute_bar", "daily_bar"):
        sub = getattr(snap, attr, None)
        if sub is None:
            continue
        for field in ("price", "close"):
            val = getattr(sub, field, None)
            if val is not None and float(val) > 0.0:
                return float(val)
    return None


class MarketDataDaemon(DaemonRunner):
    """Always-on market data, macro, earnings, and SEC EDGAR ingestion daemon."""

    def __init__(
        self,
        settings: EnvSettings,
        data_dir: Path,
        config_dir: Path,
        *,
        db: StateDB | None = None,
        bar_store: ParquetBarStore | None = None,
        market_provider: MarketDataProvider | None = None,
        macro_provider: MacroProvider | None = None,
        earnings_provider: EarningsProvider | None = None,
        sec_provider: SecFilingsProvider | None = None,
        etf_holdings_provider: EtfHoldingsProvider | None = None,
        snapshot_client: Any = None,
        trading_client: Any = None,
        network_check_fn: Callable[[], bool] | None = None,
        poll_interval_open_seconds: float = 60.0,
        poll_interval_closed_seconds: float = 300.0,
        eod_retry_interval_seconds: float = 300.0,
        min_bars_required: int = 300,
        max_network_wait_seconds: float = 120.0,
        stop_event: threading.Event | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)
        resolved_db = db or StateDB(self.data_dir / "state.db")
        super().__init__(
            daemon_name="market-data",
            db=resolved_db,
            poll_interval_seconds=poll_interval_open_seconds,
            stop_event=stop_event,
            sleep_fn=sleep_fn,
        )
        self.bar_store = bar_store or ParquetBarStore(self.data_dir / "parquet")
        self.poll_interval_open_seconds = float(poll_interval_open_seconds)
        self.poll_interval_closed_seconds = float(poll_interval_closed_seconds)
        self.eod_retry_interval_seconds = float(eod_retry_interval_seconds)
        self.min_bars_required = int(min_bars_required)
        self.max_network_wait_seconds = float(max_network_wait_seconds)

        self._injected_market_provider = market_provider is not None
        self._market_provider = market_provider
        self._macro_provider = macro_provider
        self._earnings_provider = earnings_provider
        self._sec_provider = sec_provider
        self._etf_holdings_provider = etf_holdings_provider
        self._snapshot_client = snapshot_client
        self._trading_client = trading_client
        self._owns_trading_client = False
        self._network_check_fn = network_check_fn

        self._owned_data_client: StockHistoricalDataClient | None = None
        self._last_eod_session_date: str | None = None
        self._last_eod_attempt_at: datetime | None = None
        self._last_prune_date: str | None = None
        self._latest_snapshot_at: str | None = None
        self._latest_snapshot_count: int = 0
        self._boot_network_ok: bool = False
        self._boot_backfill_done: bool = False

    def _universe_symbols(self) -> list[str]:
        app_cfg = load_app_config(self.config_dir)
        universe = load_universe(self.config_dir)
        return list(
            dict.fromkeys([u.symbol for u in universe] + [app_cfg.backtest.benchmark_symbol])
        )

    def _comparison_benchmark_symbols(self) -> list[str]:
        app_cfg = load_app_config(self.config_dir)
        universe_set = set(self._universe_symbols())
        out: list[str] = []
        for raw in app_cfg.backtest.comparison_benchmarks:
            sym = raw.strip().upper()
            if sym and sym not in universe_set and sym not in out:
                out.append(sym)
        return out

    def _equity_symbols(self) -> list[str]:
        universe = load_universe(self.config_dir)
        eq_syms = [u.symbol for u in universe if u.asset_type == AssetType.equity]
        return eq_syms if eq_syms else [u.symbol for u in universe]

    def _get_data_client(self) -> StockHistoricalDataClient | None:
        if self._snapshot_client is not None:
            return self._snapshot_client
        if self._owned_data_client is not None:
            return self._owned_data_client
        if self.settings.alpaca_api_key.strip() and self.settings.alpaca_secret_key.strip():
            self._owned_data_client = StockHistoricalDataClient(
                api_key=self.settings.alpaca_api_key,
                secret_key=self.settings.alpaca_secret_key,
            )
            return self._owned_data_client
        return None

    def _get_market_provider(self) -> MarketDataProvider | None:
        if self._market_provider is not None:
            return self._market_provider
        if self.settings.alpaca_api_key.strip() and self.settings.alpaca_secret_key.strip():
            client = self._get_data_client()
            self._market_provider = AlpacaMarketDataProvider(
                api_key=self.settings.alpaca_api_key,
                secret_key=self.settings.alpaca_secret_key,
                feed=self.settings.alpaca_data_feed,
                client=client,
            )
            return self._market_provider
        return None

    def _default_network_check(self) -> bool:
        if self._market_provider is not None or self._snapshot_client is not None:
            return True
        if not self.settings.alpaca_api_key.strip():
            return True
        try:
            socket.getaddrinfo("data.alpaca.markets", 443, type=socket.SOCK_STREAM)
            return True
        except OSError:
            return False

    def _check_network(self) -> bool:
        fn = self._network_check_fn or self._default_network_check
        return bool(fn())

    def _canonical_daily_end(self, now_utc: datetime) -> datetime:
        """Avoid persisting incomplete today's 1d bar before 16:05 ET."""
        now_et = now_utc.astimezone(NY_TZ)
        if is_nyse_trading_day(now_et.date()) and now_et.time() < dt_time(16, 5):
            prev_day_end_et = datetime.combine(
                now_et.date() - timedelta(days=1), dt_time(23, 59, 59), tzinfo=NY_TZ
            )
            return ensure_utc(prev_day_end_et)
        return now_utc

    def _save_daily_bars_safe(
        self,
        bars_map: dict[str, Any],
        *,
        now_utc: datetime,
    ) -> tuple[int, list[str]]:
        saved_rows = 0
        alerts: list[str] = []
        for sym, df in bars_map.items():
            if df is None or df.empty:
                continue
            try:
                saved_rows += self.bar_store.save_bars(
                    sym, df, timeframe="1d", allow_overwrite=False
                )
            except CorporateActionMismatchError as exc:
                msg = f"{sym}: {exc}"
                alerts.append(msg)
                logger.warning("corporate_action_mismatch", symbol=sym, error=str(exc))
                self.db.record_daemon_event(
                    self.daemon_name,
                    "error",
                    {
                        "kind": "corporate_action_mismatch",
                        "symbol": sym,
                        "error": str(exc),
                    },
                    timestamp=now_utc,
                )
        return saved_rows, alerts

    def _use_consolidated_yf_for_comparison_benchmarks(self) -> bool:
        return not self._injected_market_provider and self.settings.alpaca_data_feed == "iex"

    def _sync_consolidated_comparison_benchmarks(
        self,
        now_utc: datetime,
        *,
        period: str = "2y",
    ) -> int:
        if not self._use_consolidated_yf_for_comparison_benchmarks():
            return 0
        max_date_et = self._canonical_daily_end(now_utc).astimezone(NY_TZ).date()
        saved_total = 0
        for sym in self._comparison_benchmark_symbols():
            df = fetch_yfinance_consolidated_bars(
                sym, period=period, interval="1d", max_date_et=max_date_et
            )
            if not df.empty:
                saved_total += self.bar_store.save_bars(
                    sym, df, timeframe="1d", allow_overwrite=True
                )
        return saved_total

    def _sync_bars_gap_fill(self, symbols: list[str], now_utc: datetime) -> dict[str, Any]:
        provider = self._get_market_provider()
        if provider is None:
            return {"bars_saved": 0, "below_min_bars": [], "alerts": []}

        end_utc = self._canonical_daily_end(now_utc)
        use_yf_comp = self._use_consolidated_yf_for_comparison_benchmarks()
        sync_symbols = (
            list(symbols)
            if use_yf_comp
            else list(dict.fromkeys(symbols + self._comparison_benchmark_symbols()))
        )
        full_backfill_syms: list[str] = []
        recent_gap_syms: list[str] = []
        earliest_gap_latest_ts: datetime | None = None
        below_min: list[str] = []

        for sym in sync_symbols:
            existing = self.bar_store.load_bars(sym, timeframe="1d")
            count = len(existing)
            if count < self.min_bars_required:
                full_backfill_syms.append(sym)
            else:
                latest_ts = ensure_utc(existing["timestamp"].iloc[-1].to_pydatetime())
                if (end_utc - latest_ts) > timedelta(days=1, hours=12):
                    recent_gap_syms.append(sym)
                    if earliest_gap_latest_ts is None or latest_ts < earliest_gap_latest_ts:
                        earliest_gap_latest_ts = latest_ts

        bars_saved = 0
        alerts: list[str] = []

        if full_backfill_syms:
            start_full = end_utc - timedelta(days=520)
            if start_full < end_utc:
                fetched = provider.fetch_bars(
                    full_backfill_syms, start=start_full, end=end_utc, timeframe="1d"
                )
                saved, sym_alerts = self._save_daily_bars_safe(fetched, now_utc=now_utc)
                bars_saved += saved
                alerts.extend(sym_alerts)

        if recent_gap_syms:
            assert earliest_gap_latest_ts is not None
            start_recent = min(
                end_utc - timedelta(days=14),
                earliest_gap_latest_ts - timedelta(days=5),
            )
            if start_recent < end_utc:
                fetched = provider.fetch_bars(
                    recent_gap_syms, start=start_recent, end=end_utc, timeframe="1d"
                )
                saved, sym_alerts = self._save_daily_bars_safe(fetched, now_utc=now_utc)
                bars_saved += saved
                alerts.extend(sym_alerts)

        if use_yf_comp:
            bars_saved += self._sync_consolidated_comparison_benchmarks(now_utc, period="2y")

        for sym in symbols:
            if len(self.bar_store.load_bars(sym, timeframe="1d")) < self.min_bars_required:
                below_min.append(sym)

        return {
            "bars_saved": bars_saved,
            "below_min_bars": below_min,
            "alerts": alerts,
        }

    def _refresh_auxiliary_data(
        self,
        symbols: list[str],
        now_utc: datetime,
        *,
        lookback_days: int,
    ) -> dict[str, int]:
        macro_saved = 0
        earnings_saved = 0
        sec_saved = 0
        etf_holdings_saved = 0
        start_date = (now_utc - timedelta(days=lookback_days)).date()
        end_date = now_utc.date()
        equity_syms = self._equity_symbols()
        universe = load_universe(self.config_dir)
        etf_entries = [u for u in universe if u.asset_type == AssetType.etf]
        max_form4_per_symbol = 200
        try:
            app_cfg = load_app_config(self.config_dir)
            if app_cfg.strategy.w_insider <= 0.0:
                max_form4_per_symbol = 0
        except (FileNotFoundError, OSError, ValueError, TypeError, yaml.YAMLError):
            pass

        with httpx.Client(timeout=30.0) as http_client:
            # 1. FRED Macro (T10Y2Y, BAMLH0A0HYM2, VIXCLS, DGS3MO)
            try:
                macro_prov = self._macro_provider
                if macro_prov is None and self.settings.fred_api_key.strip():
                    macro_prov = FREDMacroProvider(
                        api_key=self.settings.fred_api_key,
                        client=http_client,
                    )
                if macro_prov is not None:
                    pts = macro_prov.fetch_macro(start_date=start_date, end_date=end_date)
                    macro_saved = self.db.upsert_macro(pts)
            except Exception as exc:  # noqa: BLE001
                logger.warning("macro_refresh_failed", error=str(exc))

            # 2. Earnings calendar (equities only — ETFs do not report corporate earnings)
            try:
                earn_prov = self._earnings_provider or YFinanceEarningsProvider()
                events = earn_prov.fetch_earnings(equity_syms)
                earnings_saved = self.db.upsert_earnings(events)
            except Exception as exc:  # noqa: BLE001
                logger.warning("earnings_refresh_failed", error=str(exc))

            # 3. SEC EDGAR Form 4 / 8-K (equities only)
            try:
                sec_prov = self._sec_provider
                if sec_prov is None and self.settings.sec_edgar_user_agent.strip():
                    sec_prov = SecEdgarProvider(
                        user_agent=self.settings.sec_edgar_user_agent,
                        cache_dir=self.data_dir / "sec_cache",
                        client=http_client,
                        max_form4_per_symbol=max_form4_per_symbol,
                    )
                if sec_prov is not None:
                    filings = sec_prov.fetch_filings(
                        symbols=equity_syms,
                        start=now_utc - timedelta(days=lookback_days),
                        end=now_utc,
                    )
                    sec_saved = self.db.upsert_sec_filings(filings)
            except Exception as exc:  # noqa: BLE001
                logger.warning("sec_refresh_failed", error=str(exc))

            # 4. ETF top holdings (for BLACKOUT_PERIOD concentration rail)
            if etf_entries:
                try:
                    etf_prov = self._etf_holdings_provider or YFinanceEtfHoldingsProvider()
                    holdings_by_etf = etf_prov.fetch_etf_holdings([u.symbol for u in etf_entries])
                    for etf_u in etf_entries:
                        if etf_u.symbol not in holdings_by_etf:
                            continue
                        weights = holdings_by_etf[etf_u.symbol]
                        is_non_eq = etf_u.sector in NON_EQUITY_ETF_SECTORS
                        if is_non_eq:
                            self.db.upsert_etf_holdings(
                                etf_u.symbol,
                                weights,
                                is_non_equity_etf=True,
                                updated_at=now_utc,
                            )
                            etf_holdings_saved += 1
                        elif weights:
                            self.db.upsert_etf_holdings(
                                etf_u.symbol,
                                weights,
                                is_non_equity_etf=False,
                                updated_at=now_utc,
                            )
                            etf_holdings_saved += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("etf_holdings_refresh_failed", error=str(exc))

        return {
            "macro_saved": macro_saved,
            "earnings_saved": earnings_saved,
            "sec_saved": sec_saved,
            "etf_holdings_saved": etf_holdings_saved,
        }

    def _poll_intraday_snapshots(self, symbols: list[str], now_utc: datetime) -> int:
        """Fetch latest snapshots in bounded batches without retaining DataFrames."""
        client = self._get_data_client()
        if client is None or not hasattr(client, "get_stock_snapshot"):
            return 0
        feed = DataFeed.SIP if self.settings.alpaca_data_feed == "sip" else DataFeed.IEX
        fresh_count = 0
        for idx in range(0, len(symbols), 25):
            batch = symbols[idx : idx + 25]
            try:
                req = StockSnapshotRequest(symbol_or_symbols=batch, feed=feed)
                resp = client.get_stock_snapshot(req)
                if isinstance(resp, dict):
                    for snap in resp.values():
                        if _extract_snapshot_price(snap) is not None:
                            fresh_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("intraday_snapshot_poll_failed", error=str(exc))
        if fresh_count > 0:
            self._latest_snapshot_at = now_utc.isoformat()
        self._latest_snapshot_count = fresh_count
        return fresh_count

    def get_sleep_seconds(self, now_utc: datetime | None = None) -> float:
        ref = ensure_utc(now_utc) if now_utc is not None else datetime.now(UTC)
        if check_market_open(ref, self._trading_client):
            return self.poll_interval_open_seconds
        return self.poll_interval_closed_seconds

    def startup(self, now_utc: datetime | None = None) -> None:
        now = ensure_utc(now_utc) if now_utc is not None else datetime.now(UTC)
        self.verify_wal_integrity(now_utc=now)
        self.write_heartbeat("starting", {"stage": "cold_boot"}, now_utc=now)
        self._boot_network_ok = wait_for_network(
            self._check_network,
            max_wait_seconds=self.max_network_wait_seconds,
            stop_event=self.stop_event,
            sleep_fn=self._sleep_fn,
        )
        symbols = self._universe_symbols()

        bar_summary: dict[str, Any] = {"bars_saved": 0, "below_min_bars": [], "alerts": []}
        aux_summary: dict[str, int] = {
            "macro_saved": 0,
            "earnings_saved": 0,
            "sec_saved": 0,
            "etf_holdings_saved": 0,
        }
        if self._boot_network_ok:
            try:
                bar_summary = self._sync_bars_gap_fill(symbols, now)
                self._boot_backfill_done = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("boot_bar_gapfill_failed", error=str(exc))
                self.db.record_daemon_event(
                    self.daemon_name,
                    "error",
                    {"kind": "boot_gapfill_error", "error": str(exc)},
                    timestamp=now,
                )
            counts = self.db.get_summary_counts()
            boot_aux_days = (
                14
                if (counts.get("sec_filings", 0) > 0 and counts.get("macro_points", 0) > 0)
                else 365
            )
            aux_summary = self._refresh_auxiliary_data(
                symbols, now, lookback_days=boot_aux_days
            )

        boot_details = {
            "network_ok": self._boot_network_ok,
            "boot_backfill_done": self._boot_backfill_done,
            "symbols_count": len(symbols),
            "bars_saved": bar_summary["bars_saved"],
            "below_min_bars_count": len(bar_summary["below_min_bars"]),
            **aux_summary,
        }
        self.db.record_daemon_event(
            self.daemon_name,
            "boot",
            boot_details,
            timestamp=now,
        )
        status = "healthy" if (self._boot_network_ok and self._boot_backfill_done) else "degraded"
        self.write_heartbeat(status, boot_details, now_utc=now)
        release_memory()

    def run_once(self, now_utc: datetime | None = None) -> dict[str, Any]:
        now = ensure_utc(now_utc) if now_utc is not None else datetime.now(UTC)
        now_et = now.astimezone(NY_TZ)
        session_date = now_et.date().isoformat()
        market_open = check_market_open(now, self._trading_client)
        symbols = self._universe_symbols()

        boot_recovered = False
        if not self._boot_backfill_done and self._check_network():
            try:
                self._sync_bars_gap_fill(symbols, now)
                self._refresh_auxiliary_data(symbols, now, lookback_days=365)
                self._boot_backfill_done = True
                self._boot_network_ok = True
                boot_recovered = True
                self.db.record_daemon_event(
                    self.daemon_name,
                    "boot",
                    {"recovered_post_outage": True, "symbols_count": len(symbols)},
                    timestamp=now,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("post_outage_boot_gapfill_failed", error=str(exc))
                self.db.record_daemon_event(
                    self.daemon_name,
                    "error",
                    {"kind": "boot_gapfill_error", "error": str(exc)},
                    timestamp=now,
                )

        all_mark_symbols = list(dict.fromkeys(symbols + self._comparison_benchmark_symbols()))
        snapshots_count = 0
        if market_open:
            snapshots_count = self._poll_intraday_snapshots(all_mark_symbols, now)

        eod_ran = False
        eod_bars_saved = 0
        eod_alerts: list[str] = []
        eod_failed = False
        pruned: dict[str, int] | None = None
        aux_summary: dict[str, int] = {
            "macro_saved": 0,
            "earnings_saved": 0,
            "sec_saved": 0,
            "etf_holdings_saved": 0,
        }

        eod_retry_ready = (
            self._last_eod_attempt_at is None
            or (now - self._last_eod_attempt_at).total_seconds() >= self.eod_retry_interval_seconds
        )
        if (
            is_nyse_trading_day(now_et.date())
            and now_et.time() >= dt_time(16, 5)
            and self._last_eod_session_date != session_date
            and eod_retry_ready
        ):
            self._last_eod_attempt_at = now
            provider = self._get_market_provider()
            eod_sync_ok = True
            if provider is not None:
                try:
                    eod_fetch_syms = (
                        list(symbols)
                        if self._use_consolidated_yf_for_comparison_benchmarks()
                        else all_mark_symbols
                    )
                    fetched = provider.fetch_bars(
                        eod_fetch_syms,
                        start=now - timedelta(days=7),
                        end=now,
                        timeframe="1d",
                    )
                    eod_bars_saved, eod_alerts = self._save_daily_bars_safe(
                        fetched, now_utc=now
                    )
                    if self._use_consolidated_yf_for_comparison_benchmarks():
                        eod_bars_saved += self._sync_consolidated_comparison_benchmarks(
                            now, period="1mo"
                        )
                except Exception as exc:  # noqa: BLE001
                    eod_sync_ok = False
                    eod_failed = True
                    logger.warning("eod_bars_fetch_failed", error=str(exc))
                    self.db.record_daemon_event(
                        self.daemon_name,
                        "error",
                        {"kind": "eod_fetch_error", "error": str(exc)},
                        timestamp=now,
                    )

            if eod_sync_ok:
                aux_summary = self._refresh_auxiliary_data(symbols, now, lookback_days=14)
                if self._last_prune_date != session_date:
                    pruned = self.db.prune_old_telemetry(keep_days=90, now_utc=now)
                    self._last_prune_date = session_date

                self._last_eod_session_date = session_date
                eod_ran = True
                self.db.record_daemon_event(
                    self.daemon_name,
                    "eod_sync",
                    {
                        "session_date": session_date,
                        "bars_saved": eod_bars_saved,
                        "alerts": eod_alerts,
                        "pruned": pruned,
                        **aux_summary,
                    },
                    timestamp=now,
                )
            release_memory()

        if eod_failed or not self._boot_backfill_done:
            hb_status = "degraded"
        elif market_open:
            hb_status = "healthy"
        else:
            hb_status = "idle_market_closed"
        details: dict[str, Any] = {
            "market_open": market_open,
            "boot_backfill_done": self._boot_backfill_done,
            "boot_recovered": boot_recovered,
            "snapshots_count": snapshots_count,
            "latest_snapshot_at": self._latest_snapshot_at,
            "eod_session_date": self._last_eod_session_date,
            "eod_ran": eod_ran,
            "eod_failed": eod_failed,
            "eod_bars_saved": eod_bars_saved,
            "alerts_count": len(eod_alerts),
        }
        self.write_heartbeat(hb_status, details, now_utc=now)
        return {"status": hb_status, **details}

    def shutdown(self) -> None:
        close_sdk_client(self._owned_data_client)
        self._owned_data_client = None
        if self._owns_trading_client:
            close_sdk_client(self._trading_client)
