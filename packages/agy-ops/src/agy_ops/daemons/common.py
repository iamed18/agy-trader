from __future__ import annotations

import gc
import logging
import os
import resource
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from datetime import time as dt_time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import structlog

from agy_trader.core.calendar import is_nyse_trading_day
from agy_trader.core.models import ensure_utc
from agy_trader.storage.db import StateDB

NY_TZ = ZoneInfo("America/New_York")
logger = structlog.get_logger(__name__)


def get_process_rss_mb() -> float:
    """Return current process maximum RSS in megabytes (cross-platform macOS/Linux)."""
    maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return round(maxrss / (1024.0 * 1024.0), 2)
    return round(maxrss / 1024.0, 2)


def is_us_market_open_fallback(now_utc: datetime) -> bool:
    """Return True if now_utc falls within regular NYSE hours (09:30-16:00 ET)."""
    now_et = ensure_utc(now_utc).astimezone(NY_TZ)
    if not is_nyse_trading_day(now_et.date()):
        return False
    cur_t = now_et.time()
    return dt_time(9, 30) <= cur_t < dt_time(16, 0)


def get_market_clock_info(
    now_utc: datetime, trading_client: Any = None
) -> tuple[bool, datetime | None]:
    """Return (is_open, next_close_utc) from Alpaca /v2/clock, falling back to local NYSE calendar."""
    utc_dt = ensure_utc(now_utc)
    if trading_client is not None and hasattr(trading_client, "get_clock"):
        try:
            clock = trading_client.get_clock()
            is_open = getattr(clock, "is_open", None)
            if isinstance(is_open, bool):
                raw_next_close = getattr(clock, "next_close", None)
                next_close_utc: datetime | None = None
                if isinstance(raw_next_close, datetime):
                    next_close_utc = ensure_utc(raw_next_close)
                elif isinstance(raw_next_close, str) and raw_next_close.strip():
                    next_close_utc = ensure_utc(datetime.fromisoformat(raw_next_close.strip()))
                return is_open, next_close_utc
        except Exception as exc:  # noqa: BLE001 - fall back to deterministic calendar
            logger.debug("alpaca_clock_fallback", error=str(exc))
    return is_us_market_open_fallback(utc_dt), None


def check_market_open(now_utc: datetime, trading_client: Any = None) -> bool:
    """Check Alpaca /v2/clock if available, falling back to local NYSE calendar."""
    is_open, _ = get_market_clock_info(now_utc, trading_client=trading_client)
    return is_open


def configure_daemon_logging(
    data_dir: Path,
    daemon_name: str,
    *,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> RotatingFileHandler:
    """Attach a bounded RotatingFileHandler under <data_dir>/logs/<daemon_name>.log."""
    log_dir = Path(data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / f"{daemon_name}.log").resolve()

    root_logger = logging.getLogger()
    if root_logger.level == logging.NOTSET:
        root_logger.setLevel(logging.INFO)

    for existing in root_logger.handlers:
        if (
            isinstance(existing, RotatingFileHandler)
            and Path(existing.baseFilename).resolve() == log_path
        ):
            return existing

    handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    root_logger.addHandler(handler)
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.KeyValueRenderer(key_order=["event"]),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    return handler


def wait_for_network(
    check_fn: Callable[[], bool],
    *,
    max_wait_seconds: float = 120.0,
    base_delay: float = 2.0,
    stop_event: threading.Event | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bool:
    """Wait with exponential backoff until check_fn() succeeds or budget/stop is reached."""

    def _probe() -> bool:
        try:
            return bool(check_fn())
        except Exception as exc:  # noqa: BLE001  # noqa: BLE001 - network probes may raise socket/HTTP/SDK errors
            logger.warning("network_probe_failed", error=str(exc))
            return False

    if stop_event is not None and stop_event.is_set():
        return False
    if _probe():
        return True
    if max_wait_seconds <= 0.0 or base_delay <= 0.0:
        return False

    elapsed = 0.0
    attempt = 0
    while elapsed < max_wait_seconds:
        if stop_event is not None and stop_event.is_set():
            return False
        remaining = max_wait_seconds - elapsed
        delay = min(base_delay * (2**attempt), 30.0, remaining)
        if delay <= 0.0:
            break
        if stop_event is not None and sleep_fn is time.sleep:
            if stop_event.wait(timeout=delay):
                return False
        else:
            sleep_fn(delay)
        elapsed += delay
        attempt += 1
        if stop_event is not None and stop_event.is_set():
            return False
        if _probe():
            return True

    return False


def close_sdk_client(client: Any) -> None:
    """Close an Alpaca SDK client's underlying HTTP session if present."""
    if client is None:
        return
    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        close_fn()
        return
    session = getattr(client, "_session", None)
    if session is not None and hasattr(session, "close"):
        session.close()


def release_memory() -> None:
    """Trigger GC and release cached torch MPS/CUDA buffers if torch is loaded."""
    gc.collect()
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return
    try:
        backends = getattr(torch_mod, "backends", None)
        mps_backend = getattr(backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            mps_mod = getattr(torch_mod, "mps", None)
            if mps_mod is not None and hasattr(mps_mod, "empty_cache"):
                mps_mod.empty_cache()
        cuda_mod = getattr(torch_mod, "cuda", None)
        if cuda_mod is not None and cuda_mod.is_available():
            cuda_mod.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch_cache_cleanup_skipped", error=str(exc))


class DaemonRunner(ABC):
    """Base lifecycle runner for always-on agy-ops daemons."""

    def __init__(
        self,
        daemon_name: str,
        db: StateDB,
        *,
        poll_interval_seconds: float = 60.0,
        stop_event: threading.Event | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.daemon_name = daemon_name.strip()
        self.db = db
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.stop_event = stop_event or threading.Event()
        self._sleep_fn = sleep_fn
        self._signals_installed = False

    def install_signal_handlers(self) -> None:
        """Register SIGTERM/SIGINT handlers to set stop_event gracefully."""
        if self._signals_installed:
            return
        if threading.current_thread() is not threading.main_thread():
            return

        def _handle_signal(signum: int, _frame: Any) -> None:
            logger.info("daemon_signal_received", daemon=self.daemon_name, signum=signum)
            self.stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handle_signal)
            except ValueError:
                pass
        self._signals_installed = True

    def verify_wal_integrity(self, *, now_utc: datetime | None = None) -> None:
        """Verify SQLite WAL integrity via PRAGMA quick_check."""
        ok, detail = self.db.check_wal_integrity()
        if ok:
            return
        ts = ensure_utc(now_utc) if now_utc is not None else datetime.now(UTC)
        self.write_heartbeat(
            "critical",
            {"wal_ok": False, "wal_detail": detail},
            now_utc=ts,
        )
        self.db.record_daemon_event(
            self.daemon_name,
            "error",
            {"kind": "wal_integrity_failed", "detail": detail},
            timestamp=ts,
        )
        raise RuntimeError(f"SQLite WAL integrity check failed for {self.daemon_name}: {detail}")

    def write_heartbeat(
        self,
        status: str,
        details: dict[str, Any] | None = None,
        *,
        now_utc: datetime | None = None,
    ) -> None:
        """Persist bounded daemon heartbeat including pid and peak/current RSS MB."""
        rss_mb = get_process_rss_mb()
        payload = {
            "pid": os.getpid(),
            "rss_mb": rss_mb,
            "peak_rss_mb": rss_mb,
            **(details or {}),
        }
        self.db.upsert_daemon_heartbeat(
            self.daemon_name,
            status,
            payload,
            now_utc=now_utc,
        )

    def get_sleep_seconds(self, now_utc: datetime | None = None) -> float:
        """Return loop sleep duration for the next cycle."""
        return self.poll_interval_seconds

    def startup(self, now_utc: datetime | None = None) -> None:
        """Perform cold-boot checks prior to entering the main loop."""
        self.verify_wal_integrity(now_utc=now_utc)

    @abstractmethod
    def run_once(self, now_utc: datetime | None = None) -> dict[str, Any]:
        """Execute one deterministic daemon cycle."""

    def shutdown(self) -> None:
        """Release owned resources on shutdown."""

    def _wait_interval(self, seconds: float) -> None:
        if seconds <= 0.0 or self.stop_event.is_set():
            return
        if self._sleep_fn is time.sleep:
            self.stop_event.wait(timeout=seconds)
        else:
            self._sleep_fn(seconds)

    def run_forever(
        self,
        *,
        max_cycles: int | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        """Run startup, bounded-memory loop until stop_event or max_cycles, then shutdown."""
        self.install_signal_handlers()
        cycles = 0
        try:
            boot_now = ensure_utc(now_fn()) if now_fn is not None else None
            self.startup(now_utc=boot_now)
            while not self.stop_event.is_set():
                cycle_now = ensure_utc(now_fn()) if now_fn is not None else datetime.now(UTC)
                try:
                    self.run_once(now_utc=cycle_now)
                except Exception as exc:
                    logger.exception("daemon_cycle_error", daemon=self.daemon_name)
                    try:
                        self.db.record_daemon_event(
                            self.daemon_name,
                            "error",
                            {"error": str(exc)},
                            timestamp=cycle_now,
                        )
                        self.write_heartbeat(
                            "error",
                            {"error": str(exc)},
                            now_utc=cycle_now,
                        )
                    except Exception as hb_exc:  # noqa: BLE001
                        logger.warning(
                            "daemon_error_heartbeat_failed",
                            daemon=self.daemon_name,
                            error=str(hb_exc),
                        )
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    break
                self._wait_interval(self.get_sleep_seconds(now_utc=cycle_now))
        finally:
            self.shutdown()
