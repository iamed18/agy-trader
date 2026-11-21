from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from agy_trader.core.config import LLMQuotaConfig
from agy_trader.core.models import ControlDials, LLMUsageRecord, ensure_utc
from agy_trader.storage.db import StateDB

ALLOWED_RESTART_SERVICES: frozenset[str] = frozenset({"market-data", "news-nlp", "trader"})
MAX_RESTARTS_PER_HOUR = 2


class ControlPlaneError(ValueError):
    """Raised when an Active Ops control plane action violates bounds or rate limits."""


class ControlPlane:
    """Strictly bounded, audited operational control plane for human and Active Ops Agent."""

    def __init__(self, db: StateDB) -> None:
        self.db = db

    def get_dials(self) -> ControlDials:
        return self.db.get_control_dials()

    def set_exposure(
        self,
        multiplier: float,
        *,
        actor: str = "ops",
        reason: str = "",
        now: datetime | None = None,
    ) -> ControlDials:
        if not reason.strip():
            raise ControlPlaneError("A non-empty reason is required for all control plane changes")
        current = self.db.get_control_dials()
        updated = ControlDials(
            pause_new_entries=current.pause_new_entries,
            exposure_multiplier=multiplier,
            allow_shorting=current.allow_shorting,
            blacklisted_symbols=dict(current.blacklisted_symbols),
            updated_by=actor,
            reason=reason.strip(),
            updated_at=ensure_utc(now) if now is not None else datetime.now(UTC),
        )
        return self.db.update_control_dials(updated, actor=actor, reason=reason.strip())

    def set_pause(
        self,
        paused: bool,
        *,
        actor: str = "ops",
        reason: str = "",
        now: datetime | None = None,
    ) -> ControlDials:
        if not reason.strip():
            raise ControlPlaneError("A non-empty reason is required for all control plane changes")
        current = self.db.get_control_dials()
        updated = ControlDials(
            pause_new_entries=paused,
            exposure_multiplier=current.exposure_multiplier,
            allow_shorting=current.allow_shorting,
            blacklisted_symbols=dict(current.blacklisted_symbols),
            updated_by=actor,
            reason=reason.strip(),
            updated_at=ensure_utc(now) if now is not None else datetime.now(UTC),
        )
        return self.db.update_control_dials(updated, actor=actor, reason=reason.strip())

    def set_shorting(
        self,
        enabled: bool,
        *,
        actor: str = "ops",
        reason: str = "",
        now: datetime | None = None,
    ) -> ControlDials:
        if not reason.strip():
            raise ControlPlaneError("A non-empty reason is required for all control plane changes")
        current = self.db.get_control_dials()
        updated = ControlDials(
            pause_new_entries=current.pause_new_entries,
            exposure_multiplier=current.exposure_multiplier,
            allow_shorting=enabled,
            blacklisted_symbols=dict(current.blacklisted_symbols),
            updated_by=actor,
            reason=reason.strip(),
            updated_at=ensure_utc(now) if now is not None else datetime.now(UTC),
        )
        return self.db.update_control_dials(updated, actor=actor, reason=reason.strip())

    def add_blacklist(
        self,
        symbol: str,
        hours: float = 48.0,
        *,
        actor: str = "ops",
        reason: str = "",
        now: datetime | None = None,
    ) -> ControlDials:
        if not reason.strip():
            raise ControlPlaneError("A non-empty reason is required for all control plane changes")
        if not math.isfinite(hours) or hours <= 0.0 or hours > 24.0 * 30.0:
            raise ControlPlaneError("Blacklist duration hours must be finite and in (0, 720]")
        cleaned = symbol.strip().upper()
        if not cleaned:
            raise ControlPlaneError("Symbol must not be empty")
        now_utc = ensure_utc(now) if now is not None else datetime.now(UTC)
        current = self.db.get_control_dials()
        # Prune already-expired entries while adding the new expiry
        new_map = {k: v for k, v in current.blacklisted_symbols.items() if v > now_utc}
        new_map[cleaned] = now_utc + timedelta(hours=hours)
        updated = ControlDials(
            pause_new_entries=current.pause_new_entries,
            exposure_multiplier=current.exposure_multiplier,
            allow_shorting=current.allow_shorting,
            blacklisted_symbols=new_map,
            updated_by=actor,
            reason=reason.strip(),
            updated_at=now_utc,
        )
        return self.db.update_control_dials(updated, actor=actor, reason=reason.strip())

    def remove_blacklist(
        self,
        symbol: str,
        *,
        actor: str = "ops",
        reason: str = "",
        now: datetime | None = None,
    ) -> ControlDials:
        if not reason.strip():
            raise ControlPlaneError("A non-empty reason is required for all control plane changes")
        cleaned = symbol.strip().upper()
        if not cleaned:
            raise ControlPlaneError("Symbol must not be empty")
        now_utc = ensure_utc(now) if now is not None else datetime.now(UTC)
        current = self.db.get_control_dials()
        new_map = {
            k: v for k, v in current.blacklisted_symbols.items() if k != cleaned and v > now_utc
        }
        updated = ControlDials(
            pause_new_entries=current.pause_new_entries,
            exposure_multiplier=current.exposure_multiplier,
            allow_shorting=current.allow_shorting,
            blacklisted_symbols=new_map,
            updated_by=actor,
            reason=reason.strip(),
            updated_at=now_utc,
        )
        return self.db.update_control_dials(updated, actor=actor, reason=reason.strip())

    def authorize_service_restart(
        self,
        service: str,
        *,
        actor: str = "ops",
        reason: str = "",
        now: datetime | None = None,
    ) -> None:
        """Validate service restart allowlist & enforce max 2 restarts/hour per service."""
        clean_reason = reason.strip()
        if not clean_reason:
            raise ControlPlaneError("A non-empty reason is required for service restarts")
        svc = service.strip().lower()
        if svc not in ALLOWED_RESTART_SERVICES:
            raise ControlPlaneError(
                f"Service {service!r} is not restartable via ops "
                f"(allowed: {sorted(ALLOWED_RESTART_SERVICES)})"
            )
        now_utc = ensure_utc(now) if now is not None else datetime.now(UTC)
        recorded = self.db.try_record_service_restart(
            service=svc,
            max_per_hour=MAX_RESTARTS_PER_HOUR,
            now_utc=now_utc,
            actor=actor,
            reason=clean_reason,
        )
        if not recorded:
            raise ControlPlaneError(
                f"Rate limit exceeded for {svc!r}: max {MAX_RESTARTS_PER_HOUR} restarts/hour"
            )

    def record_llm_usage(self, record: LLMUsageRecord) -> None:
        self.db.log_llm_usage(record)

    def check_llm_budget(
        self,
        quota: LLMQuotaConfig,
        now: datetime | None = None,
    ) -> tuple[bool, dict[str, float]]:
        now_utc = ensure_utc(now) if now is not None else datetime.now(UTC)
        return self.db.check_llm_budget(quota, now_utc)
