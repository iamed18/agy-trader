from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from agy_trader.core.config import LLMQuotaConfig
from agy_trader.core.control_plane import ControlPlane, ControlPlaneError
from agy_trader.core.models import ControlDials, LLMUsageRecord
from agy_trader.storage.db import StateDB


def test_control_dials_exposure_multiplier_bounds() -> None:
    d = ControlDials(exposure_multiplier=0.5)
    assert d.exposure_multiplier == 0.5

    for bad in (1.01, -0.01, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            ControlDials(exposure_multiplier=bad)


def test_control_plane_audit_trail_timestamps_and_blacklist_expiry(tmp_path) -> None:
    db = StateDB(tmp_path / "state.db")
    cp = ControlPlane(db)
    t0 = datetime(2025, 5, 1, 12, 0, tzinfo=UTC)

    # Empty / whitespace reason must fail on all mutations
    with pytest.raises(ControlPlaneError, match="reason"):
        cp.set_exposure(0.5, reason="   ", now=t0)
    with pytest.raises(ControlPlaneError, match="reason"):
        cp.set_pause(True, reason="", now=t0)
    with pytest.raises(ControlPlaneError, match="reason"):
        cp.set_shorting(False, reason="", now=t0)
    with pytest.raises(ControlPlaneError, match="reason"):
        cp.add_blacklist("AAPL", hours=24.0, reason="", now=t0)
    with pytest.raises(ControlPlaneError, match="reason"):
        cp.remove_blacklist("AAPL", reason="", now=t0)

    # Invalid blacklist hours & empty symbol rejected
    for bad_hours in (0.0, -5.0, 721.0, float("nan"), float("inf")):
        with pytest.raises(ControlPlaneError, match="hours"):
            cp.add_blacklist("AAPL", hours=bad_hours, reason="test", now=t0)
    with pytest.raises(ControlPlaneError, match="Symbol"):
        cp.add_blacklist("   ", hours=24.0, reason="test", now=t0)
    with pytest.raises(ControlPlaneError, match="Symbol"):
        cp.remove_blacklist("  ", reason="test", now=t0)

    # Valid mutations preserve supplied timestamps and log immutable audit trail
    d1 = cp.set_exposure(0.6, actor="alice", reason="vol spike", now=t0)
    assert d1.exposure_multiplier == 0.6
    assert d1.updated_at == t0

    t1 = t0 + timedelta(hours=1)
    cp.set_pause(True, actor="bob", reason="halting entries", now=t1)
    t2 = t0 + timedelta(hours=2)
    cp.set_shorting(False, actor="bob", reason="squeeze regime", now=t2)
    t3 = t0 + timedelta(hours=3)
    d_bl = cp.add_blacklist("aapl", hours=12.0, actor="ops", reason="sec probe", now=t3)
    assert "AAPL" in d_bl.active_blacklist(t3 + timedelta(hours=1))
    assert "AAPL" not in d_bl.active_blacklist(t3 + timedelta(hours=13))

    # Adding another symbol after AAPL expired prunes AAPL cleanly
    t4 = t3 + timedelta(hours=13)
    d_bl2 = cp.add_blacklist("MSFT", hours=24.0, actor="ops", reason="earnings", now=t4)
    assert "AAPL" not in d_bl2.blacklisted_symbols
    assert "MSFT" in d_bl2.blacklisted_symbols
    assert d_bl2.updated_at == t4

    # Removing MSFT leaves empty blacklist
    t5 = t4 + timedelta(minutes=10)
    d_unbl = cp.remove_blacklist("msft", actor="ops", reason="cleared", now=t5)
    assert d_unbl.blacklisted_symbols == {}

    # Audit log has 6 immutable entries in reverse chronological insertion order
    audit = db.get_control_audit_log(limit=20)
    assert len(audit) == 6
    assert audit[0]["actor"] == "ops"
    assert audit[0]["reason"] == "cleared"
    assert audit[-1]["actor"] == "alice"
    assert audit[-1]["dials"]["exposure_multiplier"] == 0.6


def test_authorize_service_restart_allowlist_and_rate_limit_across_250_audit_writes(
    tmp_path,
) -> None:
    db = StateDB(tmp_path / "state.db")
    cp = ControlPlane(db)
    t0 = datetime(2025, 5, 1, 10, 0, tzinfo=UTC)

    # Disallowed services & empty reason rejected
    with pytest.raises(ControlPlaneError, match="not restartable"):
        cp.authorize_service_restart("postgres", reason="bad service", now=t0)
    with pytest.raises(ControlPlaneError, match="reason"):
        cp.authorize_service_restart("trader", reason="   ", now=t0)

    # 2 restarts within 1 hour allowed
    cp.authorize_service_restart("trader", reason="restart 1", now=t0)
    cp.authorize_service_restart("trader", reason="restart 2", now=t0 + timedelta(minutes=20))

    # Spam 250 other dial audit entries in between to verify service_restart_log is immune
    # to control_audit_log pagination/eviction
    for i in range(250):
        mult = round(0.5 + (i % 50) * 0.01, 2)
        cp.set_exposure(
            mult,
            actor="ops-spam",
            reason=f"dial tweak {i}",
            now=t0 + timedelta(minutes=25, seconds=i),
        )

    # 3rd restart within 1 hour of t0 is STILL blocked
    with pytest.raises(ControlPlaneError, match="Rate limit exceeded"):
        cp.authorize_service_restart("trader", reason="restart 3", now=t0 + timedelta(minutes=45))

    # Other allowed service has its own independent bucket
    cp.authorize_service_restart("market-data", reason="md restart", now=t0 + timedelta(minutes=45))

    # After > 1 hour elapses from t0 (first restart drops out of 1h window), restart succeeds again
    cp.authorize_service_restart(
        "trader", reason="restart after cooldown", now=t0 + timedelta(hours=1, seconds=1)
    )


def test_control_plane_check_llm_budget_daily_and_monthly_caps(tmp_path) -> None:
    db = StateDB(tmp_path / "state.db")
    cp = ControlPlane(db)
    now = datetime(2025, 6, 15, 16, 0, tzinfo=UTC)

    # Monthly cap reached across earlier days
    cp.record_llm_usage(
        LLMUsageRecord(
            timestamp=datetime(2025, 6, 5, 10, 0, tzinfo=UTC),
            tier=2,
            model="pro",
            task="digest",
            prompt_tokens=2000,
            completion_tokens=1000,
            estimated_usd=5.00,
        )
    )
    allowed_m, info_m = cp.check_llm_budget(
        LLMQuotaConfig(daily_usd_cap=2.0, monthly_usd_cap=5.0), now=now
    )
    assert allowed_m is False
    assert info_m["monthly_spend"] == pytest.approx(5.0)

    # Daily cap reached on current day
    cp.record_llm_usage(
        LLMUsageRecord(
            timestamp=now - timedelta(hours=2),
            tier=1,
            model="flash",
            task="news",
            prompt_tokens=1000,
            completion_tokens=200,
            estimated_usd=2.00,
        )
    )
    allowed_d, info_d = cp.check_llm_budget(
        LLMQuotaConfig(daily_usd_cap=2.0, monthly_usd_cap=50.0), now=now
    )
    assert allowed_d is False
    assert info_d["daily_spend"] == pytest.approx(2.0)

    # Higher caps allow execution
    allowed_ok, _ = cp.check_llm_budget(
        LLMQuotaConfig(daily_usd_cap=5.0, monthly_usd_cap=50.0), now=now
    )
    assert allowed_ok is True


def test_control_plane_exact_expiry_boundary_and_db_persistence(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    db1 = StateDB(db_path)
    cp1 = ControlPlane(db1)

    t0 = datetime(2025, 7, 1, 12, 0, tzinfo=UTC)
    cp1.set_exposure(0.45, actor="alice", reason="tighten exposure", now=t0)
    cp1.set_pause(True, actor="alice", reason="halt new entries", now=t0 + timedelta(minutes=1))
    cp1.set_shorting(False, actor="alice", reason="disable shorts", now=t0 + timedelta(minutes=2))
    dials_bl = cp1.add_blacklist(
        "NVDA", hours=6.0, actor="alice", reason="event risk", now=t0 + timedelta(minutes=3)
    )

    expiry = dials_bl.blacklisted_symbols["NVDA"]
    assert "NVDA" in dials_bl.active_blacklist(expiry - timedelta(microseconds=1))
    # Exact boundary expiry == as_of must be treated as expired!
    assert "NVDA" not in dials_bl.active_blacklist(expiry)

    # Re-instantiate StateDB & ControlPlane from disk to verify SQLite persistence
    db2 = StateDB(db_path)
    cp2 = ControlPlane(db2)
    persisted = cp2.get_dials()
    assert persisted.exposure_multiplier == pytest.approx(0.45)
    assert persisted.pause_new_entries is True
    assert persisted.allow_shorting is False
    assert persisted.blacklisted_symbols == {"NVDA": expiry}
    assert "NVDA" not in persisted.active_blacklist(expiry)

