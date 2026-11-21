# Phase 1 — Independent Adversarial Audit Gate B: Risk Engine & Short-Safety

- **Auditor Model Tier:** `pro` (isolated read-only subagent)
- **Initial Verdict:** `FAIL` (2 CRITICAL, 4 HIGH, 3 MEDIUM, 1 LOW)
- **Re-Audit Verdict:** `PASS` (0 CRITICAL, 0 HIGH, 0 MEDIUM remaining)

## Summary of Caught & Resolved Issues

| Severity | Finding | Resolution |
| :--- | :--- | :--- |
| **CRITICAL** | Side-flip open legs (`Long -> Short` / `Short -> Long`) misclassified into Pass 1 and rejected as `NO_OPEN_POSITION_TO_REDUCE`. | Tracked `remaining_reducible[sym]` during batch partitioning (prioritizing `is_risk_reducing=True`) so open legs always route cleanly to Pass 2 in `risk/engine.py`. |
| **CRITICAL** | Unpriced open positions understated exposure (failed open), and `order.reference_price` vs `state.prices` divergence enabled oversized entries & bad stops. | Failed closed on any missing/invalid open position price (`MISSING_OR_INVALID_PORTFOLIO_PRICE`) or `> 2%` reference-price divergence (`REFERENCE_PRICE_DIVERGENCE`), using conservative `max(order.reference_price, state_px)` sizing. |
| **HIGH** | Partial short covers could leave fractional short remainders; rounding overshoot or zero-qty approvals in Pass 1. | Enforced integer remaining short shares on partial covers (`FRACTIONAL_SHORT_REMAINDER_REJECTED`) and exact floor-to-6-decimals on long trims. |
| **HIGH** | Service-restart rate limit bypassable via `now`/wall-clock mismatch, `LIMIT 200` audit eviction, TOCTOU race, or reason spoofing. | Added dedicated `service_restart_log` table with atomic `BEGIN IMMEDIATE` transaction in `db.py` (`try_record_service_restart`) and preserved `updated_at` timestamps consistently. |
| **HIGH** | `_trading_days_apart` ignored NYSE holidays, weakening `EARNINGS_BLACKOUT` around holiday weekends. | Implemented `_NYSEHolidayCalendar` covering all NYSE holidays with symmetric weekend/holiday session handling. |
| **HIGH** | `_resolve_stop_loss` allowed wrong-side stops on non-positive `atr_pct` or 4-decimal rounding boundary, and unbounded long stops when signal absent. | Enforced `SignalRow.atr_pct > 0`, added `MAX_FALLBACK_LONG_STOP_PCT = 0.25`, and re-verified strict directional inequality *after* rounding. |
| **MEDIUM / LOW** | `PortfolioRiskState` unvalidated dataclass (`drawdown_pct > 0`), stale/future signals, `NaN` blacklist hours, empty-signal dials in `PortfolioConstructor`, config assignment validation & `HH:MM` format. | Converted `PortfolioRiskState` to frozen Pydantic model (`drawdown_pct <= 0`), enforced signal freshness & finite checks, hardened `PortfolioConstructor` empty-signal/sector/net scaling, and added `validate_assignment=True` + `HH:MM` regex in `config.py`. |
