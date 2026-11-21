# Phase 1 — Independent Adversarial Audit Gate C: Backtest Accounting & Test Quality

- **Auditor Model Tier:** `pro` (isolated read-only subagent)
- **Initial Verdict:** `FAIL` (0 CRITICAL, 3 HIGH, 4 MEDIUM, 4 LOW)
- **Re-Audit Verdict:** `PASS` (0 CRITICAL, 0 HIGH, 0 MEDIUM remaining)

## Summary of Caught & Resolved Issues

| Severity | Finding | Resolution |
| :--- | :--- | :--- |
| **HIGH** | Dropped Pass 1 exits on missing bars + unconstrained Pass 2 fills at `t+1` open after overnight gaps (cash/exposure breaches & gap-through-stop entries). | Carried forward unexecuted Pass 1 exits without duplication, aborted Pass 2 entries that already gapped past their stop-loss, and re-checked live `t`-open equity, cash (`>= 0`), gross, net, and position headroom in `backtest/engine.py`. |
| **HIGH** | Carried-forward stale closes fed zombie signals with hardcoded `latest_bar_age_minutes=0.0`; stale `SPY` bars distorted benchmark Sharpe/vol. | Separated live bars from carried MTM marks, excluded stale/halted symbols from `pit_bars`/`pit_news`, passed true `latest_age_min` to `PortfolioRiskState`, and aligned `benchmark_closes` strictly on live `SPY` bars. |
| **HIGH** | Missing unmocked integration invariants & missing direct unit tests for `metrics.py`, `PortfolioConstructor`, `AlpacaMarketDataProvider`, partial closes/adds, and ETF slippage. | Added `tests/test_metrics_and_portfolio.py` and expanded `test_backtest_engine.py`, `test_risk_engine.py`, `test_cli.py`, `test_storage_and_data.py`, and `test_control_plane.py` to 55 comprehensive non-tautological tests. |
| **MEDIUM** | Bankruptcy return-stats blowup (`1e15`), 120-bar `EMA50` truncation drift, end-of-window reconciliation, and `latest_bar_age_minutes=None` failing open in `HardRiskEngine`. | Handled bankruptcy cleanly in `metrics.py`, widened feature lookback to 300 bars (`<0.0005%` EMA50 boundary weight), added `close_open_positions_at_end`, and changed `latest_bar_age_minutes=None` to fail closed (`STALE_MARKET_DATA`). |
| **LOW** | Ablation zero-weight config validation, CLI benchmark loading / empty-window / report UID collisions, key/timestamp normalization, and test blind spots. | All resolved across `engine.py`, `cli/main.py`, and `tests/`. |
