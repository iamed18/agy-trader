# Phase 1 Independent Adversarial Audit — Real-Data Tuning, FinBERT & Turnover Controls

- **Date:** 2026-11-21
- **Auditor:** Read-only `pro` Adversarial Auditor (`Tuning Diff Adversarial Auditor`)
- **Verdict:** **PASS** (`0 CRITICAL`, `0 HIGH`, `0` remaining code `MEDIUM` after remediation)

## Scope Reviewed
- `config/strategy.yaml`
- `src/agy_trader/core/{config,models}.py`
- `src/agy_trader/strategy/{features,portfolio}.py`
- `src/agy_trader/backtest/engine.py`
- `src/agy_trader/nlp/sentiment.py`
- `src/agy_trader/cli/main.py`
- `tests/test_{sentiment_and_features,metrics_and_portfolio,backtest_engine}.py`

## Audit Vectors Verified
1. **Lookahead Bias / Point-in-Time Safety (`PASS`):**
   - `spy_sma200_defensive_scale` uses strictly `timestamp <= as_of_utc` bars with `_is_valid_price_series` validation.
   - `abs_trend_raw` and `news_article_count` missing-news renormalization are strictly point-in-time (`0.000000` delta verified across 2025 when no news existed) and zero-division guarded.
   - `rebalance_every_n_days` preserves strict bar-$t$-close signal → bar-$(t+1)$-open execution.
2. **Risk Engine & Short-Safety Invariants (`PASS`):**
   - `config/strategy.yaml` `risk:` caps and `HardRiskEngine` remain 100% untouched.
   - Hysteresis (`hold_rank_buffer`, `exit_signal_threshold`), `abs_momentum_filter`, and `short_only_in_bear_regime` cannot override `blacklist`, `allow_shorting=False`, `pause_new_entries=True`, `shortable`, `easy_to_borrow`, or squeeze caps.
   - Non-rebalance days still run short borrow accrual, open-gap stops, intraday stops, carried Pass-1 exits, and drawdown/daily-loss tracking.

## Findings & Remediation

| Severity | Finding | Resolution |
| :--- | :--- | :--- |
| `MEDIUM` | `rebalance_deadband_pct: 0.04` could suppress Active Ops `exposure_multiplier` and `macro_regime_scale` risk-reducing trims on an 8-long portfolio (`7.5% -> 3.75%` delta `< 4%`). | **Fixed** in `portfolio.py` (`build_proposed_orders`): risk-reducing trims now bypass `rebalance_deadband_pct` whenever portfolio gross/net exceeds scaled budget or a position exceeds its single-position cap. Covered by `test_deadband_bypassed_for_ops_and_macro_derisking_trims`. |
| `MEDIUM` | Phase 2 live-vs-backtest parity requirements for `rebalance_every_n_days` persistence, single canonical EOD signal evaluation, and `>=200` SPY bar warm-up for `spy_sma200_defensive_scale`. | **Documented** in `docs/HANDOFF.md` as mandatory Phase 2 daemon design constraints. |
| `LOW` | Potential dual-selection overlap if `exit_signal_threshold <= -min_signal_threshold`. | **Fixed** in `portfolio.py`: `selected_long_syms` excluded from `short_cands`. Covered by `test_no_dual_selection_overlap_when_exit_threshold_negative`. |
| `LOW` | One-sided net bounds (`max_net < 0` or `min_net > 0`) would relax toward 0 when scaled. | **Fixed** in `portfolio.py`: clamped via `min(max_net, max_net * risk_scale)` / `max(min_net, min_net * risk_scale)`. Covered by `test_one_sided_net_bounds_tighten_never_widen`. |
| `LOW` | `FinBERTScorer` clamped `NaN` via `min(1.0, nan)`. | **Fixed** in `sentiment.py`: explicit `math.isfinite` guard raising `ValueError`. Covered by `test_finbert_scorer_rejects_non_finite_probabilities`. |
| `LOW` | CLI defaults (`--backend lexicon`, un-filtered `get_all_scored_news()`). | **Noted** in `docs/HANDOFF.md` for Phase 2 CLI/daemon polish. |

## Post-Remediation Verification
- `uv run --offline ruff check src tests`: **0 errors**
- `uv run --offline mypy src tests`: **0 errors across 34 files**
- `uv run --offline pytest -v`: **67/67 passed**
