# Phase 1 — Independent Adversarial Audit Gate A: Data & Lookahead-Bias

- **Auditor Model Tier:** `pro` (isolated read-only subagent)
- **Initial Verdict:** `FAIL` (1 CRITICAL, 3 HIGH, 4 MEDIUM, 3 LOW)
- **Re-Audit Verdict:** `PASS` (0 CRITICAL, 0 HIGH remaining)

## Summary of Caught & Resolved Issues

| Severity | Finding | Resolution |
| :--- | :--- | :--- |
| **CRITICAL** | Alpaca bars are timestamped at period start (`00:00 ET` for daily), risking same-day close leakage at intraday checkpoints. | Normalized all Alpaca bars to canonical close/availability timestamps (`16:00 America/New_York` for `1d`, `+ delta` for intraday) before range filtering in `alpaca_market.py`. |
| **HIGH** | FRED macro series release on `T+1` morning/afternoon ET (not `00:00 UTC`), plus UTC-vs-ET date shift. | Implemented conservative `America/New_York` business-day + US Federal Holiday release schedules (`macro_point_available_at_utc`) in `features.py`. |
| **HIGH** | Incremental `Adjustment.ALL` Parquet splicing could create fake split/dividend crash cliffs. | Added `CorporateActionMismatchError` detection (`rel_diff > 1e-4` on overlapping closes) + explicit `allow_overwrite` in `parquet_store.py`. |
| **HIGH** | `+/-inf` bypassed Parquet OHLCV checks; corrupt `SPY` bar could zero-divide or black out entire universe. | Added `np.isfinite` checks across `parquet_store.py` & `alpaca_market.py`, and `_is_valid_price_series` safe fallback in `features.py`. |
| **MEDIUM** | `NaT` in `DatetimeTZDtype`, single-symbol unstandardized z-scores, UTC-midnight headline hash split & dropped multi-symbol re-tags, `yfinance` `limit=12` truncation, US holidays in FRED lag. | All resolved across `parquet_store.py`, `features.py`, `models.py`, `db.py`, and `macro_and_calendar.py`. |
| **LOW** | `AlpacaNewsProvider` JSON `null` fields, lexicon `"accelerat"` stem typo + phrase negation, `Bar.vwap > 0`, `update_news_scores` validation, `synthetic.py` `days < 8`, canonical `published_at` on hash collision. | All resolved across `alpaca_news.py`, `sentiment.py`, `models.py`, `db.py`, and `synthetic.py`. |
