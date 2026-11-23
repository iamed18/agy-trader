# V4 Trading Velocity, Execution Timing & Non-Options Risk Design (`v0.4`)

This note captures the quantitative and market-structure design rationale behind the `v0.4` (`C2_LiquidAdaptive`) promotion in [`config/strategy.yaml`](../config/strategy.yaml) and [`config/universe.yaml`](../config/universe.yaml), specifically addressing trading frequency, execution timing, non-reactivity to intraday news/insider feeds, and crash/overnight-gap protection in an **unlevered, equities-only (`no options`) Alpaca paper/live architecture**.

---

## 1. Why Monthly Velocity Replaced Daily Churn

Across the 10-year survivorship-free Point-in-Time (PIT) S&P 500 Top-200 benchmark (`2017-01-03` to `2026-11-19`, `2,485` trading days, net of `10 bps`/side equity slippage, `0.28 bps` SEC sell fees, `10%` delisting haircuts, and PIT FRED `DGS3MO` cash yield):

| Strategy | Rebalance Cadence | 10Y CAGR | OOS CAGR (`2022–2026`) | 10Y Excess Sharpe | 10Y MaxDD | Annual Turnover |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `v0.2` (`T0`) | Daily (`every_n_days: 1`, `k=6`, `3.5x` ATR stop) | `+7.96%` | `+9.78%` | `0.423` | `-23.26%` | `6.31x` |
| **`v0.4` (`C2_LiquidAdaptive`)** | **Monthly (`monthly_first` @ `15:40 ET`, `k=25`)** | **`+20.24%`** | **`+16.07%`** | **`0.862`** | **`-25.61%`** | **`2.98x`** |
| `SPY` Baseline | Buy-and-hold | `+14.14%` | `+10.10%` | `0.685` | `-33.79%` | — |
| `EW PIT S&P 500 Top-200` | Monthly equal-weight | `+13.51%` | `+10.09%` | `0.636` | `-37.05%` | — |

Key velocity takeaways:
- Daily rebalancing combined with tight `3.5x ATR` stops and narrow `top_k_long: 6` leadership concentration lost return to churn, slippage, and post-drawdown cash lockups.
- Batching portfolio updates to once per month (`rebalance_schedule: "monthly_first"` with `hold_rank_buffer: 8`, `rebalance_deadband_pct: 0.015`, `top_k_long: 25`, and responsive `63d` + `126d` skip=0 volatility-adjusted momentum + `0.20` FIP + `0.30` PIT `log(MDDV)` + `sqrt(MDDV)` sizing) cut turnover by more than half (`6.31x -> 2.98x`) and extended average holding period to **`107.99` days** (`105.88` days OOS).

### Expected Order Cadence in Production

From `reports/v4_monthly_research/summary.json` (`119` monthly rebalances over `2,485` trading days):
- **Calendar-day average:** **`0.79` filled orders/day** (`1,952` orders over `2,485` days; **`0.82`/day** in `2022–2026` OOS across `1,226` days).
- **Non-rebalance trading days (~20 of 21 sessions/month):** **`0` trades on ~98.6% of days**. Intra-month trades occur only if a resting `10x ATR` protective stop triggers (`34` stop exits over 10 years, ~`3.4`/yr; `11` in 5Y OOS, ~`2.2`/yr).
- **Monthly rebalance day (`1st` NYSE trading day @ `15:40 ET`):** **`~16.1` orders** on average (`1,918` non-stop orders / `119` months — rotations, exits, and re-weights above `1.5%`; slightly fewer minor re-weight orders may occur live due to whole-share sizing with `allow_fractional_longs: false`).
- **Mid-month Day-1 deployment (`2026-11-23` @ `15:40 ET`):** Expect up to **`~25` opening buy orders** when deploying from `0` positions (excluding any symbols blocked by `BLACKOUT_PERIOD` or Item 4.02 restatement blackouts; earnings entry blackouts are disabled per Section 4).

*(Note on benchmark figures: `reports/v4_monthly_research/summary.json` runs the fractional R&D spec `allow_fractional_longs: true` yielding `+20.24%` 10Y CAGR / `+16.07%` OOS / `0.862` XS-SR / `1,952` orders, whereas the live whole-share `allow_fractional_longs: false` baseline in Section 4 yields `+20.05%` 10Y / `+15.89%` OOS / `0.857` XS-SR / `1,954` orders.)*

---

## 2. Execution Window Rationale (`15:40 ET` vs Earlier Intraday / Next Open)

- **Microstructure stability:** Market-open (`09:30–10:15 ET`) carries elevated spread width and overnight imbalance noise; `15:40 ET` accesses deep late-session liquidity while avoiding the final 5-minute (`15:55–16:00 ET`) MOC auction dislocation and reserving a 20-minute execution buffer before close.
- **Half-day handling:** [`TraderDaemon._active_rebalance_checkpoint`](../packages/agy-ops/src/agy_ops/daemons/trader.py) automatically shifts the checkpoint window to `[next_close - 20m, next_close - 2m)` on NYSE early-close sessions via Alpaca's `/v2/clock`.
- **Empirical backtest check:** Executing near the daily close (`same_close`: `+20.24%` 10Y CAGR, `0.862` Excess Sharpe, `+16.07%` OOS) marginally outperformed waiting for the next session's open (`next_open`: `+20.03%` 10Y CAGR, `0.853` Excess Sharpe, `+15.52%` OOS) while eliminating overnight signal-to-fill drift.

---

## 3. Why Intraday News & Insider Weights Are Set to `0.0`

V4 is deliberately **non-reactive day-to-day**: neither `news-nlp` nor `market-data` initiates trades on breaking headlines.

1. **`w_insider: 0.0` (empirically rejected in Phase 1):** Enabling `w_insider = 0.10` degraded 5-year Excess Sharpe (`1.403 -> 1.338`) and MaxDD (`-9.80% -> -11.41%`), and cut 1-year `Price + News` returns (`+16.35% -> +9.13%`, `0.901 -> 0.446` Excess Sharpe, turnover `7.02x -> 8.43x`), because scheduled 10b5-1 executive sales routinely penalized leading mega-cap uptrends. SEC Form 8-K **Item 4.02 accounting restatement blackouts** (`restatement_blackout_days: 10`) remain active as a hard entry block.
2. **`w_news_sentiment: 0.0` (history coverage & horizon mismatch):** Cached FinBERT news in `data/state.db` only extends ~1 year (`Finnhub` from `2025-12-11`, `Fiscal.ai` from `2026-04-29`, originally for 50 tickers) and therefore could not be validated over the 10-year `1,112`-ticker survivorship-free study (`use_news=False`). Moreover, the sentiment aggregator uses a `48`-hour half-life / `120`-hour lookback—a short-horizon signal suited to daily rebalancing (where 1Y daily 50-ticker testing improved from `+6.75%` to `+16.35%`) rather than a `~108`-day monthly holding period. `news-nlp` remains running for digests, dashboard monitoring, and building future 215-symbol archive history.

---

## 4. Holding Crash Protection & Overnight-Gap Constraints (Without Options)

### Active Four-Layer Protection
1. **Broker-resident `GTC` `10x ATR` Stop Orders:** On buys, `HardRiskEngine` calculates `stop_price = ref * max(0.05, 1.0 - 10.0 * atr_pct)`. Immediately after fills, [`AlpacaBroker.sync_protective_stops`](../packages/agy-core/src/agy_trader/execution/alpaca_broker.py) attaches a `StopOrderRequest` (`side=sell`, `time_in_force=GTC`) directly on Alpaca's servers. Live `strategy.yaml` enforces `allow_fractional_longs: false` because Alpaca restricts `GTC` stops on fractional quantities (`< 1` share / fractional remainders).
2. **Non-Widening Stop Ratchets:** When an existing long is added to on a later monthly rebalance, the new stop ratchets upward (`max(candidate_stop, prev_stop)` below current price) and never widens downward if ATR spikes.
3. **30-Second Self-Healing Reconciliation:** While the market is open (and on boot), `TraderDaemon` reconciles broker positions/orders via `reconcile_with_broker`, auto-repairing missing/undersized stops (falling back to `-10%` if no prior stop exists) and canceling duplicate/wrong-side/orphan stops.
4. **Portfolio Kill-Switches & Compliance Precedence:** Breaching `daily_loss_halt_pct: 0.08` (`-8%` day loss) or `max_drawdown_halt_pct: 0.35` (`-35%` peak-to-trough) pauses new entries automatically. Separately, any symbol in `BLACKOUT_PERIOD` is frozen completely (`sync_protective_stops` leaves its existing resting orders untouched).

### Overnight-Gap Reality Without Options
- Standard `GTC` stop orders trigger only during regular hours (`09:30–16:00 ET`), converting to a market sell; Alpaca does **not** support `stop` or `stop_limit` orders in extended hours (`04:00–09:30`, `16:00–20:00 ET`, where only `DAY` `Limit` orders are permitted).
- Even with extended-hours polling, news gaps reprice quotes in milliseconds across thin after-hours order books. **In the absence of options contracts, no order type can guarantee a fill above an overnight gap.**

| Approach (Equities-Only vs Options) | Gap Behavior | Expected Cost / Trade-Off |
| :--- | :--- | :--- |
| **25-stock diversification + `12%` cap (`current`)** | `~3%–5%` typical weight bounds a `-40%` single-stock gap to `~1.2%–2.0%` portfolio impact | Zero hedge drag (`$0`); absorbs single-stock shocks while wide `10x ATR` GTC stops cut multi-week meltdowns |
| **Earnings entry blackout (`earnings_blackout_days: null`, disabled live)** | Skipping new entries near earnings forfeits pre-earnings & PEAD drift (`+2.64%` to `+3.57%` 21d avg return on blocked picks) | Disabling `±2d` added `+0.76%`/yr on 10Y PIT S&P 500 Top-200 (`+19.29% -> +20.05%`) and `+1.29%`/yr on Live 200 (`+23.65% -> +24.94%`) with lower/flat MaxDD; backfill `#26+` worsened MaxDD, while `pre_2d + rescale_remaining` is the best fallback if ever desired |
| **Delayed-open / collared exit on gap-down** | Avoids 09:30:00 opening-auction print volatility | Loses server-side GTC autonomy and exposes position to deeper intraday selloff |
| **Protective puts *(requires options support)*** | Contractual floor across overnight gaps | Rolling single-stock puts on 25 high-momentum names typically costs `~5%–10%+/yr` in implied-vol premium; far-OTM `SPY` puts cost less (`~1%–2%`/yr) but hedge only broad-market shocks |

---

## 5. Future R&D Revisit Checklist

1. **Automated Monthly PIT Universe Reconstitution (Operational Priority before `2026-12-01`):** Wire a scheduled monthly refresh of [`config/universe.yaml`](../config/universe.yaml) and bar/earnings/SEC backfill for newly entering S&P 500 Top-200 names ahead of each `monthly_first` rebalance.
2. **Pre-Earnings Forced-Exit Ablation:** Backtest closing existing positions before earnings (distinct from the now-disabled entry-only block, or testing `pre_2d + rescale_remaining` if earnings risk tolerance narrows) on the 10-year V4 dataset.
3. **News Veto / Post-Earnings Drift (`w_pead`):** Once 215-symbol FinBERT history accrues, test negative-sentiment entry vetoes (rather than noisy 48-hour cross-sectional blends) on monthly rebalances.
4. **Portfolio Tail-Hedge Evaluation (If Options Are Added Later):** Benchmark rolling far-OTM `SPY` puts vs cash/`C3_RegimeManaged` Daniel–Moskowitz scaling during crisis regimes.
