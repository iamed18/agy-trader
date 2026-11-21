# `agy-trader` — Agent & Local Host Handoff Document

> **For a new local agent session on `local-host`:** Read this entire file before taking action. It captures every locked-in design decision, the completed Phase 1 state, mandatory engineering/quota/audit policies, immediate next steps on real Alpaca data, and the Phase 2–3 blueprint.

---

## 1. Current Status: Phase 1 Complete (Paused at Stop Gate)

- **Repository:** `https://github.com/iamed18/agy-trader` (`main`)
- **Completed Scope:** **Phase 1** (Foundation, Local SQLite WAL + Parquet Storage, Alpaca Historical Bars & Benzinga News Backfill, FRED Macro & `yfinance` Earnings Calendar, Local FinBERT & Lexicon NLP Sentiment Scorer + Point-in-Time Aggregator, Point-in-Time Feature Engine, ATR Inverse-Vol Long/Short Portfolio Constructor, Immutable Hard Risk Engine, Audited Bounded Control Plane, Event-Driven Backtester + 4-Way A/B Ablation + Walk-Forward, and Phase 1 `agy` CLI).
- **Verification State:**
  - `.venv/bin/ruff check src tests` -> **0 errors**
  - `.venv/bin/mypy src tests` -> **0 errors across 34 files**
  - `.venv/bin/pytest -v` -> **55/55 tests passing**
- **Adversarial Audit State:** All 3 Phase 1 Independent Adversarial Audit Gates cleared with **0 `CRITICAL`, 0 `HIGH`, and 0 `MEDIUM` findings remaining**:
  - [docs/audits/phase1_gate_a.md](audits/phase1_gate_a.md) — Data & Lookahead Bias (`PASS`)
  - [docs/audits/phase1_gate_b.md](audits/phase1_gate_b.md) — Risk Engine & Short-Safety Invariants (`PASS`)
  - [docs/audits/phase1_gate_c.md](audits/phase1_gate_c.md) — Backtest Accounting & Test Rigor (`PASS`)

---

## 2. Locked-In Design Decisions

All architectural decisions below were explicitly agreed with the user during discovery (`/grill-me`):

| Dimension | Locked-In Choice |
| :--- | :--- |
| **Time Horizon** | **Swing trading** (hold days to weeks; avoid PDT rule and millisecond HFT competition; evaluate signals at `10:00`, `13:00`, `15:40` ET) |
| **Decision Architecture** | **100% Deterministic Quantitative Execution** (fully backtestable) + **Active Ops AI Agent Supervisor** (no LLM discretionary trade orders) |
| **Asset Universe** | **50 curated liquid US symbols** in `config/universe.yaml`: 15 core/sector ETFs + 35 large-cap stocks with GICS sectors |
| **News & Events** | Real-time + historical **Alpaca/Benzinga news** scored locally via quantitative NLP (**FinBERT** `ProsusAI/finbert`, with zero-ML `lexicon-v1` fallback) fed directly into trade signals alongside price, volume, macro regime, and earnings calendar |
| **Risk Posture** | **Long/Short Market-Neutral capable** with immutable hard Risk Engine caps (10% max long, 6% max short, 25% sector gross, 100% max gross, `[-25%, +60%]` net, `-2%` daily halt, `-8%` max drawdown halt, `±2` NYSE trading-day earnings blackout, mandatory `shortable` + `easy_to_borrow`, whole-share shorts only, squeeze filters, mandatory directional hard stop-losses, and `allow_shorting` toggle) |
| **Live Safety Lock** | Paper mode is default (`ALPACA_PAPER=true`). Live mode requires `ALPACA_PAPER=false`, live API keys, AND `LIVE_TRADING_CONFIRMATION=I_ACCEPT_REAL_MONEY_RISK` (`core/config.py:validate_trading_mode`) |
| **Monitoring Agent Power** | **Active Ops Agent** constrained strictly to bounded, audited `agy ops ...` commands: read telemetry/P&L/news, pause/resume new entries, dial `exposure_multiplier` strictly in `[0.0, 1.0]`, toggle `allow_shorting`, add TTL symbol blacklists, and restart stuck daemons (max 2/hr/service via atomic SQLite `service_restart_log`). **Cannot** place arbitrary trades, raise hard risk caps, or edit strategy code (writes Markdown proposals instead) |
| **Data Tier & Budget** | **\$0/mo starter**: Alpaca Paper + Basic IEX bars + Benzinga News + free FRED macro + free `yfinance` earnings, behind pluggable provider interfaces (`ALPACA_DATA_FEED=sip` flip later) |
| **Observability** | `agy` CLI (`--json` on every command) + read-only local FastAPI web dashboard (`127.0.0.1:8080`, Phase 3) + webhook alerts (Discord/Slack/Telegram/email, Phase 3) |
| **Target Host & Packaging** | Always-on Linux host (`local-host`) running **native Python 3.12 services managed by `systemd`** (with optional Docker Compose) |
| **User Communication** | User is new to systematic/algorithmic trading — explain quant/trading concepts clearly in plain language and recommend safe defaults when asking design questions |

---

## 3. Mandatory Engineering Policies (Carry Forward to Phases 2 & 3)

1. **Quota Management & Tiered Model Routing:**
   - **Build-time subagents:** Do not waste premium tokens on mechanical code, boilerplate, or repetitive test runs.
     - `flash_lite`: running `pytest`/`ruff`/`mypy`, CLI smoke checks, static YAML/systemd boilerplate, simple fixtures.
     - `flash`: standard adapters, CRUD, CLI wiring, dashboard templates, unit/integration tests.
     - `inherit` / `pro`: high-stakes execution/reconciliation/risk/quant logic and independent adversarial audits.
   - **Runtime `agy-supervisor` (Phase 3):**
     - **Tier 0 (0 tokens):** Pure Python daemon heartbeat, staleness, rejection-rate, and drawdown checks.
     - **Tier 1 (fast/cheap model):** Routine hourly check-ins & daily Markdown digests from compact JSON snapshots.
     - **Tier 2 (premium reasoning model):** Escalation-only (kill-switch events, anomalies, weekly strategy review).
     - Enforced via `StateDB.log_llm_usage` & `check_llm_budget` against `config/strategy.yaml` `llm_quota` (`daily_usd_cap`, `monthly_usd_cap`), auto-downgrading to Tier 0 if exhausted.
2. **Independent Adversarial Audit Gates:**
   - At key validation points in Phase 2 and Phase 3, spawn a clean-context, **read-only** `pro` adversarial auditor subagent to attempt to break the implementation.
   - **Blocking rule:** Zero `CRITICAL` or `HIGH` findings may remain unresolved before advancing. Save audit summaries under `docs/audits/`.

---

## 4. Immediate Next Steps on `local-host` (Before Starting Phase 2)

1. **Environment & Secrets Setup (stay strictly local on `local-host` — `.env` is gitignored):**
   ```bash
   cd ~/src/agy-trader
   uv venv .venv
   uv pip install --python .venv/bin/python -e ".[dev]"
   # Optional for real local FinBERT (~2-4 GB RAM on CPU):
   # uv pip install --python .venv/bin/python -e ".[dev,nlp]"
   cp .env.example .env
   # Edit .env with ALPACA_API_KEY, ALPACA_SECRET_KEY, optional FRED_API_KEY
   ```
2. **Verify Local Suite:**
   ```bash
   .venv/bin/ruff check src tests
   .venv/bin/mypy src tests
   .venv/bin/pytest -v
   ```
3. **Run Real Historical Backfill & 4-Way A/B Ablation:**
   ```bash
   .venv/bin/agy backfill --days 730
   # Optional: re-score all backfilled Benzinga news with FinBERT if .[nlp] installed:
   # .venv/bin/agy score-news --backend finbert
   .venv/bin/agy backtest --ablate --walk-forward --folds 3
   ```
4. **Inspect Turnover & Strategy Tuning (Key Insight from Phase 1 Synthetic Dry Run):**
   - Our 365-day synthetic dry run confirmed the `-8%` drawdown kill-switch and stops protected capital (capping drawdown around `-8.5%` to `-9.4%` vs `-14.5%` on synthetic `SPY`), **but** daily rebalancing generated high turnover (`20x–41x`) and significant slippage drag.
   - Once real Alpaca historical + Benzinga news ablation results are in, review with the user whether to tune `rebalance_deadband_pct`, add holding hysteresis / minimum hold days, shift to weekly or 2x/week rebalancing, or disable shorting/news if the A/B ablation shows they don't add net risk-adjusted return after slippage & borrow costs.

---

## 5. Deferred Roadmap (Plan & Execute Only After User Approves)

### Phase 2 — Live Paper Daemons & Broker Execution (on `local-host`)
- **`src/agy_trader/execution/alpaca_broker.py`:** `TradingClient` wrapper enforcing `validate_trading_mode`, startup & per-cycle broker-to-SQLite reconciliation, deterministic `client_order_id` (`{run_id}:{symbol}:{side}`) idempotency, limit orders near NBBO with timeout fallback, mandatory broker-side hard stop-loss sync, and `agy emergency-flatten --confirm`.
- **Native Python Daemons (`src/agy_trader/daemons/`):**
  - `agy-market-data`: Alpaca `StockDataStream` WebSocket + reconnect/REST gap-fill to Parquet + daily macro/earnings refresh + SQLite heartbeat.
  - `agy-news-nlp`: Alpaca `NewsDataStream` Benzinga WebSocket + deduplication + local FinBERT/lexicon scoring to SQLite + heartbeat.
  - `agy-trader`: 30s fast loop (broker sync, stop-loss audit, drawdown kill-switches, `portfolio_snapshots`, heartbeat) + scheduled ET swing rebalance checkpoints (`10:00`, `13:00`, `15:40` ET via Alpaca market clock) running the exact Phase 1 `FeatureEngine -> PortfolioConstructor -> HardRiskEngine -> AlpacaBroker` pipeline.
- **Deployment (`deploy/systemd/*.service`, `deploy/docker-compose.yml`):** Ready-to-enable `systemd` units for `local-host` with `Restart=on-failure`, memory/CPU limits, and journald logging.
- **Recommended Phase 2 Adversarial Audit Gates:**
  - *Gate D:* Order Idempotency, Broker Reconciliation & Live Double-Lock Audit.
  - *Gate E:* Daemon Crash-Recovery, WebSocket Gap-Fill & Stop-Sync Audit.

### Phase 3 — Web Dashboard, Alerts & Active Ops Supervisor
- **Read-Only Local Web UI (`src/agy_trader/dashboard/`):** FastAPI app bound strictly to `127.0.0.1:8080` showing equity curve vs `SPY`, open long/short positions & stops, scored Benzinga news feed, daemon heartbeats, and `control_audit_log`.
- **Webhook Alerter (`src/agy_trader/notifications/alerter.py`):** Pluggable Discord/Slack/Telegram/email alerts with deduplication and severity levels (`INFO`, `WARN`, `CRITICAL`).
- **Active Ops Supervisor (`src/agy_trader/supervisor/runner.py` + `deploy/systemd/agy-supervisor.timer` + `docs/SUPERVISOR_AGENT_RUNBOOK.md`):** Tier 0/1/2 quota-aware scheduled supervisor constrained strictly to `agy ops ...` commands, writing daily digests (`reports/daily/`) and strategy proposals (`reports/proposals/`).
- **Paper-to-Live Graduation Gate (`agy readiness-check`):** Objective readiness check requiring walk-forward OOS Sharpe > `0.8`, Max DD < `15%`, 4–6 weeks unattended paper stability with zero orphan/duplicate orders, and live-paper slippage parity.

---

## 6. Key Repo Paths Quick Reference

- Configs: `config/universe.yaml`, `config/strategy.yaml`, `.env.example`
- Core & Control Plane: `src/agy_trader/core/{models,config,control_plane}.py`
- Storage: `src/agy_trader/storage/{db,parquet_store}.py`
- Data Providers: `src/agy_trader/data/{providers,alpaca_market,alpaca_news,macro_and_calendar,synthetic}.py`
- NLP Sentiment: `src/agy_trader/nlp/sentiment.py`
- Strategy & Risk: `src/agy_trader/strategy/{features,portfolio}.py`, `src/agy_trader/risk/engine.py`
- Backtesting: `src/agy_trader/backtest/{engine,metrics}.py`
- CLI: `src/agy_trader/cli/main.py`
- Tests (55 tests): `tests/test_{storage_and_data,sentiment_and_features,risk_engine,control_plane,metrics_and_portfolio,backtest_engine,cli}.py`
- Audits: `docs/audits/phase1_gate_{a,b,c}.md`
