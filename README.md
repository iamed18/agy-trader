# agy-trader

Deterministic, backtestable quantitative swing-trading system for US equities and ETFs on Alpaca, combining price/volume/macro signals with local quantitative news sentiment (FinBERT / lexicon), immutable hard risk guardrails, and an Active Ops agent control plane.

> **New local agent / session handoff:** See [`docs/HANDOFF.md`](docs/HANDOFF.md) for locked-in design decisions, Phase 1 audit status, quota/audit policies, and the Phase 2–3 roadmap.

## Phase 1 Quickstart

```bash
uv sync
cp .env.example .env

# Run synthetic backfill + 4-way A/B backtest ablation (zero API keys needed)
uv run agy backfill --synthetic --days 730
uv run agy backtest --ablate
```

For real Alpaca Paper historical data and Benzinga news, populate `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env` and run:

```bash
uv run agy backfill --days 365
uv run agy score-news --backend lexicon   # or --backend finbert after: uv sync --extra nlp
uv run agy backtest --ablate
```
