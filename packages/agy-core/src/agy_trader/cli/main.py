from __future__ import annotations

import json
import sqlite3
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from agy_trader.backtest.engine import BacktestEngine
from agy_trader.backtest.metrics import (
    BacktestMetrics,
    BacktestResult,
    CPCVSummary,
    render_ablation_markdown,
)
from agy_trader.backtest.sp500_cohorts import (
    atomic_write_text,
    fetch_and_cache_sp500_bars,
    filter_active_for_duration,
    generate_sp500_cohort_plots,
    load_or_fetch_sp500_constituents,
    render_sp500_cohort_markdown,
    run_sp500_cohort_study,
    select_point_in_time_candidates,
    validate_research_dir,
)
from agy_trader.backtest.v3_benchmark import (
    run_v3_benchmark,
)
from agy_trader.backtest.v3_universe import (
    V3_CORE_ETFS,
    fetch_and_cache_fred_macro,
    fetch_and_cache_v3_sip_bars,
    load_or_fetch_v3_equity_pool,
)
from agy_trader.core.blackout import (
    NON_EQUITY_ETF_SECTORS,
    load_universe_for_blackout,
    resolve_live_blackout,
)
from agy_trader.core.config import (
    AppConfig,
    EnvSettings,
    load_app_config,
    load_universe,
)
from agy_trader.core.control_plane import ControlPlane, ControlPlaneError
from agy_trader.core.models import AssetType, MacroPoint, NewsArticle
from agy_trader.data.alpaca_market import AlpacaMarketDataProvider
from agy_trader.data.alpaca_news import AlpacaNewsProvider
from agy_trader.data.feeds import collect_official_rss_feeds, collect_yfinance_news
from agy_trader.data.finnhub_news import FinnhubNewsProvider
from agy_trader.data.macro_and_calendar import (
    FRED_SERIES_IDS,
    FREDMacroProvider,
    YFinanceEarningsProvider,
    YFinanceEtfHoldingsProvider,
)
from agy_trader.data.news_filters import is_forward_only_source
from agy_trader.data.sec_edgar import SecEdgarProvider
from agy_trader.data.synthetic import generate_synthetic_dataset
from agy_trader.execution.alpaca_broker import AlpacaBroker
from agy_trader.nlp.sentiment import SentimentScorer, create_sentiment_scorer
from agy_trader.storage.db import StateDB
from agy_trader.storage.parquet_store import ParquetBarStore

app = typer.Typer(
    name="agy",
    help="agy-trader Phase 1 quantitative swing-trading CLI",
    no_args_is_help=True,
)
ops_app = typer.Typer(
    name="ops",
    help="Bounded Active Ops control plane",
    no_args_is_help=True,
)
app.add_typer(ops_app, name="ops")

console = Console()
err_console = Console(stderr=True)


@dataclass(slots=True)
class _CLIContext:
    data_dir: Path | None = None
    config_dir: Path | None = None


def _get_cli_ctx(ctx: typer.Context) -> _CLIContext:
    if isinstance(ctx.obj, _CLIContext):
        return ctx.obj
    return _CLIContext()


def _resolve_paths(
    ctx: typer.Context,
    data_dir: Path | None,
    config_dir: Path | None,
) -> tuple[EnvSettings, Path, Path]:
    settings = EnvSettings()
    cli_ctx = _get_cli_ctx(ctx)
    resolved_data = data_dir or cli_ctx.data_dir or settings.agy_data_dir
    resolved_config = config_dir or cli_ctx.config_dir or settings.agy_config_dir
    return settings, Path(resolved_data), Path(resolved_config)


def _open_stores(data_dir: Path) -> tuple[StateDB, ParquetBarStore]:
    db = StateDB(data_dir / "state.db")
    bar_store = ParquetBarStore(data_dir / "parquet")
    return db, bar_store


def _emit_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


def _fail(message: str, *, json_output: bool = False, code: int = 1) -> None:
    if json_output:
        typer.echo(json.dumps({"ok": False, "error": message}, indent=2))
    else:
        err_console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=code)


def _score_unscored_news(
    db: StateDB,
    scorer: SentimentScorer,
    batch_size: int = 500,
) -> int:
    total_scored = 0
    while True:
        batch = db.get_unscored_news(scorer_model=scorer.model_name, limit=batch_size)
        if not batch:
            break
        preds = scorer.score_articles(batch)
        updates = [
            (art.article_id, pred.score, pred.label, pred.confidence, pred.scorer_model)
            for art, pred in zip(batch, preds, strict=True)
        ]
        total_scored += db.update_news_scores(updates)
    return total_scored


def _parse_iso_date(raw: str | None, label: str) -> date | None:
    if raw is None:
        return None
    try:
        return date.fromisoformat(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid {label} date {raw!r}: expected YYYY-MM-DD") from exc


def _parse_bool_flag(raw: str) -> bool:
    cleaned = raw.strip().lower()
    if cleaned in {"true", "1", "yes", "on"}:
        return True
    if cleaned in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value {raw!r}: expected true or false")


def _parse_backfill_news_sources(raw: str) -> list[str]:
    allowed = {"alpaca", "finnhub"}
    parsed: list[str] = []
    for part in raw.split(","):
        cleaned = part.strip().lower()
        if not cleaned:
            continue
        if cleaned not in allowed:
            raise ValueError(
                f"Invalid news source {cleaned!r}: expected comma-separated subset of 'alpaca,finnhub'"
            )
        if cleaned not in parsed:
            parsed.append(cleaned)
    return parsed


@app.callback()
def main_callback(
    ctx: typer.Context,
    data_dir: Annotated[
        Path | None,
        typer.Option("--data-dir", help="Override data directory (default: EnvSettings.agy_data_dir)"),
    ] = None,
    config_dir: Annotated[
        Path | None,
        typer.Option("--config-dir", help="Override config directory (default: EnvSettings.agy_config_dir)"),
    ] = None,
) -> None:
    ctx.obj = _CLIContext(data_dir=data_dir, config_dir=config_dir)


@app.command("backfill")
def backfill_cmd(
    ctx: typer.Context,
    days: Annotated[int, typer.Option("--days", min=1, help="Number of historical days to backfill")] = 365,
    synthetic: Annotated[bool, typer.Option("--synthetic", help="Generate deterministic synthetic dataset")] = False,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for synthetic generation")] = 42,
    allow_overwrite: Annotated[bool, typer.Option("--allow-overwrite", help="Allow overwriting overlapping Parquet bars")] = False,
    skip_news: Annotated[bool, typer.Option("--skip-news", help="Skip fetching and scoring news articles")] = False,
    news_sources: Annotated[
        str,
        typer.Option("--news-sources", help="Comma-separated news sources to fetch: alpaca,finnhub"),
    ] = "alpaca,finnhub",
    skip_sec: Annotated[bool, typer.Option("--skip-sec", help="Skip fetching SEC EDGAR Form 4 / 8-K filings")] = False,
    score_backend: Annotated[
        str | None,
        typer.Option(
            "--score-backend",
            help="Sentiment scorer backend: lexicon|finbert (default: NLP_BACKEND, or lexicon for --synthetic)",
        ),
    ] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Backfill market bars, news, SEC EDGAR filings, macro series, and earnings calendar."""
    settings, resolved_data, resolved_config = _resolve_paths(ctx, data_dir, config_dir)
    effective_backend = (
        score_backend
        if score_backend is not None
        else ("lexicon" if synthetic else settings.nlp_backend)
    )

    try:
        _ = load_app_config(resolved_config)
        universe = load_universe(resolved_config)
        scorer = create_sentiment_scorer(effective_backend)
        selected_news_sources = _parse_backfill_news_sources(news_sources)
    except (FileNotFoundError, OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        _fail(str(exc), json_output=json_output)
        return

    db, bar_store = _open_stores(resolved_data)

    if synthetic:
        summary = generate_synthetic_dataset(
            universe=universe,
            bar_store=bar_store,
            db=db,
            days=days,
            seed=seed,
        )
        news_scored = _score_unscored_news(db, scorer)
        result_payload: dict[str, Any] = {
            "mode": "synthetic",
            "days": days,
            "seed": seed,
            "score_backend": scorer.model_name,
            **summary,
            "sec_saved": 0,
            "news_scored": news_scored,
        }
    else:
        if not settings.alpaca_api_key.strip() or not settings.alpaca_secret_key.strip():
            _fail(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY are required for live backfill (or pass --synthetic).",
                json_output=json_output,
            )
            return

        now_utc = datetime.now(UTC)
        start_utc = now_utc - timedelta(days=days)
        symbols = [u.symbol for u in universe]

        market_provider = AlpacaMarketDataProvider(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            feed=settings.alpaca_data_feed,
        )
        bars_map = market_provider.fetch_bars(
            symbols=symbols,
            start=start_utc,
            end=now_utc,
            timeframe="1d",
        )
        bars_saved = 0
        for sym, df in bars_map.items():
            if not df.empty:
                bars_saved += bar_store.save_bars(
                    sym,
                    df,
                    timeframe="1d",
                    allow_overwrite=allow_overwrite,
                )

        news_saved = 0
        if not skip_news:
            if "alpaca" in selected_news_sources:
                news_provider = AlpacaNewsProvider(
                    api_key=settings.alpaca_api_key,
                    secret_key=settings.alpaca_secret_key,
                )
                articles = news_provider.fetch_news(symbols=symbols, start=start_utc, end=now_utc)
                news_saved += db.upsert_news_articles(articles)
            if "finnhub" in selected_news_sources and settings.finnhub_api_key.strip():
                finnhub_provider = FinnhubNewsProvider(api_key=settings.finnhub_api_key)
                fh_articles = finnhub_provider.fetch_news(
                    symbols=symbols, start=start_utc, end=now_utc
                )
                news_saved += db.upsert_news_articles(fh_articles)

        macro_saved = 0
        if settings.fred_api_key.strip():
            macro_provider = FREDMacroProvider(api_key=settings.fred_api_key)
            macro_points = macro_provider.fetch_macro(
                start_date=start_utc.date(),
                end_date=now_utc.date(),
            )
            macro_saved = db.upsert_macro(macro_points)

        earnings_provider = YFinanceEarningsProvider()
        earnings_events = earnings_provider.fetch_earnings(symbols)
        earnings_saved = db.upsert_earnings(earnings_events)

        etf_holdings_saved = 0
        etf_entries = [u for u in universe if u.asset_type == AssetType.etf]
        if etf_entries:
            etf_provider = YFinanceEtfHoldingsProvider()
            holdings_by_etf = etf_provider.fetch_etf_holdings([u.symbol for u in etf_entries])
            for etf_u in etf_entries:
                if etf_u.symbol not in holdings_by_etf:
                    continue
                weights = holdings_by_etf[etf_u.symbol]
                is_non_eq = etf_u.sector in NON_EQUITY_ETF_SECTORS
                if is_non_eq:
                    db.upsert_etf_holdings(
                        etf_u.symbol,
                        weights,
                        is_non_equity_etf=True,
                        updated_at=now_utc,
                    )
                    etf_holdings_saved += 1
                elif weights:
                    db.upsert_etf_holdings(
                        etf_u.symbol,
                        weights,
                        is_non_equity_etf=False,
                        updated_at=now_utc,
                    )
                    etf_holdings_saved += 1

        sec_saved = 0
        if not skip_sec and settings.sec_edgar_user_agent.strip():
            sec_provider = SecEdgarProvider(
                user_agent=settings.sec_edgar_user_agent,
                cache_dir=resolved_data / "sec_cache",
            )
            sec_events = sec_provider.fetch_filings(
                symbols=symbols, start=start_utc, end=now_utc
            )
            sec_saved = db.upsert_sec_filings(sec_events)

        try:
            news_scored = 0 if skip_news else _score_unscored_news(db, scorer)
        except RuntimeError as exc:
            _fail(str(exc), json_output=json_output)
            return
        result_payload = {
            "mode": "alpaca",
            "days": days,
            "score_backend": scorer.model_name,
            "symbols": len(symbols),
            "bars_saved": bars_saved,
            "news_saved": news_saved,
            "sec_saved": sec_saved,
            "earnings_saved": earnings_saved,
            "etf_holdings_saved": etf_holdings_saved,
            "macro_saved": macro_saved,
            "news_scored": news_scored,
        }

    if json_output:
        _emit_json(result_payload)
        return

    table = Table(title=f"Backfill Complete ({result_payload['mode']})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for key in (
        "symbols",
        "bars_saved",
        "news_saved",
        "news_scored",
        "sec_saved",
        "earnings_saved",
        "macro_saved",
    ):
        table.add_row(key, str(result_payload[key]))
    if "etf_holdings_saved" in result_payload:
        table.add_row("etf_holdings_saved", str(result_payload["etf_holdings_saved"]))
    console.print(table)


@app.command("collect-feeds")
def collect_feeds_cmd(
    ctx: typer.Context,
    all_symbols: Annotated[
        bool,
        typer.Option("--all-symbols", help="Collect yfinance news for all universe symbols instead of ETFs only"),
    ] = False,
    max_per_symbol: Annotated[int, typer.Option("--max-per-symbol", min=1, help="Max yfinance items per symbol")] = 10,
    skip_rss: Annotated[bool, typer.Option("--skip-rss", help="Skip official Fed/EIA/FDA RSS feeds")] = False,
    skip_yfinance: Annotated[bool, typer.Option("--skip-yfinance", help="Skip yfinance news collection")] = False,
    score: Annotated[bool, typer.Option("--score", help="Score collected/unscored articles")] = False,
    score_backend: Annotated[
        str | None,
        typer.Option("--score-backend", help="Sentiment backend: lexicon|finbert (default: NLP_BACKEND)"),
    ] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Collect forward-only official RSS (Fed/EIA/FDA) and yfinance ETF news into SQLite."""
    settings, resolved_data, resolved_config = _resolve_paths(ctx, data_dir, config_dir)
    effective_backend = score_backend if score_backend is not None else settings.nlp_backend

    try:
        universe = load_universe(resolved_config)
        scorer = create_sentiment_scorer(effective_backend) if score else None
    except (FileNotFoundError, OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        _fail(str(exc), json_output=json_output)
        return

    db, _ = _open_stores(resolved_data)
    target_symbols = (
        [u.symbol for u in universe]
        if all_symbols
        else [u.symbol for u in universe if u.asset_type == AssetType.etf]
    )

    rss_articles = [] if skip_rss else collect_official_rss_feeds()
    yf_articles = (
        []
        if skip_yfinance
        else collect_yfinance_news(target_symbols, max_per_symbol=max_per_symbol)
    )

    rss_saved = db.upsert_news_articles(rss_articles)
    yf_saved = db.upsert_news_articles(yf_articles)

    news_scored = 0
    if scorer is not None:
        try:
            news_scored = _score_unscored_news(db, scorer)
        except RuntimeError as exc:
            _fail(str(exc), json_output=json_output)
            return

    payload = {
        "rss_fetched": len(rss_articles),
        "rss_saved": rss_saved,
        "yfinance_symbols": len(target_symbols) if not skip_yfinance else 0,
        "yfinance_fetched": len(yf_articles),
        "yfinance_saved": yf_saved,
        "total_saved": rss_saved + yf_saved,
        "scored": score,
        "score_backend": scorer.model_name if scorer is not None else None,
        "news_scored": news_scored,
    }
    if json_output:
        _emit_json(payload)
        return

    table = Table(title="Ops Intelligence Feeds Collected")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for key in (
        "rss_fetched",
        "rss_saved",
        "yfinance_symbols",
        "yfinance_fetched",
        "yfinance_saved",
        "total_saved",
        "news_scored",
    ):
        table.add_row(key, str(payload[key]))
    console.print(table)


@app.command("score-news")
def score_news_cmd(
    ctx: typer.Context,
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="Sentiment backend: lexicon|finbert (default: NLP_BACKEND)"),
    ] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1, help="Batch size per DB query")] = 500,
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Score all unscored news articles in SQLite using the chosen NLP backend."""
    settings, resolved_data, _ = _resolve_paths(ctx, data_dir, config_dir)
    effective_backend = backend if backend is not None else settings.nlp_backend
    try:
        scorer = create_sentiment_scorer(effective_backend)
    except ValueError as exc:
        _fail(str(exc), json_output=json_output)
        return

    db, _ = _open_stores(resolved_data)
    try:
        scored_count = _score_unscored_news(db, scorer, batch_size=batch_size)
    except RuntimeError as exc:
        _fail(str(exc), json_output=json_output)
        return
    payload = {
        "backend": effective_backend.strip().lower(),
        "scorer_model": scorer.model_name,
        "scored_count": scored_count,
    }
    if json_output:
        _emit_json(payload)
        return

    console.print(
        f"Scored [bold]{scored_count}[/bold] news article(s) using [cyan]{scorer.model_name}[/cyan]."
    )


@app.command("backtest")
def backtest_cmd(
    ctx: typer.Context,
    ablate: Annotated[bool, typer.Option("--ablate", help="Run 4-way A/B ablation matrix")] = False,
    walk_forward: Annotated[bool, typer.Option("--walk-forward", help="Run walk-forward OOS folds")] = False,
    folds: Annotated[int, typer.Option("--folds", min=1, help="Number of walk-forward folds")] = 3,
    cpcv: Annotated[bool, typer.Option("--cpcv", help="Run Combinatorial Purged Cross-Validation (CPCV)")] = False,
    n_trials: Annotated[int, typer.Option("--n-trials", min=1, help="Assumed number of trials for Deflated Sharpe Ratio (DSR)")] = 1,
    execution_timing: Annotated[
        str | None,
        typer.Option("--execution-timing", help="Override execution timing: same_close|next_open"),
    ] = None,
    allow_shorting: Annotated[bool, typer.Option("--allow-shorting/--no-shorting", help="Enable/disable shorting in single run")] = True,
    use_news: Annotated[bool, typer.Option("--use-news/--no-news", help="Enable/disable news signal in single run")] = True,
    sources: Annotated[
        str | None,
        typer.Option("--sources", help="Optional comma-separated news sources allowlist override"),
    ] = None,
    start: Annotated[str | None, typer.Option("--start", help="Optional start date (YYYY-MM-DD)")] = None,
    end: Annotated[str | None, typer.Option("--end", help="Optional end date (YYYY-MM-DD)")] = None,
    report_dir: Annotated[Path, typer.Option("--report-dir", help="Directory to write Markdown report")] = Path("reports/backtests"),
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Run event-driven backtest, 4-way ablation, walk-forward, and/or CPCV validation."""
    settings, resolved_data, resolved_config = _resolve_paths(ctx, data_dir, config_dir)

    eff_timing: Literal["same_close", "next_open"] | None = None
    try:
        start_dt = _parse_iso_date(start, "start")
        end_dt = _parse_iso_date(end, "end")
        if start_dt is not None and end_dt is not None and end_dt < start_dt:
            raise ValueError("end date must be >= start date")
        if execution_timing is not None:
            cleaned_timing = execution_timing.strip().lower()
            if cleaned_timing not in {"same_close", "next_open"}:
                raise ValueError(
                    f"Invalid execution timing {execution_timing!r}: expected 'same_close' or 'next_open'"
                )
            eff_timing = "same_close" if cleaned_timing == "same_close" else "next_open"
        app_cfg: AppConfig = load_app_config(resolved_config)
        universe = load_universe(resolved_config)
        effective_sources: list[str] | None = app_cfg.strategy.model_news_sources
        if sources is not None:
            parsed_sources = [s.strip().lower() for s in sources.split(",") if s.strip()]
            forward_requested = [s for s in parsed_sources if is_forward_only_source(s)]
            if forward_requested:
                raise ValueError(
                    f"Forward-only sources cannot be used in backtest: {', '.join(forward_requested)}"
                )
            app_cfg = app_cfg.model_copy(
                update={
                    "strategy": app_cfg.strategy.model_copy(
                        update={"model_news_sources": parsed_sources}
                    )
                }
            )
            effective_sources = parsed_sources
    except (FileNotFoundError, OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        _fail(str(exc), json_output=json_output)
        return

    db, bar_store = _open_stores(resolved_data)
    symbols = list(dict.fromkeys([u.symbol for u in universe] + [app_cfg.backtest.benchmark_symbol]))
    bars_by_symbol = bar_store.load_universe_bars(symbols, timeframe="1d")

    if not any(not df.empty for df in bars_by_symbol.values()):
        _fail(
            f"No Parquet bars found in {bar_store.base_dir}. Run `agy backfill` first.",
            json_output=json_output,
        )
        return

    scored_news = db.get_all_scored_news(sources=effective_sources)
    distinct_models = {art.scorer_model for art in scored_news if art.scorer_model}
    if len(distinct_models) > 1:
        preferred_model = "ProsusAI/finbert" if settings.nlp_backend == "finbert" else "lexicon-v1"
        preferred = db.get_all_scored_news(
            scorer_model=preferred_model,
            sources=effective_sources,
        )
        if preferred:
            scored_news = preferred

    news_by_symbol: dict[str, list[NewsArticle]] = defaultdict(list)
    for art in scored_news:
        for sym in art.symbols:
            news_by_symbol[sym].append(art)

    macro_points: list[MacroPoint] = []
    for series_id in FRED_SERIES_IDS:
        macro_points.extend(db.get_macro_series(series_id))

    earnings_by_symbol = db.get_all_earnings_map()
    sec_filings_by_symbol = db.get_sec_filings_map()
    engine = BacktestEngine(app_cfg, universe)

    results: list[BacktestResult]
    if ablate:
        results = engine.run_ablation(
            bars_by_symbol,
            news_by_symbol=news_by_symbol,
            macro_points=macro_points,
            earnings_by_symbol=earnings_by_symbol,
            sec_filings_by_symbol=sec_filings_by_symbol,
            start_date=start_dt,
            end_date=end_dt,
            execution_timing=eff_timing,
            n_trials=n_trials,
        )
    else:
        variant_label = (
            f"{'Price+Vol + News' if use_news else 'Price+Vol Only'} "
            f"({'Long/Short' if allow_shorting else 'Long-Only'})"
        )
        results = [
            engine.run(
                bars_by_symbol,
                news_by_symbol=news_by_symbol,
                macro_points=macro_points,
                earnings_by_symbol=earnings_by_symbol,
                sec_filings_by_symbol=sec_filings_by_symbol,
                start_date=start_dt,
                end_date=end_dt,
                label=variant_label,
                allow_shorting=allow_shorting,
                use_news=use_news,
                execution_timing=eff_timing,
                n_trials=n_trials,
            )
        ]

    if not results or all(res.metrics.trading_days == 0 for res in results):
        _fail(
            "Backtest window produced 0 evaluable trading days (check date range and warm-up history).",
            json_output=json_output,
        )
        return

    walk_forward_folds: list[BacktestMetrics] = []
    if walk_forward:
        walk_forward_folds = engine.run_walk_forward(
            bars_by_symbol,
            news_by_symbol=news_by_symbol,
            macro_points=macro_points,
            earnings_by_symbol=earnings_by_symbol,
            sec_filings_by_symbol=sec_filings_by_symbol,
            n_folds=folds,
            allow_shorting=allow_shorting,
            use_news=use_news,
            execution_timing=eff_timing,
        )

    cpcv_summary: CPCVSummary | None = None
    if cpcv:
        cpcv_summary = engine.run_cpcv(
            bars_by_symbol,
            news_by_symbol=news_by_symbol,
            macro_points=macro_points,
            earnings_by_symbol=earnings_by_symbol,
            sec_filings_by_symbol=sec_filings_by_symbol,
            allow_shorting=allow_shorting,
            use_news=use_news,
            execution_timing=eff_timing,
        )

    batch_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    batch_uid = uuid.uuid4().hex[:6]
    for idx, res in enumerate(results):
        run_id = f"{batch_stamp}-{idx}-{uuid.uuid4().hex[:8]}"
        db.save_backtest_run(
            run_id=run_id,
            label=res.metrics.label,
            start_date=res.metrics.start_date,
            end_date=res.metrics.end_date,
            config_json=json.dumps(
                {
                    **res.config_snapshot,
                    "ablate": ablate,
                    "walk_forward": walk_forward,
                    "folds": folds if walk_forward else 0,
                    "cpcv": cpcv,
                }
            ),
            metrics_json=res.metrics.model_dump_json(),
        )

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"backtest_{batch_stamp}_{batch_uid}.md"
    report_md = render_ablation_markdown(
        results,
        walk_forward_folds=walk_forward_folds or None,
        cpcv_summary=cpcv_summary,
    )
    report_path.write_text(report_md, encoding="utf-8")

    if json_output:
        _emit_json(
            {
                "results": [res.metrics.model_dump() for res in results],
                "walk_forward_folds": [fm.model_dump() for fm in walk_forward_folds],
                "cpcv_summary": cpcv_summary.model_dump() if cpcv_summary is not None else None,
                "report_path": str(report_path),
            }
        )
        return

    table = Table(title="Backtest Results")
    table.add_column("Variant", style="cyan")
    table.add_column("Final Equity", justify="right")
    table.add_column("Total Ret", justify="right")
    table.add_column("CAGR", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Excess Sharpe", justify="right")
    table.add_column("DSR", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", justify="right")
    for res in results:
        m = res.metrics
        dsr_str = f"{m.deflated_sharpe_prob:.3f}" if m.deflated_sharpe_prob is not None else "N/A"
        table.add_row(
            m.label,
            f"${m.final_equity:,.2f}",
            f"{m.total_return_pct:+.2f}%",
            f"{m.cagr_pct:+.2f}%",
            f"{m.sharpe_ratio:.3f}",
            f"{m.excess_sharpe_ratio:.3f}",
            dsr_str,
            f"{m.max_drawdown_pct:.2f}%",
            str(m.closed_trades),
            f"{m.win_rate_pct:.1f}%",
        )
    console.print(table)

    if walk_forward_folds:
        wf_table = Table(title=f"Walk-Forward Out-of-Sample Folds ({len(walk_forward_folds)})")
        wf_table.add_column("Fold", style="cyan")
        wf_table.add_column("Window")
        wf_table.add_column("Total Ret", justify="right")
        wf_table.add_column("Sharpe", justify="right")
        wf_table.add_column("Max DD", justify="right")
        wf_table.add_column("Trades", justify="right")
        for fm in walk_forward_folds:
            wf_table.add_row(
                fm.label,
                f"{fm.start_date}..{fm.end_date}",
                f"{fm.total_return_pct:+.2f}%",
                f"{fm.sharpe_ratio:.3f}",
                f"{fm.max_drawdown_pct:.2f}%",
                str(fm.closed_trades),
            )
        console.print(wf_table)

    if cpcv_summary is not None:
        cpcv_table = Table(
            title=f"CPCV Summary ({cpcv_summary.n_paths} paths, N={cpcv_summary.n_groups}, k={cpcv_summary.k_test_groups})"
        )
        cpcv_table.add_column("Metric", style="cyan")
        cpcv_table.add_column("Value", justify="right")
        cpcv_table.add_row("Median Sharpe", f"{cpcv_summary.median_sharpe:.3f}")
        cpcv_table.add_row("Mean Sharpe", f"{cpcv_summary.mean_sharpe:.3f}")
        cpcv_table.add_row("IQR (P25..P75)", f"{cpcv_summary.p25_sharpe:.3f} .. {cpcv_summary.p75_sharpe:.3f}")
        cpcv_table.add_row("Min..Max Sharpe", f"{cpcv_summary.min_sharpe:.3f} .. {cpcv_summary.max_sharpe:.3f}")
        cpcv_table.add_row("Positive Paths", f"{cpcv_summary.pct_positive_paths:.1f}%")
        cpcv_table.add_row("Median Excess Sharpe", f"{cpcv_summary.median_excess_sharpe:.3f}")
        console.print(cpcv_table)

    console.print(f"Markdown report saved to [green]{report_path}[/green]")


@app.command("sp500-cohorts")
def sp500_cohorts_cmd(
    ctx: typer.Context,
    n_cohorts: Annotated[int, typer.Option("--n-cohorts", min=1, help="Number of cohorts to sample")] = 20,
    cohort_size: Annotated[int, typer.Option("--cohort-size", min=1, help="Symbols per cohort")] = 50,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic sampling RNG seed")] = 20261122,
    cutoff_date: Annotated[str, typer.Option("--cutoff-date", help="Point-in-time constituent cutoff (YYYY-MM-DD)")] = "2021-11-23",
    research_dir: Annotated[Path, typer.Option("--research-dir", help="Isolated Parquet/constituent cache directory")] = Path("data/parquet_sp500_cohorts"),
    output_dir: Annotated[Path, typer.Option("--output-dir", help="Output directory for summary JSON/MD and plots")] = Path("reports/sp500_cohorts"),
    refresh_bars: Annotated[bool, typer.Option("--refresh-bars", help="Re-fetch bars from Alpaca even if cached")] = False,
    execution_timing: Annotated[
        str | None,
        typer.Option("--execution-timing", help="Override execution timing: same_close|next_open"),
    ] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory for read-only macro_snapshots")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Run S&P 500 multi-cohort backtest study with isolated research cache and plots."""
    settings, resolved_data, resolved_config = _resolve_paths(ctx, data_dir, config_dir)

    eff_timing: Literal["same_close", "next_open"] | None = None
    try:
        cutoff_dt = _parse_iso_date(cutoff_date, "cutoff-date")
        if cutoff_dt is None:
            raise ValueError("cutoff-date is required")
        if execution_timing is not None:
            cleaned_timing = execution_timing.strip().lower()
            if cleaned_timing not in {"same_close", "next_open"}:
                raise ValueError(
                    f"Invalid execution timing {execution_timing!r}: expected 'same_close' or 'next_open'"
                )
            eff_timing = "same_close" if cleaned_timing == "same_close" else "next_open"
        app_cfg: AppConfig = load_app_config(resolved_config)
        validate_research_dir(research_dir, data_dir=resolved_data)
    except (FileNotFoundError, OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        _fail(str(exc), json_output=json_output)
        return

    try:
        macro_points: list[MacroPoint] = []
        state_db_path = resolved_data / "state.db"
        if state_db_path.exists():
            ro_db = StateDB(state_db_path, read_only=True)
            try:
                for series_id in FRED_SERIES_IDS:
                    macro_points.extend(ro_db.get_macro_series(series_id))
            except sqlite3.OperationalError as db_exc:
                if "no such table" in str(db_exc).lower():
                    macro_points = []
                else:
                    raise

        constituents_path = research_dir / "sp500_constituents.json"
        all_constituents = load_or_fetch_sp500_constituents(constituents_path, refresh=refresh_bars)
        candidates, pre_cutoff_count = select_point_in_time_candidates(
            all_constituents,
            cutoff_date=cutoff_dt,
        )

        start_utc = datetime(cutoff_dt.year, cutoff_dt.month, cutoff_dt.day, 0, 0, tzinfo=UTC)
        end_utc = datetime.now(UTC)
        bars_by_symbol = fetch_and_cache_sp500_bars(
            [c.symbol for c in candidates],
            research_dir,
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            feed=settings.alpaca_data_feed,
            start_utc=start_utc,
            end_utc=end_utc,
            benchmark_symbol=app_cfg.backtest.benchmark_symbol,
            refresh=refresh_bars,
        )

        eligible = filter_active_for_duration(
            bars_by_symbol,
            candidates,
            benchmark_symbol=app_cfg.backtest.benchmark_symbol,
        )
        report, ts_bundle = run_sp500_cohort_study(
            app_config=app_cfg,
            constituents_total_count=len(all_constituents),
            pre_cutoff_count=pre_cutoff_count,
            deduped_count=len(candidates),
            eligible=eligible,
            bars_by_symbol=bars_by_symbol,
            macro_points=macro_points,
            n_cohorts=n_cohorts,
            cohort_size=cohort_size,
            seed=seed,
            cutoff_date=cutoff_dt,
            execution_timing=eff_timing,
        )
        plot_paths = generate_sp500_cohort_plots(report, ts_bundle, output_dir)
        report = report.model_copy(update={"plot_paths": [str(p) for p in plot_paths]})

        output_dir.mkdir(parents=True, exist_ok=True)
        summary_json_path = output_dir / "summary.json"
        summary_md_path = output_dir / "summary.md"
        manifest_path = output_dir / "cohorts_manifest.json"

        atomic_write_text(summary_json_path, report.model_dump_json(indent=2))
        atomic_write_text(summary_md_path, render_sp500_cohort_markdown(report))
        manifest_payload = {
            "seed": report.seed,
            "cutoff_date": report.cutoff_date,
            "n_cohorts": report.n_cohorts,
            "cohort_size": report.cohort_size,
            "active_eligible_pool_size": report.active_eligible_pool_size,
            "unique_symbols_across_cohorts": report.unique_symbols_across_cohorts,
            "cohorts": [
                {
                    "cohort_id": c.cohort_id,
                    "label": c.label,
                    "symbols": c.symbols,
                    "sector_counts": c.sector_counts,
                }
                for c in report.cohorts
            ],
        }
        atomic_write_text(manifest_path, json.dumps(manifest_payload, indent=2))
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        ValidationError,
        httpx.HTTPError,
        json.JSONDecodeError,
        sqlite3.Error,
    ) as exc:
        _fail(str(exc), json_output=json_output)
        return

    if json_output:
        _emit_json(
            {
                "summary_json": str(summary_json_path),
                "summary_md": str(summary_md_path),
                "cohorts_manifest": str(manifest_path),
                "plot_paths": report.plot_paths,
                "n_cohorts": report.n_cohorts,
                "cohort_size": report.cohort_size,
                "active_eligible_pool_size": report.active_eligible_pool_size,
                "strategy_median_cagr_pct": report.strategy_stats.cagr_pct.median,
                "strategy_median_sharpe": report.strategy_stats.sharpe_ratio.median,
            }
        )
        return

    table = Table(
        title=f"S&P 500 {report.n_cohorts}-Cohort Summary ({report.start_date} .. {report.end_date})"
    )
    table.add_column("Series", style="cyan")
    table.add_column("Median CAGR", justify="right")
    table.add_column("Median Ret", justify="right")
    table.add_column("Median Sharpe", justify="right")
    table.add_column("Median Ex-Sharpe", justify="right")
    table.add_column("Median Max DD", justify="right")
    for name, agg in (
        ("Strategy", report.strategy_stats),
        (f"EW-{report.cohort_size} Daily", report.ew_cohort_stats),
        (f"EW-{report.cohort_size} B&H", report.ew_buy_hold_stats),
        ("SPY Buy & Hold", report.spy_stats),
    ):
        table.add_row(
            name,
            f"{agg.cagr_pct.median:+.2f}%",
            f"{agg.total_return_pct.median:+.2f}%",
            f"{agg.sharpe_ratio.median:.3f}",
            f"{agg.excess_sharpe_ratio.median:.3f}",
            f"{agg.max_drawdown_pct.median:.2f}%",
        )
    console.print(table)
    console.print(
        f"Outputs saved to [green]{output_dir}[/green] "
        f"(summary.json, summary.md, cohorts_manifest.json, {len(plot_paths)} plots)"
    )


@app.command("v3-research")
def v3_research_cmd(
    ctx: typer.Context,
    research_dir: Annotated[
        Path,
        typer.Option("--research-dir", help="Isolated research Parquet and metadata cache directory"),
    ] = Path("data/parquet_v3_research"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Output directory for reports, summary JSON/MD, and plots"),
    ] = Path("reports/v3_research"),
    data_dir: Annotated[
        Path | None,
        typer.Option("--data-dir", help="Override data directory for safety checks"),
    ] = None,
    config_dir: Annotated[
        Path | None,
        typer.Option("--config-dir", help="Override config directory"),
    ] = None,
    top_n: Annotated[
        int,
        typer.Option("--top-n", min=10, help="Monthly point-in-time universe size"),
    ] = 200,
    hysteresis: Annotated[
        int,
        typer.Option("--hysteresis", min=0, help="Turnover buffer for monthly universe ranking"),
    ] = 50,
    refresh_data: Annotated[
        bool,
        typer.Option("--refresh-data", help="Re-fetch constituents, macro, and bars from providers"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON output"),
    ] = False,
) -> None:
    """Run V3 research 11-year data acquisition, progressive 6-track benchmarks, stress checks, and plots."""
    settings, resolved_data, resolved_config = _resolve_paths(ctx, data_dir, config_dir)

    try:
        app_cfg: AppConfig = load_app_config(resolved_config)
        validate_research_dir(research_dir, data_dir=resolved_data)
    except (FileNotFoundError, OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        _fail(str(exc), json_output=json_output)
        return

    try:
        pool = load_or_fetch_v3_equity_pool(research_dir, refresh=refresh_data)
        macro_points = fetch_and_cache_fred_macro(
            research_dir, fred_api_key=settings.fred_api_key, refresh=refresh_data
        )

        pool_symbols = [c.symbol for c in pool]
        all_symbols = list(dict.fromkeys(pool_symbols + list(V3_CORE_ETFS) + ["SPY"]))

        bars_by_symbol = fetch_and_cache_v3_sip_bars(
            all_symbols,
            research_dir,
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            refresh=refresh_data,
        )

        report, _ = run_v3_benchmark(
            bars_by_symbol,
            pool,
            app_config=app_cfg,
            macro_points=macro_points,
            top_n=top_n,
            hysteresis=hysteresis,
            output_dir=output_dir,
        )
        summary_json_path = output_dir / "summary.json"
        summary_md_path = output_dir / "summary.md"
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        ValidationError,
        httpx.HTTPError,
        json.JSONDecodeError,
        sqlite3.Error,
    ) as exc:
        _fail(str(exc), json_output=json_output)
        return

    if json_output:
        _emit_json(
            {
                "summary_json": str(summary_json_path),
                "summary_md": str(summary_md_path),
                "plot_paths": report.plot_paths,
                "is_selected_track_id": report.is_selected_track_id,
                "is_selected_reason": report.is_selected_reason,
                "pool_total": report.pool_total,
                "delisted_count": report.delisted_count,
                "bars_loaded": report.bars_loaded,
                "tracks_count": len(report.tracks),
            }
        )
        return

    table = Table(
        title=f"V3 Benchmark Progressive Ladder: IS ({report.is_window[0]}..{report.is_window[1]}) vs OOS ({report.oos_window[0]}..{report.oos_window[1]})"
    )
    table.add_column("Track", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("IS CAGR", justify="right")
    table.add_column("IS XS-SR", justify="right")
    table.add_column("IS MaxDD", justify="right")
    table.add_column("OOS CAGR", justify="right")
    table.add_column("OOS XS-SR", justify="right")
    table.add_column("OOS MaxDD", justify="right")
    table.add_column("Full CAGR", justify="right")
    table.add_column("Full XS-SR", justify="right")

    for t in report.tracks:
        star = " ★" if t.track_id == report.is_selected_track_id else ""
        table.add_row(
            f"{t.track_id}{star}",
            t.name,
            f"{t.is_metrics.cagr_pct:+.1f}%",
            f"{t.is_metrics.excess_sharpe_ratio:.2f}",
            f"{t.is_metrics.max_drawdown_pct:.1f}%",
            f"{t.oos_metrics.cagr_pct:+.1f}%",
            f"{t.oos_metrics.excess_sharpe_ratio:.2f}",
            f"{t.oos_metrics.max_drawdown_pct:.1f}%",
            f"{t.full_metrics.cagr_pct:+.1f}%",
            f"{t.full_metrics.excess_sharpe_ratio:.2f}",
        )
    console.print(table)
    console.print(
        f"Anti-peeking IS Winner: [bold green]{report.is_selected_track_id}[/bold green] "
        f"({report.is_selected_reason})"
    )
    console.print(
        f"Outputs saved to [green]{output_dir}[/green] "
        f"(summary.json, summary.md, {len(report.plot_paths)} plots)"
    )


@app.command("status")
def status_cmd(
    ctx: typer.Context,
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Show database counts, Parquet coverage, control dials, recent runs, and LLM budget."""
    settings, resolved_data, resolved_config = _resolve_paths(ctx, data_dir, config_dir)

    try:
        app_cfg = load_app_config(resolved_config)
    except (FileNotFoundError, OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        _fail(str(exc), json_output=json_output)
        return

    db, bar_store = _open_stores(resolved_data)
    cp = ControlPlane(db)
    counts = db.get_summary_counts()
    parquet_symbols = bar_store.list_symbols(timeframe="1d")
    dials = cp.get_dials()
    recent_runs = db.get_recent_backtest_runs(limit=5)
    budget_allowed, budget_details = cp.check_llm_budget(app_cfg.llm_quota)
    universe, universe_error = load_universe_for_blackout(resolved_config)
    env_path = Path(".env")
    blackout = resolve_live_blackout(
        env_file=env_path if env_path.exists() else None,
        require_env_file=env_path.exists(),
        fallback_raw=settings.blackout_period,
        db=db,
        universe=universe,
        universe_error=universe_error,
        observed_broker_symbols=[p["symbol"] for p in db.get_live_positions()],
    )

    payload = {
        "data_dir": str(resolved_data),
        "config_dir": str(resolved_config),
        "counts": counts,
        "parquet_symbols_count": len(parquet_symbols),
        "parquet_symbols": parquet_symbols,
        "control_dials": dials.model_dump(mode="json"),
        "blackout": blackout.to_status_dict(),
        "recent_backtest_runs": recent_runs,
        "llm_budget": {
            "allowed": budget_allowed,
            **budget_details,
        },
    }

    if json_output:
        _emit_json(payload)
        return

    counts_table = Table(title="agy-trader Status")
    counts_table.add_column("Store / Metric", style="cyan")
    counts_table.add_column("Value", justify="right")
    counts_table.add_row("parquet_symbols (1d)", str(len(parquet_symbols)))
    for k, v in counts.items():
        counts_table.add_row(k, str(v))
    counts_table.add_row("pause_new_entries", str(dials.pause_new_entries))
    counts_table.add_row("exposure_multiplier", f"{dials.exposure_multiplier:.2f}")
    counts_table.add_row("allow_shorting", str(dials.allow_shorting))
    counts_table.add_row("active_blacklist", str(len(dials.active_blacklist(datetime.now(UTC)))))
    counts_table.add_row(
        "blackout_period",
        blackout.raw_blackout_period or "none",
    )
    counts_table.add_row("blackout_halted", str(blackout.halted))
    counts_table.add_row(
        "blackout_frozen_symbols",
        ", ".join(sorted(blackout.frozen_symbols)) or "none",
    )
    if blackout.etf_weights_summary:
        counts_table.add_row(
            "blackout_etf_weights",
            "; ".join(blackout.etf_weights_summary),
        )
    counts_table.add_row(
        "llm_budget_daily",
        f"${budget_details['daily_spend']:.4f} / ${budget_details['daily_cap']:.2f} ({'OK' if budget_allowed else 'EXCEEDED'})",
    )
    console.print(counts_table)

    if recent_runs:
        runs_table = Table(title="Recent Backtest Runs")
        runs_table.add_column("Run ID", style="dim")
        runs_table.add_column("Label", style="cyan")
        runs_table.add_column("Window")
        runs_table.add_column("Sharpe", justify="right")
        runs_table.add_column("Total Ret", justify="right")
        for run in recent_runs:
            m = run.get("metrics", {})
            runs_table.add_row(
                run["run_id"],
                run["label"],
                f"{run['start_date']}..{run['end_date']}",
                f"{float(m.get('sharpe_ratio', 0.0)):.3f}",
                f"{float(m.get('total_return_pct', 0.0)):+.2f}%",
            )
        console.print(runs_table)


def _ops_control_plane(
    ctx: typer.Context,
    data_dir: Path | None,
    config_dir: Path | None,
) -> tuple[StateDB, ControlPlane]:
    _, resolved_data, _ = _resolve_paths(ctx, data_dir, config_dir)
    db, _ = _open_stores(resolved_data)
    return db, ControlPlane(db)


def _emit_dials(dials: Any, *, title: str, json_output: bool) -> None:
    payload = dials.model_dump(mode="json")
    if json_output:
        _emit_json(payload)
        return
    table = Table(title=title)
    table.add_column("Dial", style="cyan")
    table.add_column("Value")
    table.add_row("pause_new_entries", str(dials.pause_new_entries))
    table.add_row("exposure_multiplier", f"{dials.exposure_multiplier:.4f}")
    table.add_row("allow_shorting", str(dials.allow_shorting))
    table.add_row(
        "blacklisted_symbols",
        ", ".join(f"{k} (until {v.isoformat()})" for k, v in sorted(dials.blacklisted_symbols.items()))
        or "none",
    )
    table.add_row("updated_by", dials.updated_by)
    table.add_row("reason", dials.reason)
    table.add_row("updated_at", dials.updated_at.isoformat())
    console.print(table)


@ops_app.command("show")
def ops_show(
    ctx: typer.Context,
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Show current Active Ops control dials."""
    _, cp = _ops_control_plane(ctx, data_dir, config_dir)
    dials = cp.get_dials()
    _emit_dials(dials, title="Active Ops Control Dials", json_output=json_output)


@ops_app.command("pause")
def ops_pause(
    ctx: typer.Context,
    reason: Annotated[str, typer.Option("--reason", help="Audit reason for pausing new entries")],
    actor: Annotated[str, typer.Option("--actor", help="Actor identifier")] = "ops",
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Pause all new position entries (risk-reducing exits remain allowed)."""
    _, cp = _ops_control_plane(ctx, data_dir, config_dir)
    try:
        dials = cp.set_pause(True, actor=actor, reason=reason)
    except (ControlPlaneError, ValidationError, ValueError) as exc:
        _fail(str(exc), json_output=json_output)
        return
    _emit_dials(dials, title="Entries Paused", json_output=json_output)


@ops_app.command("resume")
def ops_resume(
    ctx: typer.Context,
    reason: Annotated[str, typer.Option("--reason", help="Audit reason for resuming new entries")],
    actor: Annotated[str, typer.Option("--actor", help="Actor identifier")] = "ops",
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Resume new position entries."""
    _, cp = _ops_control_plane(ctx, data_dir, config_dir)
    try:
        dials = cp.set_pause(False, actor=actor, reason=reason)
    except (ControlPlaneError, ValidationError, ValueError) as exc:
        _fail(str(exc), json_output=json_output)
        return
    _emit_dials(dials, title="Entries Resumed", json_output=json_output)


@ops_app.command("set-exposure")
def ops_set_exposure(
    ctx: typer.Context,
    multiplier: Annotated[float, typer.Argument(help="Exposure multiplier in [0.0, 1.0]")],
    reason: Annotated[str, typer.Option("--reason", help="Audit reason for exposure update")],
    actor: Annotated[str, typer.Option("--actor", help="Actor identifier")] = "ops",
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Set bounded exposure_multiplier in [0.0, 1.0]."""
    _, cp = _ops_control_plane(ctx, data_dir, config_dir)
    try:
        dials = cp.set_exposure(multiplier, actor=actor, reason=reason)
    except (ControlPlaneError, ValidationError, ValueError) as exc:
        _fail(str(exc), json_output=json_output)
        return
    _emit_dials(dials, title="Exposure Multiplier Updated", json_output=json_output)


@ops_app.command("set-shorting")
def ops_set_shorting(
    ctx: typer.Context,
    enabled: Annotated[str, typer.Argument(help="Enable or disable shorting: true|false")],
    reason: Annotated[str, typer.Option("--reason", help="Audit reason for shorting toggle")],
    actor: Annotated[str, typer.Option("--actor", help="Actor identifier")] = "ops",
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Enable or disable shorting via ControlDials.allow_shorting."""
    _, cp = _ops_control_plane(ctx, data_dir, config_dir)
    try:
        flag = _parse_bool_flag(enabled)
        dials = cp.set_shorting(flag, actor=actor, reason=reason)
    except (ControlPlaneError, ValidationError, ValueError) as exc:
        _fail(str(exc), json_output=json_output)
        return
    _emit_dials(dials, title="Shorting Setting Updated", json_output=json_output)


blacklist_app = typer.Typer(
    name="blacklist",
    help="Manage temporary symbol blacklist entries",
    no_args_is_help=True,
)
ops_app.add_typer(blacklist_app, name="blacklist")


@blacklist_app.command("add")
def ops_blacklist_add(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Argument(help="Ticker symbol to blacklist")],
    reason: Annotated[str, typer.Option("--reason", help="Audit reason for blacklisting")],
    hours: Annotated[float, typer.Option("--hours", help="Blacklist duration in hours (0, 720]")] = 48.0,
    actor: Annotated[str, typer.Option("--actor", help="Actor identifier")] = "ops",
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Temporarily blacklist a symbol from new entries."""
    _, cp = _ops_control_plane(ctx, data_dir, config_dir)
    try:
        dials = cp.add_blacklist(symbol, hours=hours, actor=actor, reason=reason)
    except (ControlPlaneError, ValidationError, ValueError) as exc:
        _fail(str(exc), json_output=json_output)
        return
    _emit_dials(dials, title=f"Blacklisted {symbol.strip().upper()}", json_output=json_output)


@blacklist_app.command("remove")
def ops_blacklist_remove(
    ctx: typer.Context,
    symbol: Annotated[str, typer.Argument(help="Ticker symbol to remove from blacklist")],
    reason: Annotated[str, typer.Option("--reason", help="Audit reason for removing blacklist")],
    actor: Annotated[str, typer.Option("--actor", help="Actor identifier")] = "ops",
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Remove a symbol from the Active Ops blacklist."""
    _, cp = _ops_control_plane(ctx, data_dir, config_dir)
    try:
        dials = cp.remove_blacklist(symbol, actor=actor, reason=reason)
    except (ControlPlaneError, ValidationError, ValueError) as exc:
        _fail(str(exc), json_output=json_output)
        return
    _emit_dials(dials, title=f"Removed {symbol.strip().upper()} from Blacklist", json_output=json_output)


@ops_app.command("audit-log")
def ops_audit_log(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Max audit entries to return")] = 20,
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Show recent Active Ops control plane audit log entries."""
    db, _ = _ops_control_plane(ctx, data_dir, config_dir)
    entries = db.get_control_audit_log(limit=limit)
    if json_output:
        _emit_json({"entries": entries})
        return

    table = Table(title=f"Control Audit Log (latest {len(entries)})")
    table.add_column("ID", justify="right")
    table.add_column("Changed At")
    table.add_column("Actor", style="cyan")
    table.add_column("Reason")
    for entry in entries:
        table.add_row(
            str(entry["id"]),
            str(entry["changed_at"]),
            str(entry["actor"]),
            str(entry["reason"]),
        )
    console.print(table)


@ops_app.command("authorize-restart")
def ops_authorize_restart(
    ctx: typer.Context,
    service: Annotated[str, typer.Argument(help="Allowlisted service name (market-data|news-nlp|trader)")],
    reason: Annotated[str, typer.Option("--reason", help="Audit reason for service restart")],
    actor: Annotated[str, typer.Option("--actor", help="Actor identifier")] = "ops",
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Authorize a rate-limited restart for an allowlisted service."""
    _, cp = _ops_control_plane(ctx, data_dir, config_dir)
    try:
        cp.authorize_service_restart(service, actor=actor, reason=reason)
    except (ControlPlaneError, ValidationError, ValueError) as exc:
        _fail(str(exc), json_output=json_output)
        return

    payload = {
        "authorized": True,
        "service": service.strip().lower(),
        "actor": actor,
        "reason": reason.strip(),
    }
    if json_output:
        _emit_json(payload)
        return

    console.print(
        f"Restart [green]authorized[/green] for [cyan]{payload['service']}[/cyan] "
        f"(actor={actor}, reason={payload['reason']})"
    )


@app.command("emergency-flatten")
def emergency_flatten_cmd(
    ctx: typer.Context,
    confirm: Annotated[
        bool,
        typer.Option("--confirm", help="Explicit confirmation flag required to flatten all positions"),
    ] = False,
    reason: Annotated[
        str,
        typer.Option("--reason", help="Audit reason for emergency flatten"),
    ] = "",
    actor: Annotated[str, typer.Option("--actor", help="Actor identifier")] = "operator",
    override_blackout: Annotated[
        bool,
        typer.Option(
            "--override-blackout",
            help="Override BLACKOUT_PERIOD freeze for legal/compliance-cleared emergencies",
        ),
    ] = False,
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Pause new entries, cancel all open orders, and market-close all open Alpaca positions."""
    if not confirm:
        _fail("Refusing emergency flatten without --confirm.", json_output=json_output)
        return
    if not reason.strip():
        _fail("A non-empty --reason is required for emergency flatten.", json_output=json_output)
        return

    settings, resolved_data, resolved_config = _resolve_paths(ctx, data_dir, config_dir)
    db, _ = _open_stores(resolved_data)
    try:
        with AlpacaBroker(
            settings,
            db,
            env_file=Path(".env") if Path(".env").exists() else None,
            config_dir=resolved_config,
        ) as broker:
            result = broker.emergency_flatten(
                confirm=confirm,
                reason=reason,
                actor=actor,
                override_blackout=override_blackout,
            )
    except (ControlPlaneError, ValidationError, ValueError, RuntimeError, OSError) as exc:
        _fail(str(exc), json_output=json_output)
        return
    except Exception as exc:  # noqa: BLE001 - surface Alpaca SDK/network errors cleanly in CLI
        _fail(str(exc), json_output=json_output)
        return

    if json_output:
        _emit_json(result)
        return

    table = Table(title="Emergency Flatten Executed")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("ok", str(result["ok"]))
    table.add_row("paper_mode", str(result["paper_mode"]))
    table.add_row("paused_new_entries", str(result["paused_new_entries"]))
    table.add_row("override_blackout", str(result.get("override_blackout", False)))
    table.add_row("canceled_orders", str(result["canceled_orders"]))
    table.add_row("close_orders_submitted", str(len(result["close_orders"])))
    table.add_row(
        "skipped_blackout_symbols",
        ", ".join(result.get("skipped_blackout_symbols", [])) or "none",
    )
    table.add_row("errors", str(len(result["errors"])))
    table.add_row("reason", str(result["reason"]))
    console.print(table)
    if not result["ok"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
