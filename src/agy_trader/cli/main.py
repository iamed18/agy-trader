from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from agy_trader.backtest.engine import BacktestEngine
from agy_trader.backtest.metrics import BacktestMetrics, BacktestResult, render_ablation_markdown
from agy_trader.core.config import (
    AppConfig,
    EnvSettings,
    load_app_config,
    load_universe,
)
from agy_trader.core.control_plane import ControlPlane, ControlPlaneError
from agy_trader.core.models import MacroPoint, NewsArticle
from agy_trader.data.alpaca_market import AlpacaMarketDataProvider
from agy_trader.data.alpaca_news import AlpacaNewsProvider
from agy_trader.data.macro_and_calendar import (
    FRED_SERIES_IDS,
    FREDMacroProvider,
    YFinanceEarningsProvider,
)
from agy_trader.data.synthetic import generate_synthetic_dataset
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
    """Backfill market bars, news, macro series, and earnings calendar."""
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

        if skip_news:
            news_saved = 0
        else:
            news_provider = AlpacaNewsProvider(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
            )
            articles = news_provider.fetch_news(symbols=symbols, start=start_utc, end=now_utc)
            news_saved = db.upsert_news_articles(articles)

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
            "earnings_saved": earnings_saved,
            "macro_saved": macro_saved,
            "news_scored": news_scored,
        }

    if json_output:
        _emit_json(result_payload)
        return

    table = Table(title=f"Backfill Complete ({result_payload['mode']})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for key in ("symbols", "bars_saved", "news_saved", "news_scored", "earnings_saved", "macro_saved"):
        table.add_row(key, str(result_payload[key]))
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
    allow_shorting: Annotated[bool, typer.Option("--allow-shorting/--no-shorting", help="Enable/disable shorting in single run")] = True,
    use_news: Annotated[bool, typer.Option("--use-news/--no-news", help="Enable/disable news signal in single run")] = True,
    start: Annotated[str | None, typer.Option("--start", help="Optional start date (YYYY-MM-DD)")] = None,
    end: Annotated[str | None, typer.Option("--end", help="Optional end date (YYYY-MM-DD)")] = None,
    report_dir: Annotated[Path, typer.Option("--report-dir", help="Directory to write Markdown report")] = Path("reports/backtests"),
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Run event-driven backtest, 4-way ablation, and/or walk-forward validation."""
    _, resolved_data, resolved_config = _resolve_paths(ctx, data_dir, config_dir)

    try:
        start_dt = _parse_iso_date(start, "start")
        end_dt = _parse_iso_date(end, "end")
        if start_dt is not None and end_dt is not None and end_dt < start_dt:
            raise ValueError("end date must be >= start date")
        app_cfg: AppConfig = load_app_config(resolved_config)
        universe = load_universe(resolved_config)
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

    scored_news = db.get_all_scored_news()
    news_by_symbol: dict[str, list[NewsArticle]] = defaultdict(list)
    for art in scored_news:
        for sym in art.symbols:
            news_by_symbol[sym].append(art)

    macro_points: list[MacroPoint] = []
    for series_id in FRED_SERIES_IDS:
        macro_points.extend(db.get_macro_series(series_id))

    earnings_by_symbol = db.get_all_earnings_map()
    engine = BacktestEngine(app_cfg, universe)

    results: list[BacktestResult]
    if ablate:
        results = engine.run_ablation(
            bars_by_symbol,
            news_by_symbol=news_by_symbol,
            macro_points=macro_points,
            earnings_by_symbol=earnings_by_symbol,
            start_date=start_dt,
            end_date=end_dt,
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
                start_date=start_dt,
                end_date=end_dt,
                label=variant_label,
                allow_shorting=allow_shorting,
                use_news=use_news,
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
            n_folds=folds,
            allow_shorting=allow_shorting,
            use_news=use_news,
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
                }
            ),
            metrics_json=res.metrics.model_dump_json(),
        )

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"backtest_{batch_stamp}_{batch_uid}.md"
    report_md = render_ablation_markdown(
        results,
        walk_forward_folds=walk_forward_folds or None,
    )
    report_path.write_text(report_md, encoding="utf-8")

    if json_output:
        _emit_json(
            {
                "results": [res.metrics.model_dump() for res in results],
                "walk_forward_folds": [fm.model_dump() for fm in walk_forward_folds],
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
    table.add_column("Max DD", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", justify="right")
    for res in results:
        m = res.metrics
        table.add_row(
            m.label,
            f"${m.final_equity:,.2f}",
            f"{m.total_return_pct:+.2f}%",
            f"{m.cagr_pct:+.2f}%",
            f"{m.sharpe_ratio:.3f}",
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

    console.print(f"Markdown report saved to [green]{report_path}[/green]")


@app.command("status")
def status_cmd(
    ctx: typer.Context,
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Show database counts, Parquet coverage, control dials, recent runs, and LLM budget."""
    _, resolved_data, resolved_config = _resolve_paths(ctx, data_dir, config_dir)

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

    payload = {
        "data_dir": str(resolved_data),
        "config_dir": str(resolved_config),
        "counts": counts,
        "parquet_symbols_count": len(parquet_symbols),
        "parquet_symbols": parquet_symbols,
        "control_dials": dials.model_dump(mode="json"),
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


if __name__ == "__main__":
    app()
