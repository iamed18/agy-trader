from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from rich.console import Console
from rich.table import Table

from agy_ops.daemons import (
    MarketDataDaemon,
    NewsNlpDaemon,
    TraderDaemon,
    configure_daemon_logging,
    get_process_rss_mb,
)
from agy_ops.dashboard.app import create_dashboard_app, serve_dashboard, validate_dashboard_host
from agy_ops.deploy.service_manifests import write_service_manifests
from agy_ops.reporting.digests import (
    build_daily_summary,
    build_morning_plan,
    build_readiness_report,
)
from agy_trader.core.config import EnvSettings
from agy_trader.storage.db import StateDB

app = typer.Typer(
    name="agy-ops",
    help="Always-on daemons, read-only dashboard, and operational reporting CLI",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


@dataclass(slots=True)
class _OpsContext:
    data_dir: Path | None = None
    config_dir: Path | None = None


def _resolve_paths(
    ctx: typer.Context,
    data_dir: Path | None,
    config_dir: Path | None,
) -> tuple[EnvSettings, Path, Path]:
    settings = EnvSettings()
    cli_ctx = ctx.obj if isinstance(ctx.obj, _OpsContext) else _OpsContext()
    resolved_data = Path(data_dir or cli_ctx.data_dir or settings.agy_data_dir)
    resolved_config = Path(config_dir or cli_ctx.config_dir or settings.agy_config_dir)
    return settings, resolved_data, resolved_config


def _emit_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.callback()
def main_callback(
    ctx: typer.Context,
    data_dir: Annotated[
        Path | None,
        typer.Option("--data-dir", help="Override data directory"),
    ] = None,
    config_dir: Annotated[
        Path | None,
        typer.Option("--config-dir", help="Override config directory"),
    ] = None,
) -> None:
    ctx.obj = _OpsContext(data_dir=data_dir, config_dir=config_dir)


@app.command("version")
def version_cmd() -> None:
    """Print agy-ops package version."""
    from agy_ops import __version__

    typer.echo(f"agy-ops {__version__}")


@app.command("readiness")
def readiness_cmd(
    ctx: typer.Context,
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Run deterministic pre-market and daemon readiness checks."""
    _, resolved_data, resolved_config = _resolve_paths(ctx, data_dir, config_dir)
    report = build_readiness_report(resolved_data, resolved_config)
    if json_output:
        _emit_json(report)
        return

    table = Table(title=f"agy-ops Readiness ({report['status']})")
    table.add_column("Check", style="cyan")
    table.add_column("Severity")
    table.add_column("Message")
    for chk in report["checks"]:
        sev = chk["severity"]
        style = "green" if sev == "ok" else ("yellow" if sev == "degraded" else "red")
        table.add_row(chk["name"], f"[{style}]{sev}[/{style}]", chk["message"])
    console.print(table)


@app.command("morning-plan")
def morning_plan_cmd(
    ctx: typer.Context,
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Generate pre-market risk, regime, blackout, and overnight news plan."""
    _, resolved_data, resolved_config = _resolve_paths(ctx, data_dir, config_dir)
    report = build_morning_plan(resolved_data, resolved_config)
    if json_output:
        _emit_json(report)
        return
    typer.echo(report["markdown_summary"])


@app.command("daily-summary")
def daily_summary_cmd(
    ctx: typer.Context,
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Generate post-close performance, fills/slippage, and drawdown summary."""
    _, resolved_data, resolved_config = _resolve_paths(ctx, data_dir, config_dir)
    report = build_daily_summary(resolved_data, resolved_config)
    if json_output:
        _emit_json(report)
        return
    typer.echo(report["markdown_summary"])


def _start_dashboard_heartbeat(
    data_dir: Path,
    *,
    host: str,
    port: int,
    stop_event: threading.Event,
) -> threading.Thread:
    def _loop() -> None:
        db = StateDB(data_dir / "state.db")
        while not stop_event.is_set():
            try:
                rss_mb = get_process_rss_mb()
                db.upsert_daemon_heartbeat(
                    "dashboard",
                    "healthy",
                    {
                        "pid": os.getpid(),
                        "rss_mb": rss_mb,
                        "peak_rss_mb": rss_mb,
                        "host": host,
                        "port": port,
                    },
                    now_utc=datetime.now(UTC),
                )
            except Exception as exc:  # noqa: BLE001
                err_console.print(f"[yellow]dashboard heartbeat warning:[/yellow] {exc}")
            if stop_event.wait(timeout=60.0):
                break

    thread = threading.Thread(target=_loop, name="dashboard-heartbeat", daemon=True)
    thread.start()
    return thread


@app.command("daemon")
def daemon_cmd(
    ctx: typer.Context,
    service: Annotated[
        str,
        typer.Argument(help="Daemon service: market-data|news-nlp|trader|dashboard"),
    ],
    once: Annotated[bool, typer.Option("--once", help="Run a single deterministic cycle and exit")] = False,
    host: Annotated[str, typer.Option("--host", help="Dashboard bind host (localhost only)")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535, help="Dashboard port")] = 8080,
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
) -> None:
    """Run one of the four always-on agy-ops services."""
    svc = service.strip().lower()
    if svc not in {"market-data", "news-nlp", "trader", "dashboard"}:
        err_console.print(
            f"[red]Error:[/red] Invalid service {service!r}; expected market-data|news-nlp|trader|dashboard"
        )
        raise typer.Exit(code=1)

    settings, resolved_data, resolved_config = _resolve_paths(ctx, data_dir, config_dir)
    if not once:
        configure_daemon_logging(resolved_data, svc)

    if svc == "dashboard":
        try:
            safe_host = validate_dashboard_host(host)
        except ValueError as exc:
            err_console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        if once:
            _ = create_dashboard_app(resolved_data, resolved_config)
            db = StateDB(resolved_data / "state.db")
            rss_mb = get_process_rss_mb()
            db.upsert_daemon_heartbeat(
                "dashboard",
                "healthy",
                {
                    "pid": os.getpid(),
                    "rss_mb": rss_mb,
                    "peak_rss_mb": rss_mb,
                    "host": safe_host,
                    "port": port,
                },
                now_utc=datetime.now(UTC),
            )
            _emit_json({"service": "dashboard", "status": "healthy", "host": safe_host, "port": port})
            return

        stop_event = threading.Event()
        hb_thread = _start_dashboard_heartbeat(
            resolved_data, host=safe_host, port=port, stop_event=stop_event
        )
        try:
            serve_dashboard(resolved_data, resolved_config, host=safe_host, port=port)
        finally:
            stop_event.set()
            hb_thread.join(timeout=2.0)
        return

    runner: MarketDataDaemon | NewsNlpDaemon | TraderDaemon
    if svc == "market-data":
        runner = MarketDataDaemon(settings, resolved_data, resolved_config)
    elif svc == "news-nlp":
        runner = NewsNlpDaemon(settings, resolved_data, resolved_config)
    else:
        runner = TraderDaemon(settings, resolved_data, resolved_config)

    if once:
        try:
            runner.startup()
            result = runner.run_once()
        finally:
            runner.shutdown()
        _emit_json({"service": svc, **result})
        return

    runner.run_forever()


@app.command("render-manifests")
def render_manifests_cmd(
    target: Annotated[
        Literal["macos", "systemd"],
        typer.Option("--target", help="Target OS init system: macos|systemd"),
    ] = "macos",
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", help="Output directory for generated service files"),
    ] = Path("deploy/macos"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root path embedded in manifests"),
    ] = None,
    username: Annotated[str, typer.Option("--username", help="OS user running the services")] = "trader",
) -> None:
    """Render macOS LaunchDaemon plists or Linux systemd unit files."""
    resolved_root = (repo_root or Path.cwd()).resolve()
    paths = write_service_manifests(
        target,
        out_dir,
        repo_root=resolved_root,
        username=username,
    )
    _emit_json({"target": target, "files": [str(p) for p in paths]})


if __name__ == "__main__":
    app()
