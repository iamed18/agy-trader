from __future__ import annotations

import json
import os
import subprocess
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
from agy_ops.notifications import (
    DEFAULT_DISCORD_FOOTER,
    DiscordWebhookNotifier,
    build_daily_summary_discord_embed,
    build_morning_discord_embed,
    build_test_discord_embed,
    redact_webhook_url,
)
from agy_ops.reporting.digests import (
    build_daily_summary,
    build_morning_plan,
    build_readiness_report,
)
from agy_trader.core.config import EnvSettings
from agy_trader.core.control_plane import ControlPlane, ControlPlaneError
from agy_trader.storage.db import StateDB

DAEMONCTL_PATH = "/usr/local/sbin/agy-daemonctl"
INSTALL_LAUNCHDAEMONS_CMD = "sudo ./deploy/macos/install_launchdaemons.sh"
SERVICE_TARGETS = ("market-data", "news-nlp", "trader", "dashboard", "all")
CONTROL_PLANE_RESTART_SERVICES = ("market-data", "news-nlp", "trader")

app = typer.Typer(
    name="agy-ops",
    help="Always-on daemons, read-only dashboard, and operational reporting CLI",
    no_args_is_help=True,
)
notify_app = typer.Typer(
    name="notify",
    help="Send operational Discord notifications to the configured webhook",
    no_args_is_help=True,
)
service_app = typer.Typer(
    name="service",
    help="Manage agy-trader macOS LaunchDaemons without interactive sudo prompts",
    no_args_is_help=True,
)
app.add_typer(notify_app, name="notify")
app.add_typer(service_app, name="service")
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
        runner = TraderDaemon(settings, resolved_data, resolved_config, require_env_file=True)

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
    username: Annotated[
        str | None,
        typer.Option("--username", help="OS user running the services (defaults to current user)"),
    ] = None,
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


def _post_discord_embed(
    settings: EnvSettings,
    embed: dict[str, Any],
    *,
    dry_run: bool,
) -> tuple[bool, str | None]:
    raw_url = settings.discord_webhook_url.strip()
    if dry_run:
        return False, redact_webhook_url(raw_url) or None
    if not raw_url:
        err_console.print(
            "[red]Error:[/red] DISCORD_WEBHOOK_URL is not configured "
            "(set it in environment/.env or pass --dry-run)."
        )
        raise typer.Exit(code=1)
    try:
        with DiscordWebhookNotifier(raw_url) as notifier:
            delivery = notifier.send_embed(
                title=embed["title"],
                description=embed["description"],
                color=embed["color"],
                fields=embed.get("fields", ()),
                footer=embed.get("footer", DEFAULT_DISCORD_FOOTER),
            )
    except (ValueError, RuntimeError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    return True, str(delivery["webhook_url"])


def _format_morning_archive_markdown(
    readiness: dict[str, Any],
    morning_plan: dict[str, Any],
) -> str:
    lines = [
        str(morning_plan.get("markdown_summary", "")),
        "",
        f"## Readiness Status: {readiness['status']}",
    ]
    issues = readiness.get("issues", [])
    if issues:
        for issue in issues:
            lines.append(
                f"- [{str(issue.get('severity', '')).upper()}] "
                f"{issue.get('check', '?')}: {issue.get('message', '')}"
            )
    else:
        lines.append("- All readiness checks passed.")
    return "\n".join(lines).rstrip() + "\n"


@notify_app.command("morning")
def notify_morning_cmd(
    ctx: typer.Context,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Render & archive report without posting to Discord"),
    ] = False,
    report_dir: Annotated[
        Path,
        typer.Option("--report-dir", help="Directory to archive morning_plan_latest.md"),
    ] = Path("reports/daily"),
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Build morning readiness & plan, archive markdown, and post to Discord."""
    settings, resolved_data, resolved_config = _resolve_paths(ctx, data_dir, config_dir)
    readiness = build_readiness_report(resolved_data, resolved_config)
    morning_plan = build_morning_plan(resolved_data, resolved_config)

    report_dir.mkdir(parents=True, exist_ok=True)
    archive_path = report_dir / "morning_plan_latest.md"
    archive_path.write_text(
        _format_morning_archive_markdown(readiness, morning_plan),
        encoding="utf-8",
    )

    embed = build_morning_discord_embed(readiness, morning_plan)
    sent, redacted_url = _post_discord_embed(settings, embed, dry_run=dry_run)

    if json_output:
        _emit_json(
            {
                "command": "morning",
                "dry_run": dry_run,
                "sent": sent,
                "readiness_status": readiness["status"],
                "archive_path": str(archive_path),
                "webhook_url": redacted_url,
                "embed": embed,
            }
        )
        return

    prefix = "[DRY RUN] " if dry_run else ""
    action = "prepared" if dry_run else "sent"
    typer.echo(
        f"{prefix}Morning notification {action} "
        f"(status={readiness['status']}, archive={archive_path})"
    )


@notify_app.command("close")
def notify_close_cmd(
    ctx: typer.Context,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Render & archive summary without posting to Discord"),
    ] = False,
    report_dir: Annotated[
        Path,
        typer.Option("--report-dir", help="Directory to archive daily_summary_latest.md"),
    ] = Path("reports/daily"),
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Build daily close summary, archive markdown, and post to Discord."""
    settings, resolved_data, resolved_config = _resolve_paths(ctx, data_dir, config_dir)
    summary = build_daily_summary(resolved_data, resolved_config)

    report_dir.mkdir(parents=True, exist_ok=True)
    archive_path = report_dir / "daily_summary_latest.md"
    archive_path.write_text(
        str(summary.get("markdown_summary", "")).rstrip() + "\n",
        encoding="utf-8",
    )

    embed = build_daily_summary_discord_embed(summary)
    sent, redacted_url = _post_discord_embed(settings, embed, dry_run=dry_run)

    if json_output:
        _emit_json(
            {
                "command": "close",
                "dry_run": dry_run,
                "sent": sent,
                "session_date_et": summary["session_date_et"],
                "archive_path": str(archive_path),
                "webhook_url": redacted_url,
                "embed": embed,
            }
        )
        return

    prefix = "[DRY RUN] " if dry_run else ""
    action = "prepared" if dry_run else "sent"
    typer.echo(
        f"{prefix}Daily close notification {action} "
        f"(date={summary['session_date_et']}, archive={archive_path})"
    )


@notify_app.command("test")
def notify_test_cmd(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Render test embed without posting to Discord"),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Send a connectivity verification embed to the configured Discord webhook."""
    settings = EnvSettings()
    embed = build_test_discord_embed()
    sent, redacted_url = _post_discord_embed(settings, embed, dry_run=dry_run)

    if json_output:
        _emit_json(
            {
                "command": "test",
                "dry_run": dry_run,
                "sent": sent,
                "webhook_url": redacted_url,
                "embed": embed,
            }
        )
        return

    prefix = "[DRY RUN] " if dry_run else ""
    action = "prepared" if dry_run else "sent"
    typer.echo(f"{prefix}Test notification {action} for configured Discord webhook")


def _service_fail(
    action: str,
    target: str,
    message: str,
    *,
    json_output: bool,
    exit_code: int = 1,
) -> None:
    if json_output:
        _emit_json({"action": action, "target": target, "error": message})
    else:
        err_console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=exit_code)


def _validate_service_target(action: str, target: str, *, json_output: bool) -> str:
    cleaned = target.strip().lower()
    if cleaned not in SERVICE_TARGETS:
        _service_fail(
            action,
            target,
            f"Invalid service target {target!r}; expected one of {', '.join(SERVICE_TARGETS)}",
            json_output=json_output,
        )
    return cleaned


def _invoke_daemonctl(action: str, target: str, *, json_output: bool) -> dict[str, Any]:
    cmd = ["/usr/bin/sudo", "-n", DAEMONCTL_PATH, action, target, "--json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        _service_fail(
            action,
            target,
            f"Failed to execute /usr/bin/sudo {DAEMONCTL_PATH}: {exc}. Run once: {INSTALL_LAUNCHDAEMONS_CMD}",
            json_output=json_output,
        )
        raise AssertionError("unreachable")  # pragma: no cover

    if proc.returncode != 0:
        combined = f"{proc.stderr}\n{proc.stdout}".lower()
        needs_install = (not Path(DAEMONCTL_PATH).exists()) or any(
            marker in combined
            for marker in (
                "password is required",
                "command not found",
                "no such file",
                "not allowed to execute",
                "may not run sudo",
            )
        )
        if needs_install:
            msg = (
                f"{DAEMONCTL_PATH} is not installed or passwordless sudo (-n) is not configured. "
                f"Run once: {INSTALL_LAUNCHDAEMONS_CMD}"
            )
        else:
            msg = ""
            if proc.stdout.strip():
                try:
                    parsed_err = json.loads(proc.stdout)
                    if isinstance(parsed_err, dict) and parsed_err.get("error"):
                        msg = str(parsed_err["error"])
                except json.JSONDecodeError:
                    pass
            if not msg:
                msg = (
                    proc.stderr.strip()
                    or proc.stdout.strip()
                    or f"agy-daemonctl {action} failed with exit code {proc.returncode}"
                )
        _service_fail(action, target, msg, json_output=json_output, exit_code=proc.returncode or 1)

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        _service_fail(
            action,
            target,
            f"Invalid JSON from agy-daemonctl: {exc}",
            json_output=json_output,
        )
        raise AssertionError("unreachable")  # pragma: no cover

    if not isinstance(payload, dict):
        _service_fail(
            action,
            target,
            "Unexpected response format from agy-daemonctl",
            json_output=json_output,
        )
    return payload


def _render_service_result(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        _emit_json(payload)
        return

    action = str(payload.get("action", "service"))
    services = payload.get("services", [])
    table = Table(title=f"agy-ops service {action}")
    table.add_column("Service", style="cyan")
    table.add_column("Loaded")
    table.add_column("State")
    table.add_column("PID")
    table.add_column("Last Exit")
    table.add_column("Plist Installed")
    show_outcome = any(isinstance(item, dict) and "outcome" in item for item in services)
    if show_outcome:
        table.add_column("Outcome")

    for item in services:
        if not isinstance(item, dict):
            continue
        loaded = bool(item.get("loaded"))
        loaded_str = "[green]yes[/green]" if loaded else "[yellow]no[/yellow]"
        pid_str = str(item["pid"]) if item.get("pid") is not None else "-"
        exit_str = (
            str(item["last_exit_status"]) if item.get("last_exit_status") is not None else "-"
        )
        plist_str = "yes" if item.get("plist_installed") else "no"
        state_str = str(item.get("state") or ("unloaded" if not loaded else "-"))
        row = [
            str(item.get("service", "")),
            loaded_str,
            state_str,
            pid_str,
            exit_str,
            plist_str,
        ]
        if show_outcome:
            row.append(str(item.get("outcome", "")))
        table.add_row(*row)
    console.print(table)


@service_app.command("status")
def service_status_cmd(
    target: Annotated[
        str,
        typer.Argument(help="Target service: market-data|news-nlp|trader|dashboard|all"),
    ] = "all",
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Inspect macOS LaunchDaemon status for one or all agy-trader services."""
    resolved = _validate_service_target("status", target, json_output=json_output)
    payload = _invoke_daemonctl("status", resolved, json_output=json_output)
    _render_service_result(payload, json_output=json_output)


@service_app.command("start")
def service_start_cmd(
    target: Annotated[
        str,
        typer.Argument(help="Target service: market-data|news-nlp|trader|dashboard|all"),
    ] = "all",
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Start one or all agy-trader macOS LaunchDaemons."""
    resolved = _validate_service_target("start", target, json_output=json_output)
    payload = _invoke_daemonctl("start", resolved, json_output=json_output)
    _render_service_result(payload, json_output=json_output)


@service_app.command("stop")
def service_stop_cmd(
    target: Annotated[
        str,
        typer.Argument(help="Target service: market-data|news-nlp|trader|dashboard|all"),
    ] = "all",
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Stop (bootout) one or all agy-trader macOS LaunchDaemons cleanly."""
    resolved = _validate_service_target("stop", target, json_output=json_output)
    payload = _invoke_daemonctl("stop", resolved, json_output=json_output)
    _render_service_result(payload, json_output=json_output)


@service_app.command("restart")
def service_restart_cmd(
    ctx: typer.Context,
    target: Annotated[
        str,
        typer.Argument(help="Target service: market-data|news-nlp|trader|dashboard|all"),
    ] = "all",
    actor: Annotated[
        str,
        typer.Option("--actor", help="Actor identifier (use 'operator' to bypass agent rate limit)"),
    ] = "ops",
    reason: Annotated[
        str,
        typer.Option("--reason", help="Audit reason for service restart"),
    ] = "",
    data_dir: Annotated[Path | None, typer.Option("--data-dir", help="Override data directory")] = None,
    config_dir: Annotated[Path | None, typer.Option("--config-dir", help="Override config directory")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Restart one or all agy-trader macOS LaunchDaemons with ControlPlane rate limiting for non-operators."""
    resolved = _validate_service_target("restart", target, json_output=json_output)
    actor_clean = actor.strip()
    if actor_clean != "operator":
        if resolved == "all":
            auth_services: tuple[str, ...] = CONTROL_PLANE_RESTART_SERVICES
        elif resolved in CONTROL_PLANE_RESTART_SERVICES:
            auth_services = (resolved,)
        else:
            auth_services = ()

        if auth_services:
            _, resolved_data, _ = _resolve_paths(ctx, data_dir, config_dir)
            cp = ControlPlane(StateDB(resolved_data / "state.db"))
            try:
                for svc in auth_services:
                    cp.authorize_service_restart(
                        svc, actor=actor_clean or "ops", reason=reason
                    )
            except (ControlPlaneError, ValueError) as exc:
                _service_fail("restart", resolved, str(exc), json_output=json_output)

    payload = _invoke_daemonctl("restart", resolved, json_output=json_output)
    _render_service_result(payload, json_output=json_output)


@service_app.command("sync-plists")
def service_sync_plists_cmd(
    target: Annotated[
        str,
        typer.Argument(help="Target service: market-data|news-nlp|trader|dashboard|all"),
    ] = "all",
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output")] = False,
) -> None:
    """Validate, install, and reload LaunchDaemon plists from deploy/macos."""
    resolved = _validate_service_target("sync-plists", target, json_output=json_output)
    payload = _invoke_daemonctl("sync-plists", resolved, json_output=json_output)
    _render_service_result(payload, json_output=json_output)


if __name__ == "__main__":
    app()
