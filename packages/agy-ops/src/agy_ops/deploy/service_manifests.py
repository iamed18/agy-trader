from __future__ import annotations

import getpass
import plistlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    name: str
    label: str
    daemon_arg: str
    description: str
    throttle_interval: int = 15
    extra_args: tuple[str, ...] = ()


DEFAULT_SERVICE_SPECS: tuple[ServiceSpec, ...] = (
    ServiceSpec(
        name="market-data",
        label="com.agytrader.market-data",
        daemon_arg="market-data",
        description="agy-trader Market Data, FRED Macro, Earnings & SEC EDGAR Daemon",
    ),
    ServiceSpec(
        name="news-nlp",
        label="com.agytrader.news-nlp",
        daemon_arg="news-nlp",
        description="agy-trader News/RSS Ingestion & FinBERT Sentiment Daemon",
    ),
    ServiceSpec(
        name="trader",
        label="com.agytrader.trader",
        daemon_arg="trader",
        description="agy-trader Broker Reconciliation, Risk & 15:40 ET Execution Daemon",
    ),
    ServiceSpec(
        name="dashboard",
        label="com.agytrader.dashboard",
        daemon_arg="dashboard",
        description="agy-trader Read-Only Portfolio Dashboard (0.0.0.0:8080, LAN)",
        extra_args=("--host", "0.0.0.0", "--port", "8080"),
    ),
)


DEFAULT_LAUNCHD_LOG_DIR = Path("/private/var/log/agy-trader")


def render_launchd_plist(
    spec: ServiceSpec,
    *,
    repo_root: Path,
    username: str | None = None,
    executable_path: Path | None = None,
    log_dir: Path | None = None,
) -> str:
    """Render a macOS pre-login LaunchDaemon XML plist for a ServiceSpec."""
    resolved_user = username or getpass.getuser()
    root = Path(repo_root)
    exe = executable_path or (root / ".venv" / "bin" / "agy-ops")
    program_args = [str(exe), "daemon", spec.daemon_arg, *spec.extra_args]
    env_vars = {
        "HOME": f"/Users/{resolved_user}",
        "PATH": (
            f"{root / '.venv' / 'bin'}:/Users/{resolved_user}/.local/bin:"
            "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        ),
        "PYTHONUNBUFFERED": "1",
        "AGY_DATA_DIR": str(root / "data"),
        "AGY_CONFIG_DIR": str(root / "config"),
    }
    payload: dict[str, object] = {
        "Label": spec.label,
        "UserName": resolved_user,
        "GroupName": "staff",
        "WorkingDirectory": str(root),
        "ProgramArguments": program_args,
        "EnvironmentVariables": env_vars,
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": int(spec.throttle_interval),
    }
    if log_dir is not None:
        logs = Path(log_dir)
        payload["StandardOutPath"] = str(logs / f"{spec.name}.stdout.log")
        payload["StandardErrorPath"] = str(logs / f"{spec.name}.stderr.log")
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True).decode("utf-8")


def render_systemd_unit(
    spec: ServiceSpec,
    *,
    repo_root: Path,
    username: str | None = None,
    executable_path: Path | None = None,
) -> str:
    """Render a portable Linux systemd service unit for a ServiceSpec."""
    resolved_user = username or getpass.getuser()
    root = Path(repo_root)
    exe = executable_path or (root / ".venv" / "bin" / "agy-ops")
    exec_cmd = " ".join([str(exe), "daemon", spec.daemon_arg, *spec.extra_args])
    return (
        f"[Unit]\n"
        f"Description={spec.description}\n"
        f"After=network-online.target\n"
        f"Wants=network-online.target\n\n"
        f"[Service]\n"
        f"Type=simple\n"
        f"User={resolved_user}\n"
        f"WorkingDirectory={root}\n"
        f"Environment=PYTHONUNBUFFERED=1\n"
        f"Environment=AGY_DATA_DIR={root / 'data'}\n"
        f"Environment=AGY_CONFIG_DIR={root / 'config'}\n"
        f"ExecStart={exec_cmd}\n"
        f"Restart=always\n"
        f"RestartSec={int(spec.throttle_interval)}\n\n"
        f"[Install]\n"
        f"WantedBy=multi-user.target\n"
    )


def write_service_manifests(
    target: Literal["macos", "systemd"],
    out_dir: Path,
    *,
    repo_root: Path,
    username: str | None = None,
) -> list[Path]:
    """Generate and write all 4 service manifest files for the given OS target."""
    resolved_user = username or getpass.getuser()
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for spec in DEFAULT_SERVICE_SPECS:
        if target == "macos":
            filename = f"{spec.label}.plist"
            content = render_launchd_plist(spec, repo_root=repo_root, username=resolved_user)
        elif target == "systemd":
            filename = f"agytrader-{spec.name}.service"
            content = render_systemd_unit(spec, repo_root=repo_root, username=resolved_user)
        else:
            raise ValueError(f"Unsupported target {target!r}: expected 'macos' or 'systemd'")
        path = dest / filename
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written
