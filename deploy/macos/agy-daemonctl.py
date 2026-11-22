#!/usr/bin/python3 -I
"""Privilege-separated launchd control helper for agy-trader macOS daemons."""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import pwd
import re
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

RUN_USER = "trader"
RUN_GROUP = "staff"
REPO_ROOT = Path("/path/to/agy-trader")
PLIST_SRC_DIR = REPO_ROOT / "deploy" / "macos"
LOG_DIR = Path("/private/var/log/agy-trader")
VENV_AGY_OPS = "/path/to/agy-trader/.venv/bin/agy-ops"
TARGET_DIR = Path("/Library/LaunchDaemons")
ALLOWED_SERVICES = ("market-data", "news-nlp", "trader", "dashboard")
ALLOWED_PLIST_KEYS = frozenset(
    {
        "Label",
        "UserName",
        "GroupName",
        "WorkingDirectory",
        "ProgramArguments",
        "EnvironmentVariables",
        "RunAtLoad",
        "KeepAlive",
        "ThrottleInterval",
        "StandardOutPath",
        "StandardErrorPath",
    }
)
LAUNCHCTL = "/bin/launchctl"
PLUTIL = "/usr/bin/plutil"

MAX_PLIST_BYTES = 65536
MAX_PROGRAM_ARGS = 16
MIN_THROTTLE_INTERVAL = 1
MAX_THROTTLE_INTERVAL = 3600
BOOTOUT_TIMEOUT_SECONDS = 5.0
BOOTOUT_POLL_INTERVAL_SECONDS = 0.25

_SANITIZED_SUBPROCESS_ENV: dict[str, str] = {
    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    "HOME": "/var/root",
    "LANG": "C.UTF-8",
}

_STATE_RE = re.compile(r"^\s*state\s*=\s*(.+)$", re.MULTILINE)
_PID_RE = re.compile(r"^\s*pid\s*=\s*(\d+)\s*$", re.MULTILINE)
_LAST_EXIT_RE = re.compile(r"^\s*last exit code\s*=\s*(.+)$", re.MULTILINE)


class PlistValidationError(ValueError):
    """Raised when a LaunchDaemon plist fails security or schema validation."""


class DaemonCtlError(RuntimeError):
    """Raised when a launchctl, plutil, or filesystem operation fails."""


def _service_label(service: str) -> str:
    if service not in ALLOWED_SERVICES:
        raise PlistValidationError(
            f"Unknown service {service!r}; allowed: {', '.join(ALLOWED_SERVICES)}"
        )
    return f"com.agytrader.{service}"


def _has_ascii_control_char(value: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


def _verify_no_symlink_log_target(log_dir: Path, log_file: Path) -> None:
    for ancestor in (log_dir, *log_dir.parents):
        if ancestor.is_symlink():
            raise PlistValidationError(f"Log path ancestor must not be a symlink: {ancestor}")

    if log_dir.exists() and not log_dir.is_dir():
        raise PlistValidationError(f"Log directory path is not a directory: {log_dir}")

    if os.geteuid() == 0:
        if not log_dir.exists():
            raise PlistValidationError(f"Log directory does not exist: {log_dir}")
        for ancestor in (log_dir, *log_dir.parents):
            st_anc = ancestor.lstat()
            if st_anc.st_uid != 0:
                raise PlistValidationError(
                    f"Log directory ancestor {ancestor} must be owned by uid 0 "
                    f"(got uid {st_anc.st_uid})"
                )
            if stat.S_IMODE(st_anc.st_mode) & 0o022:
                raise PlistValidationError(
                    f"Log directory ancestor {ancestor} must not be group/world-writable"
                )

    if log_file.is_symlink():
        raise PlistValidationError(f"Log file must not be a symlink: {log_file}")
    if log_file.exists():
        st = log_file.lstat()
        if not stat.S_ISREG(st.st_mode):
            raise PlistValidationError(f"Log path must be a regular file: {log_file}")
        if st.st_nlink != 1:
            raise PlistValidationError(f"Log file must not have multiple hardlinks: {log_file}")


def read_plist_nosymlink(path: Path) -> bytes:
    """Read plist bytes safely without following symlinks, hardlinks, or root-owned files."""
    for ancestor in path.parents:
        if ancestor.is_symlink():
            raise PlistValidationError(
                f"Plist ancestor directory must not be a symlink: {ancestor}"
            )

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError as exc:
        raise PlistValidationError(f"Cannot safely open plist {path}: {exc}") from exc

    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise PlistValidationError(f"Plist path is not a regular file: {path}")
        if st.st_nlink != 1:
            raise PlistValidationError(
                f"Plist {path} must have link count 1 (got {st.st_nlink})"
            )
        if os.geteuid() == 0:
            run_uid = pwd.getpwnam(RUN_USER).pw_uid
            if run_uid == 0:
                raise PlistValidationError(f"{RUN_USER!r} must not resolve to uid 0")
            if st.st_uid != run_uid:
                raise PlistValidationError(
                    f"Plist {path} must be owned by {RUN_USER} (uid {run_uid}), "
                    f"got uid {st.st_uid}"
                )
        if st.st_size > MAX_PLIST_BYTES:
            raise PlistValidationError(
                f"Plist {path} exceeds maximum size of {MAX_PLIST_BYTES} bytes"
            )
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            raw = handle.read(MAX_PLIST_BYTES + 1)
    finally:
        if fd >= 0:
            os.close(fd)

    if len(raw) > MAX_PLIST_BYTES:
        raise PlistValidationError(
            f"Plist {path} exceeds maximum size of {MAX_PLIST_BYTES} bytes"
        )
    return raw


def validate_and_canonicalize_plist(
    service: str,
    raw_bytes: bytes,
    *,
    log_dir: Path = LOG_DIR,
) -> bytes:
    """Strictly validate a LaunchDaemon plist and re-serialize it canonically."""
    expected_label = _service_label(service)

    if not isinstance(raw_bytes, bytes) or not raw_bytes:
        raise PlistValidationError("Plist payload must be non-empty bytes")
    if len(raw_bytes) > MAX_PLIST_BYTES:
        raise PlistValidationError(
            f"Plist payload exceeds maximum size of {MAX_PLIST_BYTES} bytes"
        )
    if b"<!ENTITY" in raw_bytes.upper():
        raise PlistValidationError("XML <!ENTITY declarations are forbidden in plists")

    try:
        data = plistlib.loads(raw_bytes, fmt=plistlib.FMT_XML)
    except Exception as exc:
        raise PlistValidationError(f"Invalid XML plist for {service}: {exc}") from exc

    if not isinstance(data, dict):
        raise PlistValidationError("Plist root element must be a dict")

    for key in data:
        if not isinstance(key, str) or key not in ALLOWED_PLIST_KEYS:
            raise PlistValidationError(f"Forbidden plist key: {key!r}")

    if data.get("Label") != expected_label:
        raise PlistValidationError(
            f"Label must be {expected_label!r}, got {data.get('Label')!r}"
        )
    if data.get("UserName") != RUN_USER:
        raise PlistValidationError(
            f"UserName must be {RUN_USER!r}, got {data.get('UserName')!r}"
        )
    if data.get("GroupName") != RUN_GROUP:
        raise PlistValidationError(
            f"GroupName must be {RUN_GROUP!r}, got {data.get('GroupName')!r}"
        )
    if data.get("WorkingDirectory") != str(REPO_ROOT):
        raise PlistValidationError(
            f"WorkingDirectory must be {str(REPO_ROOT)!r}, got {data.get('WorkingDirectory')!r}"
        )

    prog_args = data.get("ProgramArguments")
    if (
        not isinstance(prog_args, list)
        or len(prog_args) < 3
        or len(prog_args) > MAX_PROGRAM_ARGS
        or not all(isinstance(arg, str) and len(arg) > 0 for arg in prog_args)
    ):
        raise PlistValidationError(
            f"ProgramArguments must be a list of 3..{MAX_PROGRAM_ARGS} non-empty strings"
        )
    if prog_args[0] != VENV_AGY_OPS:
        raise PlistValidationError(
            f"ProgramArguments[0] must be {VENV_AGY_OPS!r}, got {prog_args[0]!r}"
        )
    if prog_args[1:3] != ["daemon", service]:
        raise PlistValidationError(
            f"ProgramArguments[1:3] must be ['daemon', {service!r}], got {prog_args[1:3]!r}"
        )

    for key, filename in (
        ("StandardOutPath", f"{service}.stdout.log"),
        ("StandardErrorPath", f"{service}.stderr.log"),
    ):
        if key in data:
            expected_path = log_dir / filename
            if not isinstance(data[key], str) or data[key] != str(expected_path):
                raise PlistValidationError(
                    f"{key} must be {str(expected_path)!r}, got {data[key]!r}"
                )
            _verify_no_symlink_log_target(log_dir, expected_path)

    if "EnvironmentVariables" in data:
        env_vars = data["EnvironmentVariables"]
        if not isinstance(env_vars, dict):
            raise PlistValidationError("EnvironmentVariables must be a dict")
        for env_key, env_val in env_vars.items():
            if not isinstance(env_key, str) or not env_key or not isinstance(env_val, str):
                raise PlistValidationError(
                    "EnvironmentVariables must map non-empty string keys to string values"
                )
            if _has_ascii_control_char(env_key) or _has_ascii_control_char(env_val):
                raise PlistValidationError(
                    f"EnvironmentVariables must not contain ASCII control characters ({env_key!r})"
                )
            upper_key = env_key.upper()
            if upper_key.startswith(("DYLD_", "LD_")) or (
                upper_key.startswith("PYTHON") and env_key != "PYTHONUNBUFFERED"
            ):
                raise PlistValidationError(
                    f"Forbidden EnvironmentVariables key: {env_key!r}"
                )

    for bool_key in ("RunAtLoad", "KeepAlive"):
        if bool_key in data and not isinstance(data[bool_key], bool):
            raise PlistValidationError(f"{bool_key} must be a boolean")

    if "ThrottleInterval" in data:
        throttle = data["ThrottleInterval"]
        if (
            not isinstance(throttle, int)
            or isinstance(throttle, bool)
            or throttle < MIN_THROTTLE_INTERVAL
            or throttle > MAX_THROTTLE_INTERVAL
        ):
            raise PlistValidationError(
                f"ThrottleInterval must be an integer in [{MIN_THROTTLE_INTERVAL}, "
                f"{MAX_THROTTLE_INTERVAL}], got {throttle!r}"
            )

    try:
        return plistlib.dumps(data, fmt=plistlib.FMT_XML, sort_keys=True)
    except (OverflowError, ValueError, TypeError) as exc:
        raise PlistValidationError(f"Cannot canonicalize plist for {service}: {exc}") from exc


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=dict(_SANITIZED_SUBPROCESS_ENV),
    )


def parse_launchctl_print(output: str) -> dict[str, Any]:
    """Extract state, pid, and last_exit_status from `launchctl print` output."""
    state_match = _STATE_RE.search(output)
    pid_match = _PID_RE.search(output)
    exit_match = _LAST_EXIT_RE.search(output)

    state = state_match.group(1).strip() if state_match else None
    pid = int(pid_match.group(1)) if pid_match else None

    last_exit_status: int | str | None = None
    if exit_match:
        raw_exit = exit_match.group(1).strip()
        if raw_exit != "(never exited)":
            try:
                last_exit_status = int(raw_exit)
            except ValueError:
                last_exit_status = raw_exit

    return {
        "state": state,
        "pid": pid,
        "last_exit_status": last_exit_status,
    }


def get_service_status(service: str, *, target_dir: Path = TARGET_DIR) -> dict[str, Any]:
    label = _service_label(service)
    dst_plist = target_dir / f"{label}.plist"
    proc = _run_cmd([LAUNCHCTL, "print", f"system/{label}"])
    loaded = proc.returncode == 0
    parsed = (
        parse_launchctl_print(proc.stdout)
        if loaded
        else {"state": None, "pid": None, "last_exit_status": None}
    )
    return {
        "service": service,
        "label": label,
        "loaded": loaded,
        "state": parsed["state"],
        "pid": parsed["pid"],
        "last_exit_status": parsed["last_exit_status"],
        "plist_installed": dst_plist.exists(),
        "plist_path": str(dst_plist),
    }


def _bootout_and_wait(label: str) -> None:
    proc = _run_cmd([LAUNCHCTL, "bootout", f"system/{label}"])
    deadline = time.monotonic() + BOOTOUT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        chk = _run_cmd([LAUNCHCTL, "print", f"system/{label}"])
        if chk.returncode != 0:
            return
        time.sleep(BOOTOUT_POLL_INTERVAL_SECONDS)
    detail = proc.stderr.strip() or proc.stdout.strip() or "timed out waiting for bootout"
    raise DaemonCtlError(f"Failed to unload system/{label}: {detail}")


def _bootstrap(label: str, dst_plist: Path) -> None:
    # Best-effort enable in case the service was previously disabled via launchctl disable
    _run_cmd([LAUNCHCTL, "enable", f"system/{label}"])
    proc = _run_cmd([LAUNCHCTL, "bootstrap", "system", str(dst_plist)])
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise DaemonCtlError(f"launchctl bootstrap failed for {dst_plist}: {detail}")


def _kickstart(label: str, *, kill_existing: bool) -> None:
    cmd = [LAUNCHCTL, "kickstart"]
    if kill_existing:
        cmd.append("-k")
    cmd.append(f"system/{label}")
    proc = _run_cmd(cmd)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise DaemonCtlError(f"launchctl kickstart failed for system/{label}: {detail}")


def start_service(service: str, *, target_dir: Path = TARGET_DIR) -> dict[str, Any]:
    status = get_service_status(service, target_dir=target_dir)
    label = status["label"]
    dst_plist = Path(status["plist_path"])

    if not status["loaded"]:
        if not status["plist_installed"]:
            raise DaemonCtlError(
                f"Cannot start {service}: {dst_plist} is not installed (run sync-plists first)"
            )
        _bootstrap(label, dst_plist)
        outcome = "bootstrapped"
    elif status["pid"] is None or status["state"] != "running":
        _kickstart(label, kill_existing=False)
        outcome = "kickstarted"
    else:
        outcome = "already_running"

    updated = get_service_status(service, target_dir=target_dir)
    return {**updated, "action": "start", "outcome": outcome}


def stop_service(service: str, *, target_dir: Path = TARGET_DIR) -> dict[str, Any]:
    status = get_service_status(service, target_dir=target_dir)
    label = status["label"]
    if status["loaded"]:
        _bootout_and_wait(label)
        outcome = "stopped"
    else:
        outcome = "already_stopped"

    updated = get_service_status(service, target_dir=target_dir)
    return {**updated, "action": "stop", "outcome": outcome}


def restart_service(service: str, *, target_dir: Path = TARGET_DIR) -> dict[str, Any]:
    status = get_service_status(service, target_dir=target_dir)
    label = status["label"]
    dst_plist = Path(status["plist_path"])

    if status["loaded"]:
        _kickstart(label, kill_existing=True)
        outcome = "restarted"
    elif status["plist_installed"]:
        _bootstrap(label, dst_plist)
        outcome = "bootstrapped"
    else:
        raise DaemonCtlError(
            f"Cannot restart {service}: {dst_plist} is not installed (run sync-plists first)"
        )

    updated = get_service_status(service, target_dir=target_dir)
    return {**updated, "action": "restart", "outcome": outcome}


def sync_service_plist(
    service: str,
    *,
    canonical_bytes: bytes | None = None,
    plist_src_dir: Path = PLIST_SRC_DIR,
    target_dir: Path = TARGET_DIR,
    log_dir: Path = LOG_DIR,
) -> dict[str, Any]:
    label = _service_label(service)
    src_plist = plist_src_dir / f"{label}.plist"
    dst_plist = target_dir / f"{label}.plist"

    if target_dir.is_symlink():
        raise PlistValidationError(f"Target directory must not be a symlink: {target_dir}")
    if not target_dir.is_dir():
        raise DaemonCtlError(f"Target directory does not exist: {target_dir}")

    if canonical_bytes is None:
        raw_bytes = read_plist_nosymlink(src_plist)
        canonical_bytes = validate_and_canonicalize_plist(service, raw_bytes, log_dir=log_dir)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(target_dir),
            prefix=f".{label}.",
            suffix=".plist.tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(canonical_bytes)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        os.chmod(tmp_path, 0o644)
        if os.geteuid() == 0:
            os.chown(tmp_path, 0, 0)

        lint_proc = _run_cmd([PLUTIL, "-lint", str(tmp_path)])
        if lint_proc.returncode != 0:
            detail = (
                lint_proc.stderr.strip() or lint_proc.stdout.strip() or f"exit {lint_proc.returncode}"
            )
            raise DaemonCtlError(f"plutil -lint failed for {label}: {detail}")

        os.replace(tmp_path, dst_plist)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    if get_service_status(service, target_dir=target_dir)["loaded"]:
        _bootout_and_wait(label)
    _bootstrap(label, dst_plist)

    updated = get_service_status(service, target_dir=target_dir)
    return {**updated, "action": "sync-plists", "outcome": "synced"}


def _resolve_targets(target: str) -> tuple[str, ...]:
    if target == "all":
        return ALLOWED_SERVICES
    if target in ALLOWED_SERVICES:
        return (target,)
    raise PlistValidationError(
        f"Invalid target {target!r}; expected one of {', '.join(ALLOWED_SERVICES)}, all"
    )


def _format_human(action: str, results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in results:
        svc = item["service"]
        loaded = item.get("loaded")
        pid = item.get("pid")
        state = item.get("state") or ("unloaded" if not loaded else "unknown")
        last_exit = item.get("last_exit_status")
        plist_ok = item.get("plist_installed")
        extra = f" outcome={item['outcome']}" if "outcome" in item else ""
        lines.append(
            f"{svc:12s} loaded={loaded} state={state} pid={pid} "
            f"last_exit={last_exit} plist_installed={plist_ok}{extra}"
        )
    return f"[{action}]\n" + "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agy-daemonctl",
        description="Privilege-separated launchctl manager for agy-trader macOS LaunchDaemons",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=(*ALLOWED_SERVICES, "all"),
        help="Target service or 'all' (default: all)",
    )
    shared.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit structured JSON output",
    )

    for action in ("status", "start", "stop", "restart", "sync-plists"):
        subparsers.add_parser(action, parents=[shared], help=f"Run {action} on daemon(s)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    action: str = args.action
    target: str = args.target
    json_output: bool = bool(args.json_output)

    try:
        services = _resolve_targets(target)
        results: list[dict[str, Any]] = []

        if action == "status":
            results = [get_service_status(svc) for svc in services]
        elif action == "start":
            results = [start_service(svc) for svc in services]
        elif action == "stop":
            results = [stop_service(svc) for svc in services]
        elif action == "restart":
            results = [restart_service(svc) for svc in services]
        elif action == "sync-plists":
            # Validate all source plists before touching /Library/LaunchDaemons
            validated: dict[str, bytes] = {}
            for svc in services:
                src_plist = PLIST_SRC_DIR / f"{_service_label(svc)}.plist"
                raw = read_plist_nosymlink(src_plist)
                validated[svc] = validate_and_canonicalize_plist(svc, raw, log_dir=LOG_DIR)
            for svc in services:
                results.append(sync_service_plist(svc, canonical_bytes=validated[svc]))
        else:
            raise DaemonCtlError(f"Unsupported action: {action!r}")

        payload = {"action": action, "target": target, "services": results}
        if json_output:
            print(json.dumps(payload, indent=2))
        else:
            print(_format_human(action, results))
        return 0
    except (PlistValidationError, DaemonCtlError, OSError) as exc:
        if json_output:
            print(json.dumps({"action": action, "target": target, "error": str(exc)}, indent=2))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
