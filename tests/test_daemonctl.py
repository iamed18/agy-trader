from __future__ import annotations

import copy
import importlib.util
import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from agy_ops.cli import app
from agy_ops.deploy.service_manifests import DEFAULT_SERVICE_SPECS, render_launchd_plist

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_MACOS_DIR = REPO_ROOT / "deploy" / "macos"
DAEMONCTL_PATH = DEPLOY_MACOS_DIR / "agy-daemonctl.py"
TEST_RUN_USER = "testtrader"
TEST_REPO_ROOT = Path("/Users/testtrader/src/agy-trader")
TEST_VENV_AGY_OPS = "/Users/testtrader/src/agy-trader/.venv/bin/agy-ops"

runner = CliRunner()


def _load_daemonctl() -> Any:
    spec = importlib.util.spec_from_file_location("agy_daemonctl", DAEMONCTL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["agy_daemonctl"] = module
    spec.loader.exec_module(module)
    return module


daemonctl = _load_daemonctl()


@pytest.fixture(autouse=True)
def _configure_daemonctl_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stamp synthetic non-PII host config onto the daemonctl template for unit tests."""
    monkeypatch.setattr(daemonctl, "RUN_USER", TEST_RUN_USER)
    monkeypatch.setattr(daemonctl, "RUN_GROUP", "staff")
    monkeypatch.setattr(daemonctl, "REPO_ROOT", TEST_REPO_ROOT)
    monkeypatch.setattr(daemonctl, "PLIST_SRC_DIR", TEST_REPO_ROOT / "deploy" / "macos")
    monkeypatch.setattr(daemonctl, "VENV_AGY_OPS", TEST_VENV_AGY_OPS)


def _valid_plist_dict(service: str = "trader", log_dir: Path | None = None) -> dict[str, Any]:
    resolved_logs = log_dir or daemonctl.LOG_DIR
    extra_args = ["--host", "127.0.0.1", "--port", "8080"] if service == "dashboard" else []
    return {
        "Label": f"com.agytrader.{service}",
        "UserName": TEST_RUN_USER,
        "GroupName": "staff",
        "WorkingDirectory": str(TEST_REPO_ROOT),
        "ProgramArguments": [TEST_VENV_AGY_OPS, "daemon", service, *extra_args],
        "EnvironmentVariables": {
            "AGY_CONFIG_DIR": str(TEST_REPO_ROOT / "config"),
            "AGY_DATA_DIR": str(TEST_REPO_ROOT / "data"),
            "HOME": f"/Users/{TEST_RUN_USER}",
            "PATH": f"{TEST_REPO_ROOT / '.venv/bin'}:/usr/bin:/bin",
            "PYTHONUNBUFFERED": "1",
        },
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 15,
        "StandardOutPath": str(resolved_logs / f"{service}.stdout.log"),
        "StandardErrorPath": str(resolved_logs / f"{service}.stderr.log"),
    }


def test_rendered_plists_and_newsyslog_template_use_root_var_log_dir(tmp_path: Path) -> None:
    assert daemonctl.LOG_DIR == Path("/private/var/log/agy-trader")
    plist_dir = tmp_path.resolve() / "plists"
    plist_dir.mkdir()
    for spec in DEFAULT_SERVICE_SPECS:
        rendered = render_launchd_plist(
            spec,
            repo_root=TEST_REPO_ROOT,
            username=TEST_RUN_USER,
        )
        plist_path = plist_dir / f"{spec.label}.plist"
        plist_path.write_text(rendered, encoding="utf-8")
        raw = daemonctl.read_plist_nosymlink(plist_path)
        parsed = plistlib.loads(raw, fmt=plistlib.FMT_XML)
        canonical = daemonctl.validate_and_canonicalize_plist(spec.name, raw)
        reparsed = plistlib.loads(canonical, fmt=plistlib.FMT_XML)
        assert reparsed == parsed
        assert reparsed["UserName"] == TEST_RUN_USER
        assert reparsed["GroupName"] == "staff"
        assert reparsed["ProgramArguments"][0] == TEST_VENV_AGY_OPS
        assert reparsed["ProgramArguments"][1:3] == ["daemon", spec.name]
        assert "StandardOutPath" not in reparsed
        assert "StandardErrorPath" not in reparsed

    newsyslog_lines = [
        line.strip()
        for line in (DEPLOY_MACOS_DIR / "com.agytrader.newsyslog.conf.tmpl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(newsyslog_lines) == 8
    for svc in daemonctl.ALLOWED_SERVICES:
        for stream in ("stdout", "stderr"):
            expected_log = f"/private/var/log/agy-trader/{svc}.{stream}.log"
            matching = [ln for ln in newsyslog_lines if ln.startswith(expected_log)]
            assert len(matching) == 1
            assert "__RUN_USER__:__RUN_GROUP__" in matching[0]
            assert matching[0].endswith("JN")
            assert "G" not in matching[0].split()[-1]


def test_daemonctl_fails_closed_when_template_unrendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemonctl, "RUN_USER", "__AGY_RUN_USER__")
    raw = plistlib.dumps(_valid_plist_dict("trader"), fmt=plistlib.FMT_XML)
    with pytest.raises(daemonctl.PlistValidationError, match="unrendered template placeholder"):
        daemonctl.validate_and_canonicalize_plist("trader", raw)
    assert daemonctl.main(["status", "trader", "--json"]) == 1


def test_daemonctl_uses_isolated_python3_shebang() -> None:
    first_line = DAEMONCTL_PATH.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "#!/usr/bin/python3 -I"


@pytest.mark.parametrize(
    ("mutator", "expected_msg"),
    [
        (lambda d: d.pop("UserName"), "UserName"),
        (lambda d: d.__setitem__("UserName", "root"), "UserName"),
        (lambda d: d.__setitem__("GroupName", "wheel"), "GroupName"),
        (lambda d: d.__setitem__("RootDirectory", "/"), "Forbidden plist key"),
        (lambda d: d.__setitem__("InitGroups", True), "Forbidden plist key"),
        (lambda d: d.__setitem__("Sockets", {"HTTP": {}}), "Forbidden plist key"),
        (lambda d: d.__setitem__("Nice", -20), "Forbidden plist key"),
        (lambda d: d.__setitem__("ProcessType", "Interactive"), "Forbidden plist key"),
        (lambda d: d.__setitem__("ThrottleInterval", 0), "ThrottleInterval"),
        (lambda d: d.__setitem__("ThrottleInterval", -5), "ThrottleInterval"),
        (lambda d: d.__setitem__("ThrottleInterval", 3601), "ThrottleInterval"),
        (lambda d: d.__setitem__("ThrottleInterval", True), "ThrottleInterval"),
        (lambda d: d.__setitem__("WorkingDirectory", "/tmp"), "WorkingDirectory"),
        (lambda d: d.__setitem__("Label", "com.agytrader.evil"), "Label"),
        (
            lambda d: d["ProgramArguments"].__setitem__(0, "/bin/sh"),
            "ProgramArguments[0]",
        ),
        (
            lambda d: d["ProgramArguments"].__setitem__(2, "other"),
            "ProgramArguments[1:3]",
        ),
        (
            lambda d: d["ProgramArguments"].extend([f"--x{i}" for i in range(20)]),
            "ProgramArguments",
        ),
        (
            lambda d: d["EnvironmentVariables"].__setitem__(
                "DYLD_INSERT_LIBRARIES", "/tmp/evil.dylib"
            ),
            "DYLD_INSERT_LIBRARIES",
        ),
        (
            lambda d: d["EnvironmentVariables"].__setitem__("PYTHONPATH", "/tmp/evil"),
            "PYTHONPATH",
        ),
        (
            lambda d: d["EnvironmentVariables"].__setitem__("LD_PRELOAD", "/tmp/evil.so"),
            "LD_PRELOAD",
        ),
        (
            lambda d: d["EnvironmentVariables"].__setitem__("FOO", "bar\nbaz"),
            "control characters",
        ),
        (
            lambda d: d["EnvironmentVariables"].__setitem__("BAD\x7fKEY", "ok"),
            "control characters",
        ),
        (
            lambda d: d.__setitem__("StandardOutPath", "/etc/pam.d/sudo"),
            "StandardOutPath",
        ),
        (
            lambda d: d.__setitem__("StandardErrorPath", "/tmp/outside.log"),
            "StandardErrorPath",
        ),
    ],
)
def test_validate_plist_rejects_privilege_escalation_vectors(
    tmp_path: Path,
    mutator: Any,
    expected_msg: str,
) -> None:
    log_dir = tmp_path.resolve() / "logs"
    log_dir.mkdir()
    payload = copy.deepcopy(_valid_plist_dict("trader", log_dir=log_dir))
    mutator(payload)
    raw = plistlib.dumps(payload, fmt=plistlib.FMT_XML)
    with pytest.raises(daemonctl.PlistValidationError, match=expected_msg.replace("[", r"\[")):
        daemonctl.validate_and_canonicalize_plist("trader", raw, log_dir=log_dir)


def test_validate_plist_rejects_xxe_and_oversized_payload(tmp_path: Path) -> None:
    log_dir = tmp_path.resolve() / "logs"
    log_dir.mkdir()
    xxe = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<!DOCTYPE plist [<!ENTITY secret SYSTEM "file:///etc/master.passwd">]>\n'
        b'<plist version="1.0"><dict/></plist>\n'
    )
    with pytest.raises(daemonctl.PlistValidationError, match="ENTITY"):
        daemonctl.validate_and_canonicalize_plist("trader", xxe, log_dir=log_dir)

    oversized = b"A" * (daemonctl.MAX_PLIST_BYTES + 1)
    with pytest.raises(daemonctl.PlistValidationError, match="maximum size"):
        daemonctl.validate_and_canonicalize_plist("trader", oversized, log_dir=log_dir)

    # Raw XML 64-bit-overflow integer parsed by plistlib.loads must raise PlistValidationError
    base_xml = plistlib.dumps(_valid_plist_dict("trader", log_dir=log_dir), fmt=plistlib.FMT_XML)
    overflow_xml = base_xml.replace(b"<integer>15</integer>", b"<integer>1000000000000000000000000000000</integer>")
    with pytest.raises(daemonctl.PlistValidationError, match="ThrottleInterval"):
        daemonctl.validate_and_canonicalize_plist("trader", overflow_xml, log_dir=log_dir)


def test_read_plist_and_log_reject_symlinks_ancestors_hardlinks_and_wrong_uid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    src_dir = root / "src"
    src_dir.mkdir()
    real_file = src_dir / "real.plist"
    real_file.write_bytes(plistlib.dumps(_valid_plist_dict("trader"), fmt=plistlib.FMT_XML))

    # Direct symlink plist rejected
    sym_plist = src_dir / "com.agytrader.trader.plist"
    sym_plist.symlink_to(real_file)
    with pytest.raises(daemonctl.PlistValidationError, match="Cannot safely open plist"):
        daemonctl.read_plist_nosymlink(sym_plist)

    # Ancestor directory symlink rejected
    sym_ancestor = root / "sym_dir"
    sym_ancestor.symlink_to(src_dir)
    with pytest.raises(daemonctl.PlistValidationError, match="ancestor directory must not be a symlink"):
        daemonctl.read_plist_nosymlink(sym_ancestor / "real.plist")

    # Hardlink (st_nlink > 1) rejected
    hardlink_plist = src_dir / "hardlink.plist"
    os.link(real_file, hardlink_plist)
    with pytest.raises(daemonctl.PlistValidationError, match="link count 1"):
        daemonctl.read_plist_nosymlink(hardlink_plist)

    # Wrong owner st_uid when running as euid == 0 rejected
    single_plist = src_dir / "single.plist"
    single_plist.write_bytes(plistlib.dumps(_valid_plist_dict("trader"), fmt=plistlib.FMT_XML))
    real_fstat = daemonctl.os.fstat

    def fake_fstat_root_owned(fd: int) -> Any:
        st = real_fstat(fd)
        return SimpleNamespace(
            st_mode=st.st_mode,
            st_nlink=st.st_nlink,
            st_uid=0,
            st_size=st.st_size,
        )

    monkeypatch.setattr(daemonctl.os, "geteuid", lambda: 0)
    monkeypatch.setattr(daemonctl.os, "fstat", fake_fstat_root_owned)
    monkeypatch.setattr(
        daemonctl.pwd, "getpwnam", lambda _user: SimpleNamespace(pw_uid=501)
    )
    with pytest.raises(daemonctl.PlistValidationError, match=f"must be owned by {TEST_RUN_USER}"):
        daemonctl.read_plist_nosymlink(single_plist)
    monkeypatch.undo()
    monkeypatch.setattr(daemonctl, "RUN_USER", TEST_RUN_USER)
    monkeypatch.setattr(daemonctl, "RUN_GROUP", "staff")
    monkeypatch.setattr(daemonctl, "REPO_ROOT", TEST_REPO_ROOT)
    monkeypatch.setattr(daemonctl, "PLIST_SRC_DIR", TEST_REPO_ROOT / "deploy" / "macos")
    monkeypatch.setattr(daemonctl, "VENV_AGY_OPS", TEST_VENV_AGY_OPS)

    # Symlinked target log file rejected
    log_dir = root / "logs"
    log_dir.mkdir()
    secret_target = root / "sudo"
    secret_target.write_text("do not overwrite", encoding="utf-8")
    (log_dir / "trader.stdout.log").symlink_to(secret_target)
    raw = plistlib.dumps(_valid_plist_dict("trader", log_dir=log_dir), fmt=plistlib.FMT_XML)
    with pytest.raises(daemonctl.PlistValidationError, match="symlink"):
        daemonctl.validate_and_canonicalize_plist("trader", raw, log_dir=log_dir)

    # Symlinked ancestor of log_dir rejected
    nested_under_sym = sym_ancestor / "nested_logs"
    raw_sym_anc = plistlib.dumps(
        _valid_plist_dict("trader", log_dir=nested_under_sym), fmt=plistlib.FMT_XML
    )
    with pytest.raises(daemonctl.PlistValidationError, match="ancestor must not be a symlink"):
        daemonctl.validate_and_canonicalize_plist("trader", raw_sym_anc, log_dir=nested_under_sym)

    # Non-root-owned log_dir rejected when euid == 0
    clean_log_dir = root / "clean_logs"
    clean_log_dir.mkdir()
    raw_clean = plistlib.dumps(
        _valid_plist_dict("trader", log_dir=clean_log_dir), fmt=plistlib.FMT_XML
    )
    monkeypatch.setattr(daemonctl.os, "geteuid", lambda: 0)
    with pytest.raises(daemonctl.PlistValidationError, match="must be owned by uid 0"):
        daemonctl.validate_and_canonicalize_plist("trader", raw_clean, log_dir=clean_log_dir)


def test_daemonctl_sync_start_stop_restart_and_sanitized_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    src_dir = root / "src"
    target_dir = root / "launchdaemons"
    log_dir = root / "logs"
    src_dir.mkdir()
    target_dir.mkdir()
    log_dir.mkdir()

    src_plist = src_dir / "com.agytrader.trader.plist"
    src_plist.write_bytes(
        plistlib.dumps(_valid_plist_dict("trader", log_dir=log_dir), fmt=plistlib.FMT_XML)
    )

    state: dict[str, Any] = {"loaded": False, "pid": None}
    calls: list[list[str]] = []
    envs: list[dict[str, str] | None] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        envs.append(kwargs.get("env"))
        if cmd[:2] == [daemonctl.PLUTIL, "-lint"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="OK\n", stderr="")
        if cmd[:2] == [daemonctl.LAUNCHCTL, "print"]:
            if not state["loaded"]:
                return subprocess.CompletedProcess(cmd, 113, stdout="", stderr="Could not find service")
            pid_line = f"\tpid = {state['pid']}\n" if state["pid"] is not None else ""
            run_state = "running" if state["pid"] is not None else "waiting"
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=f"system/com.agytrader.trader = {{\n\tstate = {run_state}\n{pid_line}\tlast exit code = 0\n}}\n",
                stderr="",
            )
        if cmd[:2] == [daemonctl.LAUNCHCTL, "enable"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:3] == [daemonctl.LAUNCHCTL, "bootstrap", "system"]:
            state["loaded"] = True
            state["pid"] = 4242
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == [daemonctl.LAUNCHCTL, "bootout"]:
            state["loaded"] = False
            state["pid"] = None
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == [daemonctl.LAUNCHCTL, "kickstart"]:
            state["pid"] = 5151
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(daemonctl.subprocess, "run", fake_run)

    sync_res = daemonctl.sync_service_plist(
        "trader", plist_src_dir=src_dir, target_dir=target_dir, log_dir=log_dir
    )
    assert sync_res["outcome"] == "synced"
    assert sync_res["loaded"] is True
    assert sync_res["pid"] == 4242
    assert (target_dir / "com.agytrader.trader.plist").exists()
    assert any(c[:2] == [daemonctl.PLUTIL, "-lint"] for c in calls)
    assert [daemonctl.LAUNCHCTL, "enable", "system/com.agytrader.trader"] in calls
    # Every subprocess call must receive a sanitized root-safe environment
    assert all(
        env == {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": "/var/root", "LANG": "C.UTF-8"}
        for env in envs
    )

    restart_res = daemonctl.restart_service("trader", target_dir=target_dir)
    assert restart_res["outcome"] == "restarted"
    assert restart_res["pid"] == 5151
    assert [daemonctl.LAUNCHCTL, "kickstart", "-k", "system/com.agytrader.trader"] in calls

    stop_res = daemonctl.stop_service("trader", target_dir=target_dir)
    assert stop_res["outcome"] == "stopped"
    assert stop_res["loaded"] is False
    assert stop_res["pid"] is None

    start_res = daemonctl.start_service("trader", target_dir=target_dir)
    assert start_res["outcome"] == "bootstrapped"
    assert start_res["loaded"] is True


@pytest.mark.parametrize(
    "action",
    ["status", "start", "stop", "sync-plists"],
)
def test_agy_ops_service_commands_invoke_sudo_noninteractive(
    monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    recorded_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded_cmds.append(cmd)
        payload = {
            "action": action,
            "target": "trader",
            "services": [
                {
                    "service": "trader",
                    "label": "com.agytrader.trader",
                    "loaded": action != "stop",
                    "state": "running" if action != "stop" else None,
                    "pid": 9001 if action != "stop" else None,
                    "last_exit_status": 0,
                    "plist_installed": True,
                    **({"outcome": action} if action != "status" else {}),
                }
            ],
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("agy_ops.cli.subprocess.run", fake_run)

    res = runner.invoke(app, ["service", action, "trader", "--json"])
    assert res.exit_code == 0, res.output
    assert recorded_cmds == [
        ["/usr/bin/sudo", "-n", "/usr/local/sbin/agy-daemonctl", action, "trader", "--json"]
    ]
    parsed = json.loads(res.stdout)
    assert parsed["action"] == action
    assert parsed["services"][0]["service"] == "trader"


def test_agy_ops_service_missing_sudo_install_emits_clean_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="sudo: a password is required\n")

    monkeypatch.setattr("agy_ops.cli.subprocess.run", fake_run)

    res = runner.invoke(app, ["service", "status", "all", "--json"])
    assert res.exit_code == 1
    payload = json.loads(res.stdout)
    assert "install_launchdaemons.sh" in payload["error"]


def test_agy_ops_service_restart_enforces_control_plane_rate_limit_for_non_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    recorded_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded_cmds.append(cmd)
        payload = {
            "action": "restart",
            "target": "trader",
            "services": [
                {
                    "service": "trader",
                    "label": "com.agytrader.trader",
                    "loaded": True,
                    "state": "running",
                    "pid": 7777,
                    "last_exit_status": 0,
                    "plist_installed": True,
                    "outcome": "restarted",
                }
            ],
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("agy_ops.cli.subprocess.run", fake_run)

    # Empty reason rejected before invoking sudo
    res_no_reason = runner.invoke(
        app,
        ["service", "restart", "trader", "--actor", "ops", "--data-dir", str(data_dir), "--json"],
    )
    assert res_no_reason.exit_code == 1
    assert "reason" in json.loads(res_no_reason.stdout)["error"]
    assert recorded_cmds == []

    # 2 restarts allowed within 1 hour
    for idx in (1, 2):
        res_ok = runner.invoke(
            app,
            [
                "service",
                "restart",
                "trader",
                "--actor",
                "ops",
                "--reason",
                f"recover {idx}",
                "--data-dir",
                str(data_dir),
                "--json",
            ],
        )
        assert res_ok.exit_code == 0, res_ok.output

    assert len(recorded_cmds) == 2

    # 3rd restart blocked by ControlPlane rate limit without calling sudo
    res_blocked = runner.invoke(
        app,
        [
            "service",
            "restart",
            "trader",
            "--actor",
            "ops",
            "--reason",
            "recover 3",
            "--data-dir",
            str(data_dir),
            "--json",
        ],
    )
    assert res_blocked.exit_code == 1
    assert "Rate limit exceeded" in json.loads(res_blocked.stdout)["error"]
    assert len(recorded_cmds) == 2

    # Human operator bypasses ControlPlane rate-limit check
    res_operator = runner.invoke(
        app,
        [
            "service",
            "restart",
            "trader",
            "--actor",
            "operator",
            "--data-dir",
            str(data_dir),
            "--json",
        ],
    )
    assert res_operator.exit_code == 0, res_operator.output
    assert len(recorded_cmds) == 3
    assert recorded_cmds[-1] == [
        "/usr/bin/sudo",
        "-n",
        "/usr/local/sbin/agy-daemonctl",
        "restart",
        "trader",
        "--json",
    ]
