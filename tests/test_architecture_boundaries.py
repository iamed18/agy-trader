from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AGY_CORE_ROOT = REPO_ROOT / "packages" / "agy-core"
AGY_OPS_ROOT = REPO_ROOT / "packages" / "agy-ops"
DASHBOARD_ROOT = AGY_OPS_ROOT / "src" / "agy_ops" / "dashboard"

FORBIDDEN_CORE_TOP_LEVEL_IMPORTS = frozenset({"agy_ops", "fastapi", "uvicorn", "starlette"})
FORBIDDEN_DASHBOARD_HTTP_METHODS = frozenset(
    {"post", "put", "delete", "patch", "options", "head", "trace", "websocket"}
)
SQL_WRITE_PREFIXES = (
    "insert ",
    "update ",
    "delete ",
    "replace ",
    "create ",
    "drop ",
    "alter ",
    "vacuum",
)
STATEDB_WRITE_METHOD_PREFIXES = (
    "upsert_",
    "insert_",
    "update_",
    "replace_",
    "record_daemon_event",
    "save_backtest_run",
    "log_llm_usage",
    "try_record_service_restart",
)


def _iter_python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def find_forbidden_core_imports(source: str, filename: str = "<string>") -> list[str]:
    """Return violations where core code imports ops or web-framework packages."""
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in FORBIDDEN_CORE_TOP_LEVEL_IMPORTS:
                    violations.append(f"{filename}:{node.lineno}: forbidden import {alias.name!r}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top in FORBIDDEN_CORE_TOP_LEVEL_IMPORTS:
                violations.append(
                    f"{filename}:{node.lineno}: forbidden import from {node.module!r}"
                )
    return violations


def _extract_str_constant(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def find_dashboard_violations(source: str, filename: str = "<string>") -> list[str]:
    """Return architectural violations in read-only dashboard modules."""
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "alpaca_broker" in alias.name or alias.name.endswith("AlpacaBroker"):
                    violations.append(
                        f"{filename}:{node.lineno}: forbidden broker import {alias.name!r}"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                if alias.name == "AlpacaBroker" or "alpaca_broker" in mod:
                    violations.append(
                        f"{filename}:{node.lineno}: forbidden AlpacaBroker import from {mod!r}"
                    )
        elif isinstance(node, ast.Name) and node.id == "AlpacaBroker":
            violations.append(f"{filename}:{node.lineno}: forbidden reference to AlpacaBroker")
        elif isinstance(node, ast.Call):
            func = node.func
            attr_name = func.attr if isinstance(func, ast.Attribute) else None
            func_name = func.id if isinstance(func, ast.Name) else None

            # 1. Non-GET FastAPI / APIRouter route registrations
            if attr_name in FORBIDDEN_DASHBOARD_HTTP_METHODS:
                violations.append(
                    f"{filename}:{node.lineno}: non-GET FastAPI route method .{attr_name}()"
                )
            elif attr_name in {"api_route", "add_api_route"}:
                methods_kw = next((kw for kw in node.keywords if kw.arg == "methods"), None)
                if methods_kw is None:
                    violations.append(
                        f"{filename}:{node.lineno}: .{attr_name}() must explicitly specify methods=['GET']"
                    )
                elif isinstance(methods_kw.value, (ast.List, ast.Tuple, ast.Set)):
                    methods = {
                        str(elt.value).upper()
                        for elt in methods_kw.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    }
                    if not methods or methods != {"GET"}:
                        violations.append(
                            f"{filename}:{node.lineno}: .{attr_name}() has non-GET methods {sorted(methods)}"
                        )
                else:
                    violations.append(
                        f"{filename}:{node.lineno}: .{attr_name}() methods must be literal ['GET']"
                    )

            # 2. StateDB must always be opened with read_only=True
            if func_name == "StateDB" or attr_name == "StateDB":
                ro_kw = next((kw for kw in node.keywords if kw.arg == "read_only"), None)
                if (
                    ro_kw is None
                    or not isinstance(ro_kw.value, ast.Constant)
                    or ro_kw.value.value is not True
                ):
                    violations.append(
                        f"{filename}:{node.lineno}: StateDB must be constructed with read_only=True"
                    )

            # 3. Direct SQLite writes or StateDB write helpers
            if attr_name is not None and attr_name.startswith(STATEDB_WRITE_METHOD_PREFIXES):
                violations.append(
                    f"{filename}:{node.lineno}: forbidden SQLite write call .{attr_name}()"
                )
            if attr_name in {"execute", "executemany", "executescript"} and node.args:
                sql_str = _extract_str_constant(node.args[0])
                if sql_str is not None:
                    normalized = " ".join(sql_str.strip().lower().split())
                    if any(normalized.startswith(prefix) for prefix in SQL_WRITE_PREFIXES):
                        violations.append(
                            f"{filename}:{node.lineno}: forbidden SQL write statement in .{attr_name}()"
                        )
            if (
                isinstance(func, ast.Attribute)
                and attr_name == "connect"
                and isinstance(func.value, ast.Name)
                and func.value.id == "sqlite3"
            ):
                uri_kw = next((kw for kw in node.keywords if kw.arg == "uri"), None)
                if (
                    uri_kw is None
                    or not isinstance(uri_kw.value, ast.Constant)
                    or uri_kw.value.value is not True
                ):
                    violations.append(
                        f"{filename}:{node.lineno}: direct sqlite3.connect must use uri=True with mode=ro"
                    )

    return violations


def test_workspace_packages_importable_from_expected_roots() -> None:
    agy_trader = importlib.import_module("agy_trader")
    agy_ops = importlib.import_module("agy_ops")

    assert agy_trader.__file__ is not None
    assert agy_ops.__file__ is not None
    assert Path(agy_trader.__file__).resolve().is_relative_to(AGY_CORE_ROOT.resolve())
    assert Path(agy_ops.__file__).resolve().is_relative_to(AGY_OPS_ROOT.resolve())


def test_agy_core_has_no_ops_or_web_imports() -> None:
    core_files = _iter_python_files(AGY_CORE_ROOT / "src")
    assert len(core_files) > 0

    violations: list[str] = []
    for py_file in core_files:
        source = py_file.read_text(encoding="utf-8")
        violations.extend(
            find_forbidden_core_imports(source, str(py_file.relative_to(REPO_ROOT)))
        )

    assert violations == [], "Architectural boundary violations in agy-core:\n" + "\n".join(
        violations
    )


def test_dashboard_is_strictly_read_only_and_broker_free() -> None:
    dashboard_files = _iter_python_files(DASHBOARD_ROOT)
    violations: list[str] = []
    for py_file in dashboard_files:
        source = py_file.read_text(encoding="utf-8")
        violations.extend(
            find_dashboard_violations(source, str(py_file.relative_to(REPO_ROOT)))
        )

    assert violations == [], "Dashboard boundary violations:\n" + "\n".join(violations)


@pytest.mark.parametrize(
    ("bad_source", "expected_fragment"),
    [
        ("from agy_ops.cli import app\n", "agy_ops"),
        ("import fastapi\n", "fastapi"),
        ("from uvicorn import run\n", "uvicorn"),
        ("import starlette.responses\n", "starlette"),
    ],
)
def test_core_import_guard_catches_forbidden_imports(
    bad_source: str, expected_fragment: str
) -> None:
    violations = find_forbidden_core_imports(bad_source)
    assert len(violations) == 1
    assert expected_fragment in violations[0]


@pytest.mark.parametrize(
    ("bad_source", "expected_fragment"),
    [
        ("from agy_trader.execution.alpaca_broker import AlpacaBroker\n", "AlpacaBroker"),
        ("db = StateDB(path)\n", "read_only=True"),
        ("db.upsert_live_order()\n", "upsert_live_order"),
        ("conn.execute('INSERT INTO live_orders VALUES (1)')\n", "SQL write"),
        ("@app.post('/api/order')\ndef x(): pass\n", ".post()"),
        ("@app.api_route('/api/x', methods=['GET', 'POST'])\ndef x(): pass\n", "non-GET"),
    ],
)
def test_dashboard_guard_catches_mutations_and_broker(
    bad_source: str, expected_fragment: str
) -> None:
    violations = find_dashboard_violations(bad_source)
    assert any(expected_fragment in v for v in violations), violations
