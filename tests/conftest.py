from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture(autouse=True)
def _isolate_tests_from_live_repo_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolate tests from the operator's live repo-root .env and working directory."""
    iso_cwd = tmp_path / "_isolated_cwd"
    iso_cwd.mkdir(exist_ok=True)
    monkeypatch.chdir(iso_cwd)
    monkeypatch.setenv("AGY_CONFIG_DIR", str(REPO_CONFIG_DIR))
    monkeypatch.delenv("BLACKOUT_PERIOD", raising=False)
