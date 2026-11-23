from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from dotenv import dotenv_values

from agy_ops.notifications.discord import DiscordWebhookNotifier

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_CONFIG_DIR = REPO_ROOT / "config"


def _live_discord_webhook_url() -> str:
    """Return the operator's live DISCORD_WEBHOOK_URL from the repo .env (never logged)."""
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return ""
    try:
        value = dotenv_values(env_path).get("DISCORD_WEBHOOK_URL") or ""
    except (OSError, UnicodeDecodeError):
        return ""
    return value.strip()


_LIVE_DISCORD_WEBHOOK_URL = _live_discord_webhook_url()


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
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)


@pytest.fixture(autouse=True)
def _forbid_live_discord_webhook(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hard-fail any test that tries to build a notifier for the live Discord webhook.

    Belt-and-braces on top of .env isolation: even if the live URL leaks into settings by
    another path, no test may construct a DiscordWebhookNotifier pointed at it. Tests using
    sample webhook URLs (mocked via respx) are unaffected.
    """
    if not _LIVE_DISCORD_WEBHOOK_URL:
        return
    original_init = DiscordWebhookNotifier.__init__

    def _guarded_init(self: DiscordWebhookNotifier, webhook_url: str, *args: Any, **kwargs: Any) -> None:
        if str(webhook_url).strip() == _LIVE_DISCORD_WEBHOOK_URL:
            raise RuntimeError(
                "Test attempted to use the LIVE Discord webhook from the repo .env; "
                "inject a fake notifier or a sample webhook URL instead."
            )
        original_init(self, webhook_url, *args, **kwargs)

    monkeypatch.setattr(DiscordWebhookNotifier, "__init__", _guarded_init)
