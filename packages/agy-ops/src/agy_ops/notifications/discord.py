from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import Any, Self
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

ALLOWED_DISCORD_HOSTS: frozenset[str] = frozenset(
    {
        "discord.com",
        "discordapp.com",
        "ptb.discord.com",
        "canary.discord.com",
    }
)

DISCORD_MAX_TITLE_LEN = 256
DISCORD_MAX_DESCRIPTION_LEN = 4096
DISCORD_MAX_FIELDS = 25
DISCORD_MAX_FIELD_NAME_LEN = 256
DISCORD_MAX_FIELD_VALUE_LEN = 1024
DISCORD_MAX_FOOTER_LEN = 2048
DISCORD_MAX_TOTAL_EMBED_CHARS = 6000
MAX_RETRY_DELAY_SECONDS = 10.0

COLOR_GREEN = 0x22C55E
COLOR_AMBER = 0xF59E0B
COLOR_RED = 0xEF4444
COLOR_BLUE = 0x3B82F6

DEFAULT_DISCORD_FOOTER = "agy-trader"
DEFAULT_DISCORD_CHANNEL_LABEL = "Discord alerts"

_WEBHOOK_PATH_RE = re.compile(r"^/api/webhooks/(\d+)/([A-Za-z0-9_-]+)$")
_WEBHOOK_REDACT_RE = re.compile(r"(/webhooks/[^/\s]+/)([^/?#\s]+)", re.IGNORECASE)
_WEBHOOK_URL_REDACT_RE = re.compile(r"(/webhooks/[^/?#]+/)([^?#]+)", re.IGNORECASE)
_USERINFO_REDACT_RE = re.compile(r"(://[^/:@\s]+:)[^/@\s]+(@)")
_VALID_PNG_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.png$")


def redact_webhook_url(url: str) -> str:
    """Mask the secret token segment in a Discord webhook URL."""
    stripped = url.strip()
    if not stripped:
        return ""
    masked = _USERINFO_REDACT_RE.sub(r"\1***\2", stripped)
    return _WEBHOOK_URL_REDACT_RE.sub(r"\1***", masked)


def _sanitize_error_text(text: str, raw_url: str) -> str:
    redacted_url = redact_webhook_url(raw_url)
    sanitized = _WEBHOOK_REDACT_RE.sub(r"\1***", text)
    if raw_url:
        sanitized = sanitized.replace(raw_url, redacted_url)
        parsed = urlsplit(raw_url)
        match = _WEBHOOK_PATH_RE.fullmatch(parsed.path)
        if match:
            token = match.group(2)
            if token:
                sanitized = sanitized.replace(token, "***")
    return sanitized


def validate_discord_webhook_url(url: str) -> str:
    """Validate that url is an HTTPS Discord webhook URL with numeric ID and token."""
    cleaned = url.strip()
    if not cleaned:
        raise ValueError("Discord webhook URL must not be empty")

    redacted = redact_webhook_url(cleaned)
    if any(ord(ch) < 32 or ord(ch) == 127 or ch.isspace() for ch in cleaned):
        raise ValueError(
            "Discord webhook URL must not contain whitespace or control characters: "
            f"{redacted}"
        )

    parsed = urlsplit(cleaned)

    if parsed.scheme != "https":
        raise ValueError(f"Discord webhook URL must use https:// scheme: {redacted}")

    if parsed.username is not None or parsed.password is not None or parsed.port is not None:
        raise ValueError(f"Discord webhook URL must not include credentials or port: {redacted}")

    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_DISCORD_HOSTS or parsed.netloc.lower() != host:
        raise ValueError(f"Discord webhook host {host!r} is not allowed: {redacted}")

    if parsed.query or parsed.fragment or "?" in cleaned or "#" in cleaned:
        raise ValueError(
            f"Discord webhook URL must not contain query parameters or fragments: {redacted}"
        )

    match = _WEBHOOK_PATH_RE.fullmatch(parsed.path)
    if match is None:
        raise ValueError(
            "Discord webhook URL path must match /api/webhooks/<digits>/<token>: "
            f"{redacted}"
        )

    webhook_id, token = match.group(1), match.group(2)
    return f"https://{host}/api/webhooks/{webhook_id}/{token}"


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def _embed_char_count(
    title: str,
    description: str,
    fields: list[dict[str, Any]],
    footer: str,
) -> int:
    field_chars = sum(len(f["name"]) + len(f["value"]) for f in fields)
    return len(title) + len(description) + len(footer) + field_chars


def normalize_discord_embed(
    *,
    title: str,
    description: str,
    color: int,
    fields: Sequence[Mapping[str, Any]] = (),
    footer: str = DEFAULT_DISCORD_FOOTER,
    image_url: str | None = None,
) -> dict[str, Any]:
    """Normalize and truncate embed fields to strictly satisfy Discord API limits."""
    norm_title = _truncate(str(title), DISCORD_MAX_TITLE_LEN)
    norm_desc = _truncate(str(description), DISCORD_MAX_DESCRIPTION_LEN)
    norm_footer = _truncate(str(footer), DISCORD_MAX_FOOTER_LEN)

    norm_fields: list[dict[str, Any]] = []
    for raw_field in list(fields)[:DISCORD_MAX_FIELDS]:
        raw_name = str(raw_field.get("name", "")).strip() or "-"
        raw_value = str(raw_field.get("value", "")).strip() or "-"
        norm_fields.append(
            {
                "name": _truncate(raw_name, DISCORD_MAX_FIELD_NAME_LEN),
                "value": _truncate(raw_value, DISCORD_MAX_FIELD_VALUE_LEN),
                "inline": bool(raw_field.get("inline", False)),
            }
        )

    excess = _embed_char_count(norm_title, norm_desc, norm_fields, norm_footer) - (
        DISCORD_MAX_TOTAL_EMBED_CHARS
    )
    if excess > 0 and norm_desc:
        target_desc_len = max(0, len(norm_desc) - excess)
        norm_desc = _truncate(norm_desc, target_desc_len)
        excess = _embed_char_count(norm_title, norm_desc, norm_fields, norm_footer) - (
            DISCORD_MAX_TOTAL_EMBED_CHARS
        )

    while excess > 0 and norm_fields:
        last = norm_fields[-1]
        if len(last["value"]) > 1:
            reducible = len(last["value"]) - 1
            if reducible >= excess:
                last["value"] = _truncate(last["value"], len(last["value"]) - excess)
            else:
                norm_fields.pop()
        else:
            norm_fields.pop()
        excess = _embed_char_count(norm_title, norm_desc, norm_fields, norm_footer) - (
            DISCORD_MAX_TOTAL_EMBED_CHARS
        )

    if excess > 0 and norm_footer:
        norm_footer = _truncate(norm_footer, max(0, len(norm_footer) - excess))
        excess = _embed_char_count(norm_title, norm_desc, norm_fields, norm_footer) - (
            DISCORD_MAX_TOTAL_EMBED_CHARS
        )

    if excess > 0 and norm_title:
        norm_title = _truncate(norm_title, max(0, len(norm_title) - excess))

    embed: dict[str, Any] = {
        "title": norm_title,
        "description": norm_desc,
        "color": int(color),
    }
    if norm_fields:
        embed["fields"] = norm_fields
    if norm_footer:
        embed["footer"] = {"text": norm_footer}
    if image_url is not None and str(image_url).strip():
        embed["image"] = {"url": str(image_url).strip()}
    return embed


class DiscordWebhookNotifier:
    """HTTP client for posting rich embeds to a Discord webhook with retries & redaction."""

    def __init__(
        self,
        webhook_url: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 15.0,
        max_retries: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
        username: str = "agy-trader",
    ) -> None:
        if timeout <= 0.0:
            raise ValueError("timeout must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self.webhook_url = validate_discord_webhook_url(webhook_url)
        self.redacted_url = redact_webhook_url(self.webhook_url)
        self._external_client = client
        self._owned_client: httpx.Client | None = None
        self._timeout = float(timeout)
        self._max_retries = int(max_retries)
        self._sleep_fn = sleep_fn
        self.username = username

    def __enter__(self) -> Self:
        if self._external_client is None and self._owned_client is None:
            self._owned_client = httpx.Client(timeout=self._timeout, follow_redirects=False)
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the internally owned httpx.Client if open."""
        if self._owned_client is not None:
            self._owned_client.close()
            self._owned_client = None

    def _compute_retry_delay(self, response: httpx.Response, attempt: int) -> float:
        if response.status_code == 429:
            header_val = response.headers.get("Retry-After")
            if header_val is not None:
                try:
                    parsed = float(header_val.strip())
                    if parsed >= 0.0:
                        return min(MAX_RETRY_DELAY_SECONDS, parsed)
                except ValueError:
                    pass
            try:
                body = response.json()
            except ValueError:
                body = None
            if isinstance(body, dict) and "retry_after" in body:
                try:
                    parsed = float(body["retry_after"])
                    if parsed >= 0.0:
                        return min(MAX_RETRY_DELAY_SECONDS, parsed)
                except (TypeError, ValueError):
                    pass

        return min(MAX_RETRY_DELAY_SECONDS, float(2 ** min(attempt, 10)))

    def _post_with_retries(
        self,
        client: httpx.Client,
        payload: dict[str, Any],
        *,
        image_bytes: bytes | None = None,
        image_filename: str = "cumulative_performance.png",
    ) -> httpx.Response:
        if image_bytes is not None and _VALID_PNG_FILENAME_RE.fullmatch(image_filename) is None:
            raise ValueError(f"Invalid PNG attachment filename: {image_filename!r}")
        for attempt in range(self._max_retries + 1):
            try:
                if image_bytes is None:
                    response = client.post(self.webhook_url, json=payload, follow_redirects=False)
                else:
                    response = client.post(
                        self.webhook_url,
                        data={"payload_json": json.dumps(payload)},
                        files={"files[0]": (image_filename, image_bytes, "image/png")},
                        follow_redirects=False,
                    )
            except httpx.RequestError as exc:
                safe_detail = _sanitize_error_text(str(exc), self.webhook_url)
                raise RuntimeError(
                    f"Discord webhook request failed for {self.redacted_url}: "
                    f"{type(exc).__name__}: {safe_detail}"
                ) from None

            status = response.status_code
            is_retryable = status == 429 or 500 <= status < 600
            if not is_retryable:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    raise RuntimeError(
                        f"Discord webhook returned HTTP {status} for {self.redacted_url}"
                    ) from None
                return response

            if attempt >= self._max_retries:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    raise RuntimeError(
                        f"Discord webhook returned HTTP {status} after {self._max_retries + 1} "
                        f"attempt(s) for {self.redacted_url}"
                    ) from None

            delay = self._compute_retry_delay(response, attempt)
            logger.warning(
                "Discord webhook %s returned HTTP %s (attempt %d/%d); retrying in %.2fs",
                self.redacted_url,
                status,
                attempt + 1,
                self._max_retries + 1,
                delay,
            )
            self._sleep_fn(delay)

        raise AssertionError("Unreachable")

    def send_embed(
        self,
        *,
        title: str,
        description: str,
        color: int,
        fields: Sequence[dict[str, Any]] = (),
        footer: str = DEFAULT_DISCORD_FOOTER,
        image_url: str | None = None,
        image_bytes: bytes | None = None,
        image_filename: str = "cumulative_performance.png",
    ) -> dict[str, Any]:
        """Post a single Discord embed to the configured webhook URL."""
        if image_bytes is not None and _VALID_PNG_FILENAME_RE.fullmatch(image_filename) is None:
            raise ValueError(f"Invalid PNG attachment filename: {image_filename!r}")
        resolved_image_url = (
            f"attachment://{image_filename}"
            if (image_bytes is not None and image_url is None)
            else image_url
        )
        embed = normalize_discord_embed(
            title=title,
            description=description,
            color=color,
            fields=fields,
            footer=footer,
            image_url=resolved_image_url,
        )
        payload: dict[str, Any] = {
            "username": self.username,
            "embeds": [embed],
        }
        if image_bytes is not None:
            payload["attachments"] = [{"id": 0, "filename": image_filename}]

        active_client = self._external_client or self._owned_client
        cm = (
            nullcontext(active_client)
            if active_client is not None
            else httpx.Client(timeout=self._timeout, follow_redirects=False)
        )
        with cm as client:
            response = self._post_with_retries(
                client,
                payload,
                image_bytes=image_bytes,
                image_filename=image_filename,
            )

        response_data: Any = None
        if response.content:
            try:
                response_data = response.json()
            except ValueError:
                response_data = None

        return {
            "ok": True,
            "status_code": response.status_code,
            "webhook_url": self.redacted_url,
            "payload": payload,
            "response": response_data,
        }


def _format_headline_list(
    items: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    limit: int = 2,
) -> list[str]:
    lines: list[str] = []
    for item in list(items)[:limit]:
        syms = item.get("symbols") or []
        sym_str = ",".join(str(s) for s in syms[:3]) if syms else str(item.get("symbol", "") or "-")
        headline = _truncate(str(item.get("headline", "") or "").strip(), 90)
        if headline:
            lines.append(f"- [{prefix}:{sym_str}] {headline}")
    return lines


def build_morning_discord_embed(
    readiness: Mapping[str, Any],
    morning_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a Discord embed specification from readiness and morning plan reports."""
    status = str(readiness.get("status", "CRITICAL")).upper()
    session_date = str(
        morning_plan.get("session_date_et") or readiness.get("session_date_et") or "N/A"
    )

    if status == "READY":
        color = COLOR_GREEN
        title = f"AGY Morning Readiness & Plan ({session_date})"
    elif status == "DEGRADED":
        color = COLOR_AMBER
        title = f"[ALERT] AGY Morning Readiness & Plan ({session_date})"
    else:
        color = COLOR_RED
        title = f"[ALERT] AGY Morning Readiness & Plan ({session_date})"

    equity = float(morning_plan.get("equity", 0.0))
    drawdown_pct = float(morning_plan.get("drawdown_pct", 0.0))
    description = (
        f"Status: **{status}** | Equity: **${equity:,.2f}** | Drawdown: **{drawdown_pct:.2f}%**"
    )

    issues = list(readiness.get("issues", ()))
    checks = list(readiness.get("checks", ()))
    if issues:
        issue_lines = [
            f"- [{str(i.get('severity', '')).upper()}] {i.get('check', '?')}: {i.get('message', '')}"
            for i in issues[:5]
        ]
        if len(issues) > 5:
            issue_lines.append(f"- (+{len(issues) - 5} more issue(s))")
        readiness_value = "\n".join(issue_lines)
    else:
        readiness_value = f"READY ({len(checks)} checks passed, 0 issues)"

    positions = list(morning_plan.get("positions", ()))
    pos_lines = [
        f"Equity: ${equity:,.2f} (DD {drawdown_pct:.2f}%) | Open Positions: {len(positions)}"
    ]
    for pos in positions[:8]:
        stop_price = pos.get("stop_price")
        stop_str = f"${float(stop_price):,.2f}" if stop_price is not None else "MISSING"
        pos_lines.append(
            f"- **{pos.get('symbol', '?')}**: qty={pos.get('qty', 0)}, "
            f"mark=${float(pos.get('market_price', 0.0)):,.2f}, stop={stop_str}"
        )
    if len(positions) > 8:
        pos_lines.append(f"- (+{len(positions) - 8} more position(s))")

    dials = morning_plan.get("control_dials") or {}
    macro_scale = float(morning_plan.get("macro_regime_scale", 1.0))
    macro_value = (
        f"Macro Regime Scale: `{macro_scale:.4f}`\n"
        f"Dials: paused=`{bool(dials.get('pause_new_entries', False))}`, "
        f"multiplier=`{float(dials.get('exposure_multiplier', 1.0)):.2f}`, "
        f"allow_shorting=`{bool(dials.get('allow_shorting', True))}`"
    )
    if dials.get("reason"):
        macro_value += f"\nReason: {dials['reason']}"

    earnings = list(morning_plan.get("earnings_blackouts", ()))
    restatements = list(morning_plan.get("restatement_blackouts", ()))
    earn_days = morning_plan.get("earnings_blackout_days")
    if "earnings_blackout_days" in morning_plan and (
        earn_days is None or (isinstance(earn_days, int) and earn_days < 0)
    ):
        earnings_str = "Disabled"
    else:
        earnings_str = (
            ", ".join(f"{e.get('symbol')} ({e.get('earnings_date')})" for e in earnings[:10])
            or "None"
        )
        if len(earnings) > 10:
            earnings_str += f" (+{len(earnings) - 10} more)"
    restate_str = ", ".join(str(r.get("symbol")) for r in restatements[:10]) or "None"
    blackouts_value = (
        f"Earnings Blackouts: {earnings_str}\n"
        f"Item 4.02 Restatement Blackouts: {restate_str}"
    )

    model_news = list(morning_plan.get("overnight_model_news", ()))
    sec_filings = list(morning_plan.get("overnight_sec_filings", ()))
    fwd_feeds = list(morning_plan.get("overnight_forward_feeds", ()))
    highlight_lines = [
        (
            f"Counts (24h): {len(model_news)} model news, {len(sec_filings)} SEC filings, "
            f"{len(fwd_feeds)} RSS/YF feeds"
        )
    ]
    highlight_lines.extend(_format_headline_list(model_news, prefix="News", limit=2))
    highlight_lines.extend(_format_headline_list(fwd_feeds, prefix="RSS", limit=1))
    for ev in sec_filings[:2]:
        items_str = ",".join(str(x) for x in (ev.get("items") or [])) or "-"
        highlight_lines.append(
            f"- [SEC:{ev.get('symbol', '?')}] Form {ev.get('form_type', '?')} (items={items_str})"
        )

    raw_fields = [
        {"name": "Readiness Status & Issues", "value": readiness_value, "inline": False},
        {
            "name": "Portfolio Equity / Positions / Stops",
            "value": "\n".join(pos_lines),
            "inline": False,
        },
        {"name": "Macro Regime Scale & Dials", "value": macro_value, "inline": False},
        {
            "name": "Today's Earnings / Item-4.02 Blackouts",
            "value": blackouts_value,
            "inline": False,
        },
        {
            "name": "Overnight News / SEC / RSS Highlights",
            "value": "\n".join(highlight_lines),
            "inline": False,
        },
    ]
    blackout = morning_plan.get("blackout") or {}
    if blackout.get("active"):
        if blackout.get("halted"):
            bo_val = (
                f"HALTED (fail-closed): {blackout.get('halt_reason') or 'invalid BLACKOUT_PERIOD'}"
            )
            color = COLOR_RED
        else:
            frozen_str = ", ".join(blackout.get("frozen_symbols") or ()) or "None"
            etf_str = "; ".join(blackout.get("etf_weights_summary") or ()) or "None"
            bo_val = f"Frozen: `{frozen_str}`\nETF Exposure: {etf_str}"
            if blackout.get("warnings"):
                bo_val += f"\nWarnings: {'; '.join(blackout['warnings'])}"
        raw_fields.append(
            {"name": "BLACKOUT_PERIOD Compliance Rail", "value": bo_val, "inline": False}
        )
    norm = normalize_discord_embed(
        title=title,
        description=description,
        color=color,
        fields=raw_fields,
        footer=DEFAULT_DISCORD_FOOTER,
    )
    return {
        "title": norm["title"],
        "description": norm["description"],
        "color": norm["color"],
        "fields": norm.get("fields", []),
        "footer": str((norm.get("footer") or {}).get("text", DEFAULT_DISCORD_FOOTER)),
    }


def build_daily_summary_discord_embed(
    summary: Mapping[str, Any],
    *,
    image_url: str | None = None,
) -> dict[str, Any]:
    """Build a Discord embed specification from a daily close summary report."""
    session_date = str(summary.get("session_date_et") or "N/A")
    equity = float(summary.get("equity", 0.0))
    daily_ret = float(summary.get("daily_return_pct", 0.0))
    cum_ret = float(summary.get("cumulative_return_pct", 0.0))
    spy_daily = summary.get("spy_daily_return_pct")
    ew_daily = summary.get("ew50_daily_return_pct")
    spmo_daily = summary.get("spmo_daily_return_pct")
    qmom_daily = summary.get("qmom_daily_return_pct")
    spy_cum = summary.get("spy_cumulative_return_pct")
    ew_cum = summary.get("ew50_cumulative_return_pct")
    spmo_cum = summary.get("spmo_cumulative_return_pct")
    qmom_cum = summary.get("qmom_cumulative_return_pct")

    color = COLOR_GREEN if daily_ret >= 0.0 else COLOR_RED
    title = f"AGY Daily Close Summary ({session_date})"
    if spy_daily is not None:
        description = (
            f"Equity: **${equity:,.2f}** | Daily P&L: **{daily_ret:+.2f}%** "
            f"(SPY `{float(spy_daily):+.2f}%`) | Cumulative: **{cum_ret:+.2f}%**"
        )
    else:
        description = (
            f"Equity: **${equity:,.2f}** | Daily P&L: **{daily_ret:+.2f}%** | "
            f"Cumulative: **{cum_ret:+.2f}%**"
        )

    spy_daily_str = f"{float(spy_daily):+.2f}%" if spy_daily is not None else "N/A"
    ew_daily_str = f"{float(ew_daily):+.2f}%" if ew_daily is not None else "N/A"
    spmo_daily_str = f"{float(spmo_daily):+.2f}%" if spmo_daily is not None else "N/A"
    qmom_daily_str = f"{float(qmom_daily):+.2f}%" if qmom_daily is not None else "N/A"
    spy_cum_str = f"{float(spy_cum):+.2f}%" if spy_cum is not None else "N/A"
    ew_cum_str = f"{float(ew_cum):+.2f}%" if ew_cum is not None else "N/A"
    spmo_cum_str = f"{float(spmo_cum):+.2f}%" if spmo_cum is not None else "N/A"
    qmom_cum_str = f"{float(qmom_cum):+.2f}%" if qmom_cum is not None else "N/A"
    pnl_value = (
        f"**Daily:** `{daily_ret:+.2f}%` (`SPY`: `{spy_daily_str}` | `EW-50`: `{ew_daily_str}` | `SPMO`: `{spmo_daily_str}` | `QMOM`: `{qmom_daily_str}`)\n"
        f"**Cumulative:** `{cum_ret:+.2f}%` (`SPY`: `{spy_cum_str}` | `EW-50`: `{ew_cum_str}` | `SPMO`: `{spmo_cum_str}` | `QMOM`: `{qmom_cum_str}` since 1st trade)"
    )

    drawdown_pct = float(summary.get("drawdown_pct", 0.0))
    halt_pct = float(summary.get("max_drawdown_halt_pct", -8.0))
    headroom_pct = float(summary.get("drawdown_headroom_pct", 0.0))
    dd_value = (
        f"Current Drawdown: `{drawdown_pct:.2f}%`\n"
        f"Halt Threshold: `{halt_pct:.2f}%` | Headroom: `{headroom_pct:.2f}%`"
    )

    news = list(summary.get("notable_news", ()))
    sec = list(summary.get("notable_sec_filings", ()))
    fwd = list(summary.get("notable_forward_feeds", ()))
    events_lines = [
        f"Counts (24h): {len(news)} model news, {len(sec)} SEC filings, {len(fwd)} RSS/YF feeds"
    ]
    events_lines.extend(_format_headline_list(news, prefix="News", limit=2))
    events_lines.extend(_format_headline_list(fwd, prefix="RSS", limit=1))
    for ev in sec[:2]:
        items_str = ",".join(str(x) for x in (ev.get("items") or [])) or "-"
        events_lines.append(
            f"- [SEC:{ev.get('symbol', '?')}] Form {ev.get('form_type', '?')} (items={items_str})"
        )

    raw_fields = [
        {
            "name": "Daily & Cumulative P&L vs SPY, EW-50, SPMO & QMOM",
            "value": pnl_value,
            "inline": False,
        },
        {"name": "Drawdown Headroom", "value": dd_value, "inline": False},
        {
            "name": "Notable News / SEC / RSS Events",
            "value": "\n".join(events_lines),
            "inline": False,
        },
    ]
    blackout = summary.get("blackout") or {}
    if blackout.get("active"):
        if blackout.get("halted"):
            bo_val = (
                f"HALTED (fail-closed): {blackout.get('halt_reason') or 'invalid BLACKOUT_PERIOD'}"
            )
        else:
            frozen_str = ", ".join(blackout.get("frozen_symbols") or ()) or "None"
            etf_str = "; ".join(blackout.get("etf_weights_summary") or ()) or "None"
            bo_val = f"Frozen: `{frozen_str}`\nETF Exposure: {etf_str}"
            if blackout.get("warnings"):
                bo_val += f"\nWarnings: {'; '.join(blackout['warnings'])}"
        raw_fields.append(
            {"name": "BLACKOUT_PERIOD Compliance Rail", "value": bo_val, "inline": False}
        )
    norm = normalize_discord_embed(
        title=title,
        description=description,
        color=color,
        fields=raw_fields,
        footer=DEFAULT_DISCORD_FOOTER,
        image_url=image_url,
    )
    out: dict[str, Any] = {
        "title": norm["title"],
        "description": norm["description"],
        "color": norm["color"],
        "fields": norm.get("fields", []),
        "footer": str((norm.get("footer") or {}).get("text", DEFAULT_DISCORD_FOOTER)),
    }
    if "image" in norm:
        out["image"] = norm["image"]
    return out


def build_test_discord_embed(channel_label: str = DEFAULT_DISCORD_CHANNEL_LABEL) -> dict[str, Any]:
    """Build a connectivity verification Discord embed."""
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    raw_fields = [
        {"name": "Target Channel", "value": channel_label, "inline": True},
        {"name": "Status", "value": "Connectivity Verified", "inline": True},
        {"name": "Timestamp", "value": now_str, "inline": False},
    ]
    norm = normalize_discord_embed(
        title="AGY Discord Notification Test",
        description=f"Connectivity check for **{channel_label}**.",
        color=COLOR_BLUE,
        fields=raw_fields,
        footer=DEFAULT_DISCORD_FOOTER,
    )
    return {
        "title": norm["title"],
        "description": norm["description"],
        "color": norm["color"],
        "fields": norm.get("fields", []),
        "footer": str((norm.get("footer") or {}).get("text", DEFAULT_DISCORD_FOOTER)),
    }


def build_blackout_alert_discord_embed(
    blackout: Mapping[str, Any],
    *,
    affected_positions: Sequence[str] = (),
    affected_orders: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a Discord compliance alert embed for BLACKOUT_PERIOD halts/freezes."""
    halted = bool(blackout.get("halted", False))
    frozen_list = [str(s) for s in (blackout.get("frozen_symbols") or ())]
    warnings_list = [str(w) for w in (blackout.get("warnings") or ())]
    etf_summary = [str(x) for x in (blackout.get("etf_weights_summary") or ())]
    pos_list = [str(p) for p in affected_positions]
    ord_list = [str(o) for o in affected_orders]

    if halted:
        color = COLOR_RED
        title = "[COMPLIANCE HALT] BLACKOUT_PERIOD Fail-Closed"
        description = (
            f"All portfolio trading is **HALTED** due to `.env` / `BLACKOUT_PERIOD` failure: "
            f"`{blackout.get('halt_reason') or 'unknown error'}`.\n"
            "Zero orders were placed and existing resting Alpaca orders were left untouched "
            "for manual compliance decision."
        )
    else:
        color = COLOR_AMBER
        title = "[COMPLIANCE] BLACKOUT_PERIOD Active Freeze"
        frozen_label = ", ".join(frozen_list) or "None"
        description = (
            f"Compliance freeze active for **{frozen_label}**.\n"
            "Zero orders were placed and existing resting Alpaca orders were left untouched "
            "for manual compliance decision."
        )

    raw_fields: list[dict[str, Any]] = [
        {
            "name": "Frozen Symbols",
            "value": ", ".join(frozen_list) if frozen_list else ("ALL (HALTED)" if halted else "None"),
            "inline": True,
        },
        {
            "name": "Held Positions Affected",
            "value": ", ".join(pos_list) if pos_list else "None",
            "inline": True,
        },
        {
            "name": "Untouched Resting Orders",
            "value": ", ".join(ord_list[:15]) if ord_list else "None",
            "inline": False,
        },
    ]
    if etf_summary:
        raw_fields.append(
            {
                "name": "ETF Concentration Summary",
                "value": "\n".join(f"- {item}" for item in etf_summary[:10]),
                "inline": False,
            }
        )
    if warnings_list:
        raw_fields.append(
            {
                "name": "Fail-Closed Warnings",
                "value": "\n".join(f"- {w}" for w in warnings_list[:10]),
                "inline": False,
            }
        )

    norm = normalize_discord_embed(
        title=title,
        description=description,
        color=color,
        fields=raw_fields,
        footer=DEFAULT_DISCORD_FOOTER,
    )
    return {
        "title": norm["title"],
        "description": norm["description"],
        "color": norm["color"],
        "fields": norm.get("fields", []),
        "footer": str((norm.get("footer") or {}).get("text", DEFAULT_DISCORD_FOOTER)),
    }
