# macOS Always-On `LaunchDaemon` & Reboot Recovery Drill (`REBOOT_DRILL.md`)

This checklist verifies that all four `agy-trader` services (`market-data`, `news-nlp`, `trader`, `dashboard`) start **automatically before user login**, survive process crashes and temporary network outages, preserve SQLite WAL integrity, and protect open positions with resting GTC stop-losses.

---

## 1. Prerequisites & FileVault Power-Outage Behavior

`LaunchDaemon`s in `/Library/LaunchDaemons/` launch at boot **before** user login once macOS reaches the login window.

- **Always-on power & auto-restart after power loss (AC power):**
  `install_launchdaemons.sh` automatically applies:
  ```bash
  sudo pmset -a sleep 0 displaysleep 15 autorestart 1
  pmset -g | grep -E "sleep|displaysleep|autorestart"
  ```
  This keeps the Mac awake for pre-market, intraday, and post-close daemon cycles while still allowing the display to sleep after 15 minutes and restarting automatically after AC power loss.
- **FileVault Note:**
  - Check status: `fdesetup status`
  - If **FileVault is On**:
    - An unexpected power outage pauses at the pre-boot FileVault unlock screen until you enter your password (meanwhile, resting broker-side GTC stop-loss orders on Alpaca continue protecting open positions).
    - For **planned maintenance reboots**, bypass the FileVault pre-boot screen once with:
      ```bash
      sudo fdesetup authrestart
      ```
  - If **FileVault is Off**: macOS boots straight to the login window and starts all `LaunchDaemon`s unattended.

---

## 2. One-Time Install & Zero-Prompt `agy-ops service` Management

From the repository root (`<repo-root>`), run the installer **once** with `sudo` (it automatically detects your non-root `$SUDO_USER` and `$REPO_ROOT`, or you can pass `AGY_RUN_USER=<user>`):

```bash
sudo ./deploy/macos/install_launchdaemons.sh
```

This one-time setup:
- Renders host-local (gitignored) `deploy/macos/com.agytrader.*.plist` files as `<user>` via `agy-ops render-manifests`
- Stamps and installs the privilege-separated helper `/usr/local/sbin/agy-daemonctl` (`root:wheel`, `0755`, running under `/usr/bin/python3 -I`)
- Installs a `visudo`-validated `/etc/sudoers.d/agytrader` policy granting `<user>` passwordless `NOPASSWD` access strictly to `/usr/local/sbin/agy-daemonctl`
- Configures `pmset` (`sleep 0`, `displaysleep 15`, `autorestart 1`) and renders `/etc/newsyslog.d/com.agytrader.conf` from `deploy/macos/com.agytrader.newsyslog.conf.tmpl`
- Validates, installs, and bootstraps all 4 `LaunchDaemon` plists in `/Library/LaunchDaemons/`

After this one-time install, unprivileged shells and AGY agent sessions (as `<user>`, with **no interactive sudo password prompt**) can manage all 4 daemons via `agy-ops service`:

```bash
uv run --offline agy-ops service status
uv run --offline agy-ops service status --json
uv run --offline agy-ops service start all
uv run --offline agy-ops service stop dashboard
uv run --offline agy-ops service restart trader --actor operator --reason "manual check"
uv run --offline agy-ops service restart trader --actor ops --reason "stale feed recovery"
uv run --offline agy-ops service sync-plists all
```

> **Note on agent rate limits:** `agy-ops service restart` enforces `ControlPlane.authorize_service_restart(...)` (max 2 restarts/hour per service for `market-data`, `news-nlp`, `trader`, requiring `--reason`) whenever `--actor` is not `operator`.

Verify logs and initial readiness:

```bash
tail -n 20 <repo-root>/data/logs/*.log
tail -n 20 /private/var/log/agy-trader/*.log
uv run --offline agy-ops service status
uv run --offline agy-ops readiness
```

- `<repo-root>/data/logs/*.log`: structured Python application logs (`RotatingFileHandler`, owned by `<user>:<group>`).
- `/private/var/log/agy-trader/*.log`: `launchd` stdout/stderr logs (owned by `<user>:<group>` `0644` inside `root:wheel` `0755` `/private/var/log/agy-trader/`, rotated by `newsyslog`).

Expected: `status: READY` (or `idle_market_closed` heartbeats outside `09:30–16:00 ET`), WAL integrity `ok`, and dashboard reachable at `http://127.0.0.1:8080`.

---

## 3. Chaos Drill 1: `kill -9` Auto-Restart Verification

Confirm `launchd` `KeepAlive=true` + `ThrottleInterval=15` automatically restarts a killed daemon (running as `<user>`, so no `sudo` is required) and reconciles clean state:

```bash
pgrep -fl "agy-ops daemon trader"
kill -9 "$(pgrep -f 'agy-ops daemon trader' | head -n 1)"
sleep 20
pgrep -fl "agy-ops daemon trader"
uv run --offline agy-ops service status trader
uv run --offline agy-ops readiness --json
```

Verify:
1. A new PID is running for `agy-ops daemon trader`.
2. `state.db` `daemon_events` records a new `boot` and `reconcile` event.
3. Zero duplicate orders were submitted (`live_orders` count unchanged).

---

## 4. Chaos Drill 2: Cold-Boot Network Outage Backoff

1. Temporarily disconnect Wi-Fi / Ethernet (or disable interface) and restart a daemon with no interactive `sudo`:
   ```bash
   uv run --offline agy-ops service restart trader --actor operator
   ```
2. Observe `<repo-root>/data/logs/*.log` and `/private/var/log/agy-trader/trader.stderr.log` / `trader.stdout.log`: the daemon waits with exponential backoff (`wait_for_network`, up to 120s) instead of crash-looping.
3. Reconnect network and confirm the daemon completes startup, reconciles with Alpaca, and emits a `"healthy"` or `"idle_market_closed"` heartbeat.

---

## 5. Full Mac Reboot Verification

1. Trigger a reboot (use `sudo fdesetup authrestart` if FileVault is enabled, or `sudo reboot`):
   ```bash
   sudo fdesetup authrestart   # or: sudo reboot
   ```
2. **Wait 60 seconds after boot** (even before logging into the GUI if FileVault is off, or after unlocking if FileVault is on).
3. Run post-reboot checks:
   ```bash
   cd <repo-root>
   uv run --offline agy-ops service status
   uv run --offline agy-ops readiness --json
   curl -s http://127.0.0.1:8080/api/health | python3 -m json.tool
   ```
4. Confirm:
   - All 4 `com.agytrader.*` daemons are running with non-zero PIDs.
   - `wal_ok == true`.
   - `agy-ops readiness` reports `READY` (zero missing GTC stops on open positions).
   - `http://127.0.0.1:8080` serves the live read-only portfolio dashboard.
