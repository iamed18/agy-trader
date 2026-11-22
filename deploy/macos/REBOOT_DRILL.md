# macOS Always-On `LaunchDaemon` & Reboot Recovery Drill (`REBOOT_DRILL.md`)

This checklist verifies that all four `agy-trader` services (`market-data`, `news-nlp`, `trader`, `dashboard`) start **automatically before user login**, survive process crashes and temporary network outages, preserve SQLite WAL integrity, and protect open positions with resting GTC stop-losses.

---

## 1. Prerequisites & FileVault Power-Outage Behavior

`LaunchDaemon`s in `/Library/LaunchDaemons/` launch at boot **before** user login once macOS reaches the login window.

- **Auto-restart after power loss (AC power):**
  ```bash
  sudo pmset -a autorestart 1
  pmset -g | grep autorestart
  ```
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

## 2. Install `LaunchDaemon`s

From the repository root (`/path/to/agy-trader`):

```bash
sudo ./deploy/macos/install_launchdaemons.sh
```

Verify all 4 services are loaded:

```bash
sudo launchctl list | grep com.agytrader
```

Verify logs and initial readiness:

```bash
tail -n 20 data/logs/*.log
uv run --offline agy-ops readiness
```

Expected: `status: READY` (or `idle_market_closed` heartbeats outside `09:30–16:00 ET`), WAL integrity `ok`, and dashboard reachable at `http://127.0.0.1:8080`.

---

## 3. Chaos Drill 1: `kill -9` Auto-Restart Verification

Confirm `launchd` `KeepAlive=true` + `ThrottleInterval=15` automatically restarts a killed daemon and reconciles clean state:

```bash
pgrep -fl "agy-ops daemon trader"
sudo kill -9 "$(pgrep -f 'agy-ops daemon trader' | head -n 1)"
sleep 20
pgrep -fl "agy-ops daemon trader"
uv run --offline agy-ops readiness --json
```

Verify:
1. A new PID is running for `agy-ops daemon trader`.
2. `state.db` `daemon_events` records a new `boot` and `reconcile` event.
3. Zero duplicate orders were submitted (`live_orders` count unchanged).

---

## 4. Chaos Drill 2: Cold-Boot Network Outage Backoff

1. Temporarily disconnect Wi-Fi / Ethernet (or disable interface) and kickstart a daemon:
   ```bash
   sudo launchctl kickstart -k system/com.agytrader.trader
   ```
2. Observe `data/logs/trader.stderr.log` / `trader.stdout.log`: the daemon waits with exponential backoff (`wait_for_network`, up to 120s) instead of crash-looping.
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
   cd /path/to/agy-trader
   sudo launchctl list | grep com.agytrader
   uv run --offline agy-ops readiness --json
   curl -s http://127.0.0.1:8080/api/health | python3 -m json.tool
   ```
4. Confirm:
   - All 4 `com.agytrader.*` daemons are running with non-zero PIDs.
   - `wal_ok == true`.
   - `agy-ops readiness` reports `READY` (zero missing GTC stops on open positions).
   - `http://127.0.0.1:8080` serves the live read-only portfolio dashboard.
