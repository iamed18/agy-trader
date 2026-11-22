#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Error: uninstall_launchdaemons.sh must be run as root (use: sudo $0)" >&2
  exit 1
fi

TARGET_DIR="/Library/LaunchDaemons"
PLISTS=(
  "com.agytrader.market-data.plist"
  "com.agytrader.news-nlp.plist"
  "com.agytrader.trader.plist"
  "com.agytrader.dashboard.plist"
)

for plist in "${PLISTS[@]}"; do
  label="${plist%.plist}"
  dst="${TARGET_DIR}/${plist}"
  echo "Unloading system/${label}..."
  launchctl bootout "system/${label}" 2>/dev/null || true
  if [[ -f "${dst}" ]]; then
    rm -f "${dst}"
    echo "Removed ${dst}"
  fi
done

if [[ -f "/etc/newsyslog.d/com.agytrader.conf" ]]; then
  rm -f "/etc/newsyslog.d/com.agytrader.conf"
  echo "Removed /etc/newsyslog.d/com.agytrader.conf"
fi

echo "All agy-trader LaunchDaemons unloaded and removed."
