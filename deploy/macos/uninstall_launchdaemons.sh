#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Error: uninstall_launchdaemons.sh must be run as root (use: sudo $0)" >&2
  exit 1
fi

TARGET_DIR="/Library/LaunchDaemons"
DAEMONCTL_DST="/usr/local/sbin/agy-daemonctl"
SUDOERS_DST="/etc/sudoers.d/agytrader"
NEWSYSLOG_DST="/etc/newsyslog.d/com.agytrader.conf"
LAUNCHD_LOG_DIR="/private/var/log/agy-trader"
PLISTS=(
  "com.agytrader.market-data.plist"
  "com.agytrader.news-nlp.plist"
  "com.agytrader.trader.plist"
  "com.agytrader.dashboard.plist"
)

if [[ -x "${DAEMONCTL_DST}" ]]; then
  echo "Stopping all agy-trader LaunchDaemons via ${DAEMONCTL_DST}..."
  "${DAEMONCTL_DST}" stop all || true
else
  for plist in "${PLISTS[@]}"; do
    label="${plist%.plist}"
    echo "Unloading system/${label}..."
    launchctl bootout "system/${label}" 2>/dev/null || true
  done
fi

for plist in "${PLISTS[@]}"; do
  dst="${TARGET_DIR}/${plist}"
  if [[ -f "${dst}" ]]; then
    rm -f "${dst}"
    echo "Removed ${dst}"
  fi
done

if [[ -f "${DAEMONCTL_DST}" ]]; then
  rm -f "${DAEMONCTL_DST}"
  echo "Removed ${DAEMONCTL_DST}"
fi

if [[ -f "${SUDOERS_DST}" ]]; then
  rm -f "${SUDOERS_DST}"
  echo "Removed ${SUDOERS_DST}"
fi

if [[ -f "${NEWSYSLOG_DST}" ]]; then
  rm -f "${NEWSYSLOG_DST}"
  echo "Removed ${NEWSYSLOG_DST}"
fi

if [[ -d "${LAUNCHD_LOG_DIR}" ]]; then
  rm -f "${LAUNCHD_LOG_DIR}/"*.log* 2>/dev/null || true
  rmdir "${LAUNCHD_LOG_DIR}" 2>/dev/null || true
  echo "Cleaned ${LAUNCHD_LOG_DIR}"
fi

echo "All agy-trader LaunchDaemons, helper, sudoers policy, logs, and newsyslog config removed."
