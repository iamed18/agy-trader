#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Error: install_launchdaemons.sh must be run as root (use: sudo $0)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET_DIR="/Library/LaunchDaemons"
DATA_DIR="${REPO_ROOT}/data"
LOG_DIR="${DATA_DIR}/logs"
VENV_BIN="${REPO_ROOT}/.venv/bin/agy-ops"
NEWSYSLOG_SRC="${SCRIPT_DIR}/com.agytrader.newsyslog.conf"
NEWSYSLOG_DST="/etc/newsyslog.d/com.agytrader.conf"
RUN_USER="trader"

PLISTS=(
  "com.agytrader.market-data.plist"
  "com.agytrader.news-nlp.plist"
  "com.agytrader.trader.plist"
  "com.agytrader.dashboard.plist"
)

if [[ ! -x "${VENV_BIN}" ]]; then
  echo "Error: missing executable ${VENV_BIN}. Run 'uv sync --offline' as ${RUN_USER} first." >&2
  exit 1
fi

mkdir -p "${DATA_DIR}" "${LOG_DIR}"
chown "${RUN_USER}:staff" "${DATA_DIR}"
chown -R "${RUN_USER}:staff" "${LOG_DIR}"

for plist in "${PLISTS[@]}"; do
  src="${SCRIPT_DIR}/${plist}"
  if [[ ! -f "${src}" ]]; then
    echo "Error: missing plist ${src}" >&2
    exit 1
  fi
  echo "Validating ${plist}..."
  plutil -lint "${src}"
done

if [[ -d "/etc/newsyslog.d" && -f "${NEWSYSLOG_SRC}" ]]; then
  cp "${NEWSYSLOG_SRC}" "${NEWSYSLOG_DST}"
  chown root:wheel "${NEWSYSLOG_DST}"
  chmod 644 "${NEWSYSLOG_DST}"
  echo "Installed log rotation config at ${NEWSYSLOG_DST}"
fi

for plist in "${PLISTS[@]}"; do
  label="${plist%.plist}"
  src="${SCRIPT_DIR}/${plist}"
  dst="${TARGET_DIR}/${plist}"

  # Unload existing service and wait briefly for launchd teardown before bootstrapping
  if launchctl print "system/${label}" >/dev/null 2>&1; then
    launchctl bootout "system/${label}" 2>/dev/null || true
    for _ in $(seq 1 10); do
      if ! launchctl print "system/${label}" >/dev/null 2>&1; then
        break
      fi
      sleep 0.5
    done
  fi

  cp "${src}" "${dst}"
  chown root:wheel "${dst}"
  chmod 644 "${dst}"

  echo "Bootstrapping system/${label}..."
  launchctl bootstrap system "${dst}"
done

echo "All agy-trader LaunchDaemons installed and bootstrapped."
echo "Verify with: uv run --offline agy-ops readiness"
