#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Error: install_launchdaemons.sh must be run as root (use: sudo $0)" >&2
  exit 1
fi

verify_root_dir() {
  local dir="$1"
  if [[ -L "${dir}" || ! -d "${dir}" ]]; then
    echo "Error: ${dir} must be a real directory (not a symlink)" >&2
    exit 1
  fi
  local uid_mode uid mode
  uid_mode="$(stat -f '%u %Lp' "${dir}")"
  uid="${uid_mode%% *}"
  mode="${uid_mode##* }"
  if [[ "${uid}" != "0" ]]; then
    echo "Error: ${dir} must be owned by uid 0 (got uid=${uid})" >&2
    exit 1
  fi
  if (( (8#${mode} & 8#022) != 0 )); then
    echo "Error: ${dir} must not be group/world-writable (got mode=${mode})" >&2
    exit 1
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
APP_LOG_DIR="${DATA_DIR}/logs"
LAUNCHD_LOG_DIR="/private/var/log/agy-trader"
VENV_BIN="${REPO_ROOT}/.venv/bin/agy-ops"
DAEMONCTL_SRC="${SCRIPT_DIR}/agy-daemonctl.py"
DAEMONCTL_DIR="/usr/local/sbin"
DAEMONCTL_DST="${DAEMONCTL_DIR}/agy-daemonctl"
SUDOERS_DIR="/etc/sudoers.d"
SUDOERS_DST="${SUDOERS_DIR}/agytrader"
NEWSYSLOG_SRC="${SCRIPT_DIR}/com.agytrader.newsyslog.conf"
NEWSYSLOG_DST="/etc/newsyslog.d/com.agytrader.conf"
RUN_USER="trader"

if [[ ! -x "${VENV_BIN}" ]]; then
  echo "Error: missing executable ${VENV_BIN}. Run 'uv sync --offline' as ${RUN_USER} first." >&2
  exit 1
fi

if [[ ! -f "${DAEMONCTL_SRC}" ]]; then
  echo "Error: missing helper script ${DAEMONCTL_SRC}" >&2
  exit 1
fi

# Create unprivileged data/app-log directories strictly as RUN_USER (never chown as root in ~<user>)
sudo -u "${RUN_USER}" mkdir -p "${DATA_DIR}" "${APP_LOG_DIR}"

# Verify root-owned system prefixes before installing privileged helper
verify_root_dir "/usr"
verify_root_dir "/usr/local"
verify_root_dir "/private"
verify_root_dir "/private/var"
verify_root_dir "/private/var/log"

if [[ -L "${DAEMONCTL_DIR}" ]]; then
  echo "Error: ${DAEMONCTL_DIR} must not be a symlink" >&2
  exit 1
fi
mkdir -p "${DAEMONCTL_DIR}"
chown root:wheel "${DAEMONCTL_DIR}"
chmod 0755 "${DAEMONCTL_DIR}"

if [[ -L "${LAUNCHD_LOG_DIR}" ]]; then
  echo "Error: ${LAUNCHD_LOG_DIR} must not be a symlink" >&2
  exit 1
fi
mkdir -p "${LAUNCHD_LOG_DIR}"
chown root:wheel "${LAUNCHD_LOG_DIR}"
chmod 0755 "${LAUNCHD_LOG_DIR}"

tmp_ctl=""
tmp_sudoers=""
cleanup_tmp_files() {
  if [[ -n "${tmp_ctl:-}" && -f "${tmp_ctl}" ]]; then
    rm -f "${tmp_ctl}"
  fi
  if [[ -n "${tmp_sudoers:-}" && -f "${tmp_sudoers}" ]]; then
    rm -f "${tmp_sudoers}"
  fi
}
trap cleanup_tmp_files EXIT

# Install privilege-separated daemonctl helper atomically
tmp_ctl="$(mktemp "${DAEMONCTL_DIR}/.agy-daemonctl.tmp.XXXXXX")"
cp "${DAEMONCTL_SRC}" "${tmp_ctl}"
chown root:wheel "${tmp_ctl}"
chmod 0755 "${tmp_ctl}"
mv -f "${tmp_ctl}" "${DAEMONCTL_DST}"
tmp_ctl=""
echo "Installed daemonctl helper at ${DAEMONCTL_DST}"

# Install validated passwordless sudoers drop-in scoped strictly to agy-daemonctl
install -d -o root -g wheel -m 0750 "${SUDOERS_DIR}"
tmp_sudoers="$(mktemp "${SUDOERS_DIR}/.agytrader.tmp.XXXXXX")"

cat > "${tmp_sudoers}" <<'EOF'
Defaults!/usr/local/sbin/agy-daemonctl !requiretty, env_reset, always_set_home, secure_path="/usr/bin:/bin:/usr/sbin:/sbin"
trader ALL=(root) NOPASSWD: /usr/local/sbin/agy-daemonctl
trader ALL=(root) NOPASSWD: /usr/local/sbin/agy-daemonctl *
EOF

if ! /usr/sbin/visudo -cf "${tmp_sudoers}"; then
  echo "Error: visudo validation failed for ${tmp_sudoers}; aborting without modifying ${SUDOERS_DST}" >&2
  exit 1
fi

chown root:wheel "${tmp_sudoers}"
chmod 0440 "${tmp_sudoers}"
mv -f "${tmp_sudoers}" "${SUDOERS_DST}"
tmp_sudoers=""
echo "Installed validated sudoers policy at ${SUDOERS_DST}"

# Configure macOS always-on power resilience (safe best-effort on VMs/laptops)
echo "Configuring macOS power resilience (sleep=0, displaysleep=15, autorestart=1)..."
pmset -a sleep 0 displaysleep 15 autorestart 1 || true
pmset -g || true

# Install log rotation config
if [[ -d "/etc/newsyslog.d" && -f "${NEWSYSLOG_SRC}" ]]; then
  cp "${NEWSYSLOG_SRC}" "${NEWSYSLOG_DST}"
  chown root:wheel "${NEWSYSLOG_DST}"
  chmod 0644 "${NEWSYSLOG_DST}"
  echo "Installed log rotation config at ${NEWSYSLOG_DST}"
fi

# Validate, install, and bootstrap all 4 LaunchDaemons via agy-daemonctl
echo "Validating and syncing all LaunchDaemons..."
"${DAEMONCTL_DST}" sync-plists all

echo "All agy-trader LaunchDaemons installed and bootstrapped."
echo "Unprivileged management ready: uv run --offline agy-ops service status"
echo "Verify readiness with: uv run --offline agy-ops readiness"
