#!/usr/bin/env bash
# Proxmox LXC bootstrap for BackyardMonitor (uv + systemd)
# Run inside a fresh Debian 12 LXC as root.

set -euo pipefail

APP_DIR="/opt/backyardmonitor"
REPO_URL="https://github.com/posix4e/backyardmonitor.git"
UV_BIN="${HOME}/.local/bin/uv"

echo "[1/7] Install prerequisites (git, curl)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git curl

echo "[2/7] Install uv (Python package manager/runtime)"
if [ ! -x "${UV_BIN}" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if ! command -v uv >/dev/null 2>&1; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "[3/7] Clone or update repo at ${APP_DIR}"
if [ -d "${APP_DIR}/.git" ]; then
  git -C "${APP_DIR}" fetch --all
  git -C "${APP_DIR}" reset --hard origin/main
else
  mkdir -p "${APP_DIR}"
  git clone "${REPO_URL}" "${APP_DIR}"
fi

echo "[4/7] Ensure data directory"
mkdir -p "${APP_DIR}/data"

echo "[5/7] Check for ${APP_DIR}/.env (required)"
if [ ! -f "${APP_DIR}/.env" ]; then
  cat >&2 <<'EOF'
ERROR: .env not found.

Copy your configured .env into the LXC:

  scp .env root@<LXC-IP>:/opt/backyardmonitor/.env

Then rerun:

  bash /opt/backyardmonitor/scripts/proxmox_lxc_uv_bootstrap.sh
EOF
  exit 1
fi

echo "[6/7] Write systemd unit"
UNIT_FILE="/etc/systemd/system/backyardmonitor.service"
cat > "${UNIT_FILE}" <<'UNIT'
[Unit]
Description=Backyard Monitor (uv, git checkout)
After=network.target

[Service]
WorkingDirectory=/opt/backyardmonitor
EnvironmentFile=/opt/backyardmonitor/.env
ExecStart=%h/.local/bin/uv run backyardmonitor --host 0.0.0.0 --port 8080 --env-file /opt/backyardmonitor/.env
Restart=always
RestartSec=2
User=root

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now backyardmonitor

echo "[7/7] Sync environment (first run)"
"${UV_BIN}" sync || true

echo
echo "BackyardMonitor deployed. Visit: http://<LXC-IP>:8080/"
echo "Update later: cd ${APP_DIR} && git fetch && git reset --hard origin/main && ${UV_BIN} sync && systemctl restart backyardmonitor"

