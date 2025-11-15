#!/usr/bin/env bash
# Proxmox LXC bootstrap for BackyardMonitor
# Run this inside a fresh Debian 12 LXC as root.
# It installs Docker + Compose, clones the repo, and brings it up
# using the .env you provide (copy your existing .env to /opt/backyardmonitor/.env).

set -euo pipefail

REPO_URL="https://github.com/posix4e/backyardmonitor.git"
APP_DIR="/opt/backyardmonitor"

echo "[1/6] Installing prerequisites (apt, ca-certificates, curl, git, gnupg)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ca-certificates curl git gnupg

echo "[2/6] Installing Docker Engine and compose-plugin"
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
CODENAME=$( . /etc/os-release && echo "$VERSION_CODENAME" )
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian ${CODENAME} stable" > /etc/apt/sources.list.d/docker.list
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "[3/6] Cloning repository to ${APP_DIR}"
if [ -d "${APP_DIR}/.git" ]; then
  echo "Repo already present, pulling latest..."
  git -C "${APP_DIR}" pull --ff-only || true
else
  mkdir -p "${APP_DIR}"
  git clone "${REPO_URL}" "${APP_DIR}"
fi

echo "[4/6] Ensuring data directory exists"
mkdir -p "${APP_DIR}/data"

echo "[5/6] Checking for .env at ${APP_DIR}/.env"
if [ ! -f "${APP_DIR}/.env" ]; then
  cat >&2 <<'EOF'
ERROR: .env not found.

Copy your existing .env from your workstation into the LXC:

  scp .env root@<LXC-IP>:/opt/backyardmonitor/.env

Then re-run this script:

  bash /opt/backyardmonitor/scripts/proxmox_lxc_bootstrap.sh

Note: The compose file mounts ./.env into the container as /app/.env:ro
EOF
  exit 1
fi

echo "[6/6] Building and starting with docker compose"
cd "${APP_DIR}"
docker compose up -d --build

echo "\nDone. Access the UI at:  http://<LXC-IP>:8080/"
echo "To update later:        cd ${APP_DIR} && git pull && docker compose up -d --build"
echo "If you need to change RTSP later, update ${APP_DIR}/.env and run: docker compose up -d"

