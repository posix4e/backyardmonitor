Backyard Monitor — Deploy (uv + systemd)

This app now runs directly from a Git checkout using uv, without Docker or Watchtower.

Prerequisites
- Linux host with systemd
- Git available
- uv installed for the runtime user (e.g., root):
  - curl -LsSf https://astral.sh/uv/install.sh | sh
  - Ensure ~/.local/bin is on PATH

Layout
- Repo path: /opt/backyardmonitor
- Env file: /opt/backyardmonitor/.env
- Service: backyardmonitor (systemd)

Systemd unit
Create /etc/systemd/system/backyardmonitor.service with:

[Unit]
Description=Backyard Monitor (uv, git checkout)
After=network.target

[Service]
WorkingDirectory=/opt/backyardmonitor
EnvironmentFile=/opt/backyardmonitor/.env
ExecStart=/root/.local/bin/uv run backyardmonitor --host 0.0.0.0 --port 8080 --env-file /opt/backyardmonitor/.env
Restart=always
RestartSec=2
User=root

[Install]
WantedBy=multi-user.target

Then run:
- systemctl daemon-reload
- systemctl enable --now backyardmonitor

Updating (deploy)
- cd /opt/backyardmonitor
- git fetch && git reset --hard origin/main
- /root/.local/bin/uv sync
- systemctl restart backyardmonitor

Optional helper script (deploy.sh)
Place in /opt/backyardmonitor/deploy.sh and make it executable:

#!/usr/bin/env bash
set -euo pipefail
cd /opt/backyardmonitor
git fetch origin
git reset --hard origin/main
/root/.local/bin/uv sync
systemctl restart backyardmonitor
echo "Deploy complete: $(date)"

Notes
- .env holds RTSP_URL and other knobs; see .env.example for available variables.
- Docker and Watchtower are no longer used. Any compose files are obsolete.
