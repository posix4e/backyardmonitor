# Backyard Monitor (Rewrite)

A minimal single-process web server to:
- Draw parking spot polygons on a live frame
- Start/stop a background capture loop
- Store events in a lightweight SQLite DB

No separate CLI is needed; everything runs behind the webserver.

## Quick Start

1) Configure environment

- Copy `.env.example` to `.env` and set `RTSP_URL`.
- Optionally set `DATA_DIR` (default `./data`) and `AUTO_START=true|false`.

2) Install deps (using uv or pip)

- uv: `uv sync`
- pip: `python -m venv .venv && . .venv/bin/activate && pip install -e .`

3) Run the server

- Shortcut: `uv run backyardmonitor --reload --port 8080 --env-file .env`
  - Or: `uvicorn bm.app:app --reload --port 8080 --env-file .env`
 - Open http://localhost:8080 — one page for live view, spot calibration, and a generic JSON editor for zones, events, and image management.

## Proxmox LXC (Docker) Deploy

Use this when running on a Proxmox host (e.g., root@192.168.2.4) in a Debian 12 LXC with nesting.

- Create the LXC on the Proxmox host (example VMID 101):
  - `pveam update && pveam download local debian-12-standard_12*amd64.tar.zst`
  - `pct create 101 local:vztmpl/debian-12-standard_12*amd64.tar.zst -hostname backyardmonitor -net0 name=eth0,bridge=vmbr0,ip=dhcp -cores 2 -memory 2048 -rootfs local-lvm:8 -features nesting=1,keyctl=1 -unprivileged 0`
  - `pct start 101 && pct enter 101`

- Inside the LXC, run the bootstrap script:
  - `bash -c "curl -fsSL https://raw.githubusercontent.com/posix4e/backyardmonitor/main/scripts/proxmox_lxc_bootstrap.sh -o /root/proxmox_lxc_bootstrap.sh && bash /root/proxmox_lxc_bootstrap.sh"`
  - If `curl` access to GitHub is restricted, you can copy this repo into the LXC manually and run `bash /opt/backyardmonitor/scripts/proxmox_lxc_bootstrap.sh`.

- Provide your existing `.env` (from this repo on your workstation):
  - `scp .env root@<LXC-IP>:/opt/backyardmonitor/.env`
  - Then re-run: `bash /opt/backyardmonitor/scripts/proxmox_lxc_bootstrap.sh`

- Access the UI at `http://<LXC-IP>:8080/`.

Notes:
- The compose file mounts `./data` to `/data` in the container and `./.env` to `/app/.env:ro`.
- Ensure the LXC network can reach your camera in `.env` (`RTSP_URL`).

## Calibrate

- Spots
  - Click to place parking spot rectangles.
  - Click a spot to select; use arrow keys to nudge (Shift for faster); Delete to remove.
  - Rotate selected with the Rotate buttons; rename via the Name field.
  - The app tracks visual changes per spot and logs spot_change events with durations.
  - Zones persist to `DATA_DIR/zones.json`. You can also edit zones JSON directly in the right-side Data panel.

- Events
  - Use the Data panel to list recent events, edit event kind/meta JSON, or delete events.

## Notes

- This is the clean rewrite with minimal pieces. Detection is intentionally simple/placeholder initially; the focus is easy setup and manual calibration.
- Old code is preserved under `reference_main/` on this branch for reference only.

## Retention and storage

Environment knobs (with sane defaults):
- `STORE_FULL_FRAMES` (default: false)
- `STORE_CROPS` (default: false)
- `STORE_THUMBS` (default: true)
- `JPEG_QUALITY` (default: 80)
- `RETAIN_DAYS` (default: 3)
- `MAX_EVENTS` (default: 5000)
- `MAX_STORAGE_GB` (default: 10)

Use the Data panel → Images to view summary and cleanup orphans. You can also trigger retention manually via `POST /api/retention/apply`.
