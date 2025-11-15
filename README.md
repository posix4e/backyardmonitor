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
- RTSP is opened via FFmpeg with TCP by default for reliability. You can override with `RTSP_TRANSPORT=udp` if your camera requires UDP.

2) Install deps (using uv or pip)

- uv: `uv sync`
- pip: `python -m venv .venv && . .venv/bin/activate && pip install -e .`

3) Run the server

- Shortcut: `uv run backyardmonitor --reload --port 8080 --env-file .env`
  - Or: `uvicorn bm.app:app --reload --port 8080 --env-file .env`
 - Open http://localhost:8080 — one page for live view, spot calibration, and a generic JSON editor for zones, events, and image management.

## Deployment

Recommended: uv + systemd (no Docker). See `DEPLOY.md` for the unit file and a one‑command `deploy.sh`.

Quick run locally:
- `uv sync`
- `uv run backyardmonitor --reload --port 8080 --env-file .env`

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

## Capture reliability

- RTSP is opened with the FFmpeg backend and `rtsp_transport=tcp` (plus a 5s socket timeout) by default to reduce packet loss and stream breakage.
- The capture loop drops likely garbage frames (all-black/white or extremely low-variance grey) and resyncs after consecutive bad reads or idle periods. This prevents corrupt frames from propagating into event detection.
