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

GPU Acceleration (optional)

You can offload HEVC/H.264 decode to the GPU. Two common setups are Intel iGPU (VAAPI) and NVIDIA (NVDEC/CUDA).

- Proxmox LXC passthrough
  - Intel/AMD (VAAPI): edit /etc/pve/lxc/101.conf and add:
    - lxc.cgroup2.devices.allow: c 226:* rwm
    - lxc.mount.entry: /dev/dri dev/dri none bind,optional,create=dir
  - NVIDIA (NVDEC): install NVIDIA drivers on the host, then add to 101.conf:
    - lxc.cgroup2.devices.allow: c 195:* rwm
    - lxc.cgroup2.devices.allow: c 507:* rwm  # nvidia-uvm (major varies by distro)
    - lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,create=file
    - lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,create=file
    - lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,create=file,optional
    - lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,create=file,optional
  - Restart the container after editing the config: pct stop 101 && pct start 101

- Inside the container (LXC 101)
  - Intel/AMD: apt install -y ffmpeg vainfo mesa-va-drivers i965-va-driver-shaders intel-media-va-driver-non-free (as applicable)
    - Verify: vainfo
  - NVIDIA: install matching NVIDIA driver userspace + FFmpeg built with NVDEC support (e.g., ffmpeg from distro with --enable-nvdec)
    - Verify: nvidia-smi

- App configuration (.env)
  - For NVIDIA NVDEC:
    - HWACCEL=cuda
    - HWACCEL_DEVICE=0        # GPU index (or leave empty)
  - For Intel/AMD VAAPI:
    - HWACCEL=vaapi
    - HWACCEL_DEVICE=/dev/dri/renderD128
  - These flags are passed through to FFmpeg via OpenCV and will be used if supported. The app still falls back to CPU safely.

- Alternative (GStreamer pipeline)
  - For more reliable HW decode, you can run capture via GStreamer and appsink (requires OpenCV built with GStreamer). Example pipelines:
    - VAAPI HEVC: rtspsrc location=rtsp://... protocols=tcp latency=0 ! rtph265depay ! h265parse ! vaapih265dec ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=2
    - NVIDIA HEVC: rtspsrc location=rtsp://... protocols=tcp latency=0 ! rtph265depay ! h265parse ! nvh265dec ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=2
  - If you go this route, set the service ExecStart to use an env var (e.g., CAPTURE_URI) and modify the app to open with CAP_GSTREAMER using that pipeline.

Notes on stability
- We already set err_detect=explode and discardcorrupt to prefer failing/resyncing over returning low-detail frames.
- HW acceleration depends on distro FFmpeg/OpenCV build. If unsupported, options are ignored and CPU decode is used.
