from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    rtsp_url: str | None
    data_dir: Path
    auto_start: bool
    analysis_enabled: bool
    analysis_interval_sec: int
    # Perception thresholds
    phash_min_bits: int = 14
    phash_stable_ms: int = 1200
    # Performance
    capture_max_fps: float = 5.0
    frame_jpeg_fps: float = 2.0
    # Auto-resync behavior for RTSP/capture
    capture_idle_resync_ms: int = 2500
    capture_fail_resync_count: int = 10
    capture_reopen_delay_ms: int = 300
    # Storage/retention
    store_full_frames: bool = False
    store_crops: bool = False
    store_thumbs: bool = True
    jpeg_quality: int = 80
    retain_days: int = 3
    max_events: int = 5000
    max_storage_gb: int = 10

    @classmethod
    def from_env(cls) -> "Settings":
        rtsp_url = os.getenv("RTSP_URL")
        data_dir = Path(os.getenv("DATA_DIR", "data")).resolve()
        auto_start = os.getenv("AUTO_START", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        analysis_enabled = os.getenv("ANALYSIS_ENABLED", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            analysis_interval_sec = int(os.getenv("ANALYSIS_INTERVAL_SEC", "15"))
        except Exception:
            analysis_interval_sec = 15
        # Perception
        try:
            phash_min_bits = max(0, int(os.getenv("PHASH_MIN_BITS", "14")))
        except Exception:
            phash_min_bits = 14
        try:
            phash_stable_ms = max(0, int(os.getenv("PHASH_STABLE_MS", "1200")))
        except Exception:
            phash_stable_ms = 1200
        # Retention/storage knobs
        try:
            jpeg_quality = max(40, min(95, int(os.getenv("JPEG_QUALITY", "80"))))
        except Exception:
            jpeg_quality = 80
        try:
            retain_days = max(0, int(os.getenv("RETAIN_DAYS", "3")))
        except Exception:
            retain_days = 3
        try:
            max_events = max(100, int(os.getenv("MAX_EVENTS", "5000")))
        except Exception:
            max_events = 5000
        try:
            max_storage_gb = max(1, int(os.getenv("MAX_STORAGE_GB", "10")))
        except Exception:
            max_storage_gb = 10
        store_full_frames = os.getenv("STORE_FULL_FRAMES", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        store_crops = os.getenv("STORE_CROPS", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        store_thumbs = os.getenv("STORE_THUMBS", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        # Performance knobs
        try:
            capture_max_fps = float(os.getenv("CAPTURE_MAX_FPS", "5"))
        except Exception:
            capture_max_fps = 5.0
        try:
            frame_jpeg_fps = float(os.getenv("FRAME_JPEG_FPS", "2"))
        except Exception:
            frame_jpeg_fps = 2.0
        # Auto-resync knobs
        try:
            capture_idle_resync_ms = max(0, int(os.getenv("CAPTURE_IDLE_RESYNC_MS", "2500")))
        except Exception:
            capture_idle_resync_ms = 2500
        try:
            capture_fail_resync_count = max(1, int(os.getenv("CAPTURE_FAIL_RESYNC_COUNT", "10")))
        except Exception:
            capture_fail_resync_count = 10
        try:
            capture_reopen_delay_ms = max(0, int(os.getenv("CAPTURE_REOPEN_DELAY_MS", "300")))
        except Exception:
            capture_reopen_delay_ms = 300

        return cls(
            rtsp_url=rtsp_url,
            data_dir=data_dir,
            auto_start=auto_start,
            analysis_enabled=analysis_enabled,
            analysis_interval_sec=analysis_interval_sec,
            phash_min_bits=phash_min_bits,
            phash_stable_ms=phash_stable_ms,
            store_full_frames=store_full_frames,
            store_crops=store_crops,
            store_thumbs=store_thumbs,
            jpeg_quality=jpeg_quality,
            retain_days=retain_days,
            max_events=max_events,
            max_storage_gb=max_storage_gb,
            capture_max_fps=capture_max_fps,
            frame_jpeg_fps=frame_jpeg_fps,
            capture_idle_resync_ms=capture_idle_resync_ms,
            capture_fail_resync_count=capture_fail_resync_count,
            capture_reopen_delay_ms=capture_reopen_delay_ms,
        )
