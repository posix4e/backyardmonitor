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
    # Detection method and ROI-diff parameters
    detector_method: str = "phash"  # one of: phash, roi_diff
    roi_diff_alpha: float = 0.05
    roi_diff_threshold: float = 0.02
    roi_diff_min_pixels: int = 600
    roi_diff_cooldown_ms: int = 0
    # Category defaults (semantic, independent of ROI/gate geometry)
    category_gate_stable_ms: int = 400
    category_gate_phash_min_bits: int = 10
    category_gate_roi_diff_threshold: float = 0.035
    category_gate_min_pixels: int = 800
    category_gate_cooldown_ms: int = 3000
    category_gate_flow_mag_min: float = 0.5
    category_parking_stable_ms: int = 1200
    category_parking_phash_min_bits: int = 14
    category_parking_roi_diff_threshold: float = 0.02
    category_parking_min_pixels: int = 600
    category_parking_cooldown_ms: int = 45000
    # Road/lane category
    category_road_stable_ms: int = 800
    category_road_phash_min_bits: int = 16
    category_road_roi_diff_threshold: float = 0.08
    category_road_min_pixels: int = 1200
    category_road_cooldown_ms: int = 5000
    category_road_flow_mag_min: float = 1.0
    # LLM analysis
    llm_enabled: bool = False
    llm_provider: str = "openrouter"
    llm_model_fast: str = "google/gemini-2.5-flash"
    llm_timeout_sec: int = 20
    # UI tuning
    ui_frame_refresh_ms: int = 10000
    # LLM burst suppression window
    llm_burst_window_ms: int = 1500

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
            capture_idle_resync_ms = max(
                0, int(os.getenv("CAPTURE_IDLE_RESYNC_MS", "2500"))
            )
        except Exception:
            capture_idle_resync_ms = 2500
        try:
            capture_fail_resync_count = max(
                1, int(os.getenv("CAPTURE_FAIL_RESYNC_COUNT", "10"))
            )
        except Exception:
            capture_fail_resync_count = 10
        try:
            capture_reopen_delay_ms = max(
                0, int(os.getenv("CAPTURE_REOPEN_DELAY_MS", "300"))
            )
        except Exception:
            capture_reopen_delay_ms = 300
        # Detection method
        detector_method = os.getenv("DETECTOR_METHOD", "phash").strip().lower()
        try:
            roi_diff_alpha = float(os.getenv("ROI_DIFF_ALPHA", "0.05"))
        except Exception:
            roi_diff_alpha = 0.05
        try:
            roi_diff_threshold = float(os.getenv("ROI_DIFF_THRESHOLD", "0.02"))
        except Exception:
            roi_diff_threshold = 0.02
        try:
            roi_diff_min_pixels = int(os.getenv("ROI_DIFF_MIN_PIXELS", "600"))
        except Exception:
            roi_diff_min_pixels = 600
        try:
            roi_diff_cooldown_ms = int(os.getenv("ROI_DIFF_COOLDOWN_MS", "0"))
        except Exception:
            roi_diff_cooldown_ms = 0

        # Category defaults (optional)
        def _int(env, default):
            try:
                return int(os.getenv(env, str(default)))
            except Exception:
                return default

        def _float(env, default):
            try:
                return float(os.getenv(env, str(default)))
            except Exception:
                return default

        category_gate_stable_ms = _int("CATEGORY_GATE_STABLE_MS", 400)
        category_gate_phash_min_bits = _int("CATEGORY_GATE_PHASH_MIN_BITS", 10)
        category_gate_roi_diff_threshold = _float(
            "CATEGORY_GATE_ROI_DIFF_THRESHOLD", 0.035
        )
        category_gate_min_pixels = _int("CATEGORY_GATE_MIN_PIXELS", 800)
        category_gate_cooldown_ms = _int("CATEGORY_GATE_COOLDOWN_MS", 3000)
        category_gate_flow_mag_min = _float("CATEGORY_GATE_FLOW_MAG_MIN", 0.5)
        category_parking_stable_ms = _int("CATEGORY_PARKING_STABLE_MS", 1200)
        category_parking_phash_min_bits = _int("CATEGORY_PARKING_PHASH_MIN_BITS", 14)
        category_parking_roi_diff_threshold = _float(
            "CATEGORY_PARKING_ROI_DIFF_THRESHOLD", 0.02
        )
        category_parking_min_pixels = _int("CATEGORY_PARKING_MIN_PIXELS", 600)
        category_parking_cooldown_ms = _int("CATEGORY_PARKING_COOLDOWN_MS", 45000)
        # Road
        category_road_stable_ms = _int("CATEGORY_ROAD_STABLE_MS", 800)
        category_road_phash_min_bits = _int("CATEGORY_ROAD_PHASH_MIN_BITS", 16)
        category_road_roi_diff_threshold = _float(
            "CATEGORY_ROAD_ROI_DIFF_THRESHOLD", 0.08
        )
        category_road_min_pixels = _int("CATEGORY_ROAD_MIN_PIXELS", 1200)
        category_road_cooldown_ms = _int("CATEGORY_ROAD_COOLDOWN_MS", 5000)
        category_road_flow_mag_min = _float("CATEGORY_ROAD_FLOW_MAG_MIN", 1.0)
        # LLM
        llm_enabled = os.getenv("LLM_ENABLED", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        llm_provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
        llm_model_fast = os.getenv("LLM_MODEL_FAST", "google/gemini-2.5-flash").strip()
        try:
            llm_timeout_sec = int(os.getenv("LLM_TIMEOUT_SEC", "20"))
        except Exception:
            llm_timeout_sec = 20
        # UI
        try:
            ui_frame_refresh_ms = max(500, int(os.getenv("UI_FRAME_REFRESH_MS", "10000")))
        except Exception:
            ui_frame_refresh_ms = 10000
        try:
            llm_burst_window_ms = max(200, int(os.getenv("LLM_BURST_WINDOW_MS", "1500")))
        except Exception:
            llm_burst_window_ms = 1500

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
            detector_method=detector_method,
            roi_diff_alpha=roi_diff_alpha,
            roi_diff_threshold=roi_diff_threshold,
            roi_diff_min_pixels=roi_diff_min_pixels,
            roi_diff_cooldown_ms=roi_diff_cooldown_ms,
            category_gate_stable_ms=category_gate_stable_ms,
            category_gate_phash_min_bits=category_gate_phash_min_bits,
            category_gate_roi_diff_threshold=category_gate_roi_diff_threshold,
            category_gate_min_pixels=category_gate_min_pixels,
            category_gate_cooldown_ms=category_gate_cooldown_ms,
            category_gate_flow_mag_min=category_gate_flow_mag_min,
            category_parking_stable_ms=category_parking_stable_ms,
            category_parking_phash_min_bits=category_parking_phash_min_bits,
            category_parking_roi_diff_threshold=category_parking_roi_diff_threshold,
            category_parking_min_pixels=category_parking_min_pixels,
            category_parking_cooldown_ms=category_parking_cooldown_ms,
            category_road_stable_ms=category_road_stable_ms,
            category_road_phash_min_bits=category_road_phash_min_bits,
            category_road_roi_diff_threshold=category_road_roi_diff_threshold,
            category_road_min_pixels=category_road_min_pixels,
            category_road_cooldown_ms=category_road_cooldown_ms,
            category_road_flow_mag_min=category_road_flow_mag_min,
            llm_enabled=llm_enabled,
            llm_provider=llm_provider,
            llm_model_fast=llm_model_fast,
            llm_timeout_sec=llm_timeout_sec,
            ui_frame_refresh_ms=ui_frame_refresh_ms,
            llm_burst_window_ms=llm_burst_window_ms,
        )
