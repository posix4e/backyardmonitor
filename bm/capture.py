from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


Frame = Tuple[np.ndarray, float]  # (BGR image, timestamp)


@dataclass
class CaptureState:
    running: bool = False
    width: int = 0
    height: int = 0
    last_ts: float = 0.0


class VideoCaptureWorker:
    def __init__(self, source: str, max_fps: float | None = None,
                 idle_resync_ms: int = 2500,
                 fail_resync_count: int = 10,
                 reopen_delay_ms: int = 300):
        self.source = source
        self.max_fps = float(max_fps) if (max_fps is not None and max_fps > 0) else 0.0
        self.idle_resync_ms = int(idle_resync_ms)
        self.fail_resync_count = int(fail_resync_count)
        self.reopen_delay_ms = int(reopen_delay_ms)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self.state = CaptureState(running=False)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="VideoCapture", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        import logging

        log = logging.getLogger("uvicorn.error")

        def _open() -> cv2.VideoCapture:
            import os

            # Prefer TCP transport for RTSP to avoid UDP packet loss.
            # Use OpenCV FFmpeg backend with capture options when available.
            backend = getattr(cv2, "CAP_FFMPEG", 1900)
            if str(self.source).lower().startswith("rtsp"):
                # Allow override via env var: RTSP_TRANSPORT=udp|tcp (default tcp)
                rtsp_transport = os.getenv("RTSP_TRANSPORT", "tcp").lower()
                # Apply FFmpeg capture options: transport + a sane socket timeout.
                # stimeout is in microseconds; here ~5s.
                # Additional FFmpeg opts to reduce latency and avoid grey frames after stalls
                # - fflags=nobuffer, flags=low_delay, reorder_queue_size=0, max_delay ~0.5s
                # - small probesize/analyzeduration to start decoding fast
                # - rw_timeout/stimeout socket timeouts
                # - use_wallclock_as_timestamps=1 to progress timestamps
                opts = (
                    f"rtsp_transport;{rtsp_transport}"
                    "|fflags;nobuffer|flags;low_delay|reorder_queue_size;0|max_delay;500000"
                    "|probesize;32768|analyzeduration;0|rw_timeout;5000000|stimeout;5000000|use_wallclock_as_timestamps;1"
                )
                # Avoid clobbering unrelated options if user already set them; merge best-effort.
                existing = os.getenv("OPENCV_FFMPEG_CAPTURE_OPTIONS")
                if existing and existing != opts:
                    # Simple merge without duplicates
                    kv = {tuple(p.split(";", 1)) for p in existing.split("|") if ";" in p}
                    for part in opts.split("|"):
                        if ";" in part:
                            kv.add(tuple(part.split(";", 1)))
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(
                        f"{k};{v}" for (k, v) in kv
                    )
                else:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
                try:
                    cap_local = cv2.VideoCapture(self.source, backend)
                except Exception:
                    cap_local = cv2.VideoCapture(self.source)
            else:
                cap_local = cv2.VideoCapture(self.source)
            if not cap_local.isOpened():
                time.sleep(max(0.05, self.reopen_delay_ms / 1000.0))
                try:
                    cap_local = cv2.VideoCapture(self.source, backend)
                except Exception:
                    cap_local = cv2.VideoCapture(self.source)
            return cap_local

        def _valid_frame(img: np.ndarray) -> bool:
            try:
                if img is None:
                    return False
                if not isinstance(img, np.ndarray):
                    return False
                if img.ndim not in (2, 3):
                    return False
                h, w = img.shape[:2]
                if h <= 0 or w <= 0:
                    return False
                # Most streams are color; tolerate grayscale, but ensure dtype/values sane
                if img.dtype != np.uint8:
                    return False
                # Fast heuristics: reject near-uniform or all-black/white frames
                small = cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA)
                if small.ndim == 3:
                    small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                mean = float(np.mean(small))
                std = float(np.std(small))
                # Reject nearly all-dark or all-bright frames
                if mean < 3.0 or mean > 252.0:
                    return False
                # Reject extremely low-detail frames
                if std < 3.5:
                    return False
                # If almost all pixels are at extreme ends, reject
                # Compute simple saturation ratio using thresholds
                zeros = float((small <= 1).sum())
                highs = float((small >= 254).sum())
                total = float(small.size)
                if total > 0 and ((zeros / total) > 0.98 or (highs / total) > 0.98):
                    return False
                return True
            except Exception:
                return False

        cap = _open()
        try:
            # Keep buffer small to reduce latency/grey frames after stalls (best-effort).
            cap.set(getattr(cv2, "CAP_PROP_BUFFERSIZE", 38), 2)
        except Exception:
            pass
        self.state.running = cap.isOpened()
        min_period = (1.0 / self.max_fps) if self.max_fps > 0 else 0.0
        last_push = 0.0
        last_good_ts = time.time()
        consecutive_fail = 0
        warmup_valid_needed = 3  # discard a few early frames after (re)open
        try:
            while not self._stop.is_set() and cap.isOpened():
                # simple fps limiter to reduce CPU burn from decoding
                if min_period > 0.0:
                    now = time.time()
                    dt = now - last_push
                    if dt < min_period:
                        time.sleep(min_period - dt)

                ok, frame = cap.read()
                ts = time.time()
                if not ok or frame is None or not _valid_frame(frame):
                    consecutive_fail += 1
                    # Resync on too many consecutive failures or idle
                    idle_ms = (ts - last_good_ts) * 1000.0
                    if (
                        consecutive_fail >= max(1, self.fail_resync_count)
                        or (self.idle_resync_ms > 0 and idle_ms >= self.idle_resync_ms)
                    ):
                        try:
                            log.info(
                                f"capture resync: fails={consecutive_fail} idle_ms={idle_ms:.0f}"
                            )
                        except Exception:
                            pass
                        try:
                            cap.release()
                        except Exception:
                            pass
                        time.sleep(max(0.05, self.reopen_delay_ms / 1000.0))
                        cap = _open()
                        try:
                            cap.set(getattr(cv2, "CAP_PROP_BUFFERSIZE", 38), 2)
                        except Exception:
                            pass
                        self.state.running = cap.isOpened()
                        consecutive_fail = 0
                        warmup_valid_needed = 3
                        # Continue loop to try next read
                        continue
                    time.sleep(0.05)
                    continue
                last_push = ts
                last_good_ts = ts
                consecutive_fail = 0
                if warmup_valid_needed > 0:
                    # Skip publishing first few valid frames after (re)open; often grey/blank
                    warmup_valid_needed -= 1
                    continue
                h, w = frame.shape[:2]
                self.state.width, self.state.height = w, h
                self.state.last_ts = ts
                with self._lock:
                    self._latest = (frame, ts)
        finally:
            self.state.running = False
            try:
                cap.release()
            except Exception:
                pass

    @staticmethod
    def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not ok:
            raise RuntimeError("Failed to encode JPEG")
        return bytes(buf)
