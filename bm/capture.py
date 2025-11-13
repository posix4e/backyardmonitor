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
    def __init__(self, source: str):
        self.source = source
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
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            # attempt a delayed retry for streams that come up slowly
            time.sleep(1.0)
            cap = cv2.VideoCapture(self.source)
        self.state.running = cap.isOpened()
        try:
            while not self._stop.is_set() and cap.isOpened():
                ok, frame = cap.read()
                ts = time.time()
                if not ok or frame is None:
                    time.sleep(0.05)
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
