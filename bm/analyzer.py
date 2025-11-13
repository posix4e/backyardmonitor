from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class AnalysisTaskMeta:
    id: str
    ts: float
    image_path: str  # relative to data dir, e.g., analysis/imgs/<id>.jpg
    json_path: str  # relative to data dir, e.g., analysis/tasks/<id>.json


class AnalysisWorker:
    def __init__(self, data_dir: Path, interval_sec: int = 15):
        self.data_dir = data_dir
        self.interval_sec = max(3, int(interval_sec))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # folders
        self.root = self.data_dir / "analysis"
        self.imgs = self.root / "imgs"
        self.tasks = self.root / "tasks"
        self.results = self.root / "results"
        for d in (self.imgs, self.tasks, self.results):
            d.mkdir(parents=True, exist_ok=True)

        # capture getter — to be set by app
        self.get_latest_frame = None  # type: ignore
        self.get_spots = None  # type: ignore

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="AnalysisWorker", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                # keep going
                pass
            # wait
            for _ in range(self.interval_sec * 10):
                if self._stop.is_set():
                    break
                time.sleep(0.1)

    def _tick(self):
        if not callable(self.get_latest_frame):
            return
        latest = self.get_latest_frame()
        if not latest:
            return
        frame, ts = latest
        # downscale to max width 800 for analysis
        h, w = frame.shape[:2]
        if w > 800:
            scale = 800.0 / w
            dw, dh = int(w * scale), int(h * scale)
            img = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        else:
            img = frame
        # write image and task json
        tid = str(int(ts * 1000))
        img_rel = f"analysis/imgs/{tid}.jpg"
        img_path = self.data_dir / img_rel
        cv2.imwrite(str(img_path), img)
        # gather spots
        spots = []
        if callable(self.get_spots):
            try:
                spots = self.get_spots() or []
            except Exception:
                spots = []
        task = {
            "id": tid,
            "ts": ts,
            "image": img_rel,
            "size": {"w": int(w), "h": int(h)},
            "spots": spots,
        }
        json_rel = f"analysis/tasks/{tid}.json"
        json_path = self.data_dir / json_rel
        import json

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(task, f)
