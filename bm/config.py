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

    @classmethod
    def from_env(cls) -> "Settings":
        rtsp_url = os.getenv("RTSP_URL")
        data_dir = Path(os.getenv("DATA_DIR", "data")).resolve()
        auto_start = os.getenv("AUTO_START", "true").lower() in {"1", "true", "yes", "on"}
        analysis_enabled = os.getenv("ANALYSIS_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
        try:
            analysis_interval_sec = int(os.getenv("ANALYSIS_INTERVAL_SEC", "15"))
        except Exception:
            analysis_interval_sec = 15
        return cls(rtsp_url=rtsp_url, data_dir=data_dir, auto_start=auto_start,
                   analysis_enabled=analysis_enabled, analysis_interval_sec=analysis_interval_sec)
