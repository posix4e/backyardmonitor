from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from importlib.metadata import PackageNotFoundError, version as pkg_version

from .config import Settings
from .capture import VideoCaptureWorker
from .storage import EventStore
from .analyzer import AnalysisWorker
from .zones import load_zones, save_zones
from .llm import LLMWorker, LLMConfig


class AppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.data_dir = settings.data_dir
        self.frames_dir = self.data_dir / "frames"
        self.analysis_dir = self.data_dir / "analysis"
        self.db_path = self.data_dir / "events.db"
        self.zones_path = self.data_dir / "zones.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.events = EventStore(self.db_path)
        self.capture: Optional[VideoCaptureWorker] = None
        self.analysis: Optional[AnalysisWorker] = None
        self.llm: Optional[LLMWorker] = None
        # in-memory spot image signature tracker: { spot_id: {sig: str, since: float} }
        self.spot_tracker: dict[str, dict[str, float | str]] = {}
        # Comparison filtering baseline (events older than this ts are ignored for prev/compare)
        self.compare_baseline_ts: float = 0.0
        # Cached JPEG to avoid re-encoding every request
        self._last_jpg: bytes | None = None
        self._last_jpg_ts: float = 0.0
        # Previous frame-wide signature removed; we no longer suppress global-lighting changes

    def ensure_capture(self) -> None:
        if not self.settings.rtsp_url:
            return
        if self.capture is None:
            self.capture = VideoCaptureWorker(
                self.settings.rtsp_url,
                max_fps=self.settings.capture_max_fps,
                idle_resync_ms=self.settings.capture_idle_resync_ms,
                fail_resync_count=self.settings.capture_fail_resync_count,
                reopen_delay_ms=self.settings.capture_reopen_delay_ms,
            )
        # start only if not running
        self.capture.start()

    def restart_capture(self) -> None:
        try:
            if self.capture:
                self.capture.stop()
        except Exception:
            pass
        try:
            self.capture = None
        except Exception:
            pass
        self.ensure_capture()


def create_app() -> FastAPI:
    settings = Settings.from_env()
    state = AppState(settings)

    app = FastAPI(title="Backyard Monitor (Rewrite)")

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    # Serve data dir to expose analysis images
    app.mount("/data", StaticFiles(directory=state.data_dir), name="data")

    @app.on_event("startup")
    async def on_startup() -> None:
        if settings.rtsp_url and settings.auto_start:
            state.ensure_capture()
        # Start background analysis task writer if enabled
        if settings.analysis_enabled:
            worker = AnalysisWorker(state.data_dir, settings.analysis_interval_sec)
            worker.get_latest_frame = lambda: (
                state.capture.latest() if state.capture else None
            )

            def _get_spots():
                z = load_zones(state.zones_path)
                return [
                    {"id": s.id, "name": s.name, "polygon": s.polygon} for s in z.spots
                ]

            worker.get_spots = _get_spots
            worker.start()
            state.analysis = worker
        # Background retention sweep (hourly)
        try:
            import threading, time as _t

            def _retention_loop():
                while True:
                    try:
                        apply_retention_sweep()
                    except Exception:
                        pass
                    _t.sleep(3600)

            threading.Thread(target=_retention_loop, daemon=True).start()
        except Exception:
            pass
        # Start LLM worker if enabled
        try:
            if settings.llm_enabled:
                cfg = LLMConfig(
                    provider=settings.llm_provider,
                    model_fast=settings.llm_model_fast,
                    timeout_sec=int(settings.llm_timeout_sec or 20),
                )
                llm = LLMWorker(state.frames_dir, state.events, cfg)
                llm.start()
                state.llm = llm
        except Exception:
            pass

    @app.get("/", response_class=HTMLResponse)
    async def index():
        path = Path(__file__).parent / "static" / "index.html"
        return path.read_text(encoding="utf-8")

    # Single-page app; legacy routes removed

    @app.get("/api/status", response_class=JSONResponse)
    async def api_status():
        zones = load_zones(state.zones_path)
        running = state.capture.state.running if state.capture else False
        last_ts = state.capture.state.last_ts if state.capture else 0
        return {
            "running": running,
            "last_ts": last_ts,
            "width": (state.capture.state.width if state.capture else 0),
            "height": (state.capture.state.height if state.capture else 0),
            "zones_defined": bool(zones.spots),
            "events": state.events.recent(20),
        }

    @app.get("/api/version", response_class=JSONResponse)
    async def api_version():
        try:
            ver = pkg_version("backyardmonitor")
        except PackageNotFoundError:
            ver = "0.0.0"
        import os

        build = os.getenv("GIT_SHA") or os.getenv("APP_BUILD") or ""
        return {"name": "backyardmonitor", "version": ver, "build": build}

    @app.get("/api/badge", response_class=JSONResponse)
    async def api_badge():
        try:
            ver = pkg_version("backyardmonitor")
        except PackageNotFoundError:
            ver = "0.0.0"
        import os

        build = os.getenv("GIT_SHA") or os.getenv("APP_BUILD") or ""
        label = os.getenv("BADGE_LABEL") or "deployed"
        color = os.getenv("BADGE_COLOR") or "blue"
        msg = f"v{ver}"
        if build:
            msg = f"{msg} ({build[:7]})"
        return {
            "schemaVersion": 1,
            "label": label,
            "message": msg,
            "color": color,
            "cacheSeconds": int(os.getenv("BADGE_CACHE_SECONDS") or 300),
        }

    @app.get("/api/spot_stats", response_class=JSONResponse)
    async def api_spot_stats():
        import numpy as np
        import cv2

        zones = load_zones(state.zones_path)
        now = time.time()
        latest = state.capture.latest() if state.capture else None
        frame = latest[0] if latest else None

        method = (state.settings.detector_method or "phash").lower()

        def _category_params(category: str | None):
            # Default unlabeled spots to 'spot' semantics (alias of previous 'parking')
            c = (category or "spot").strip().lower()
            if c == "gate":
                return {
                    "stable_ms": int(state.settings.category_gate_stable_ms or 0),
                    "phash_min_bits": int(
                        state.settings.category_gate_phash_min_bits or 0
                    ),
                    "roi_diff_threshold": float(
                        state.settings.category_gate_roi_diff_threshold or 0.0
                    ),
                    "roi_diff_min_pixels": int(
                        state.settings.category_gate_min_pixels
                        or state.settings.roi_diff_min_pixels
                        or 0
                    ),
                    "cooldown_ms": int(
                        state.settings.category_gate_cooldown_ms
                        or state.settings.roi_diff_cooldown_ms
                        or 0
                    ),
                    "flow_mag_min": float(
                        state.settings.category_gate_flow_mag_min or 0.0
                    ),
                }
            elif c in ("spot", "parking"):
                return {
                    "stable_ms": int(state.settings.category_parking_stable_ms or 0),
                    "phash_min_bits": int(
                        state.settings.category_parking_phash_min_bits or 0
                    ),
                    "roi_diff_threshold": float(
                        state.settings.category_parking_roi_diff_threshold or 0.0
                    ),
                    "roi_diff_min_pixels": int(
                        state.settings.category_parking_min_pixels
                        or state.settings.roi_diff_min_pixels
                        or 0
                    ),
                    "cooldown_ms": int(
                        state.settings.category_parking_cooldown_ms
                        or state.settings.roi_diff_cooldown_ms
                        or 0
                    ),
                    "flow_mag_min": 0.0,
                }
            elif c == "road":
                return {
                    "stable_ms": int(state.settings.category_road_stable_ms or 0),
                    "phash_min_bits": int(
                        state.settings.category_road_phash_min_bits or 0
                    ),
                    "roi_diff_threshold": float(
                        state.settings.category_road_roi_diff_threshold or 0.0
                    ),
                    "roi_diff_min_pixels": int(
                        state.settings.category_road_min_pixels
                        or state.settings.roi_diff_min_pixels
                        or 0
                    ),
                    "cooldown_ms": int(
                        state.settings.category_road_cooldown_ms
                        or state.settings.roi_diff_cooldown_ms
                        or 0
                    ),
                    "flow_mag_min": float(
                        state.settings.category_road_flow_mag_min or 0.0
                    ),
                }
            # Fallback to global
            return {
                "stable_ms": int(state.settings.phash_stable_ms or 0),
                "phash_min_bits": int(state.settings.phash_min_bits or 0),
                "roi_diff_threshold": float(state.settings.roi_diff_threshold or 0.02),
                "roi_diff_min_pixels": int(state.settings.roi_diff_min_pixels or 0),
                "cooldown_ms": int(state.settings.roi_diff_cooldown_ms or 0),
                "flow_mag_min": 0.0,
            }

        def _hamming(a: str | None, b: str | None) -> int:
            if not a or not b or len(a) != len(b):
                return 999
            return sum(1 for i in range(len(a)) if a[i] != b[i])

        def _roi_phash(img) -> str | None:
            try:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                g = cv2.GaussianBlur(g, (3, 3), 0)
                small = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA)
                smallf = small.astype(np.float32)
                dct = cv2.dct(smallf)
                dct_low = dct[:8, :8].copy()
                vals = dct_low.flatten()
                median = (
                    float(np.median(vals[1:]))
                    if len(vals) > 1
                    else float(np.median(vals))
                )
                bits = (vals > median).astype(np.uint8)
                return "".join("1" if b else "0" for b in bits)
            except Exception:
                return None

        stats = []

        # Helper: whether there is a previous event for this spot after the compare baseline
        def _has_prev_event(spot_id: str) -> bool:
            try:
                baseline = float(state.compare_baseline_ts or 0.0)
                prevs = state.events.spot_events(spot_id, 1)
                if not prevs:
                    return False
                ts_prev = float(prevs[0].get("ts") or 0.0)
                return ts_prev > 0.0 and (baseline <= 0.0 or ts_prev >= baseline)
            except Exception:
                return False

        for s in zones.spots:
            xs = [p[0] for p in s.polygon] or [0]
            ys = [p[1] for p in s.polygon] or [0]
            x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
            x2, y2 = int(max(xs)), int(max(ys))
            sig = None
            rec = state.spot_tracker.get(s.id)
            if frame is not None and x2 > x1 and y2 > y1:
                crop = frame[
                    max(0, y1) : min(frame.shape[0], y2),
                    max(0, x1) : min(frame.shape[1], x2),
                ]
                if crop.size > 0:
                    # Always compute a pHash for UI change visualization
                    sig = _roi_phash(crop)
                    if method == "roi_diff":
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        # initialize or update baseline
                        if not rec or rec.get("baseline") is None:
                            state.spot_tracker[s.id] = rec = {
                                "baseline": gray.astype(np.float32),
                                "since": now,
                            }
                        base = rec.get("baseline")
                        if isinstance(base, np.ndarray) and base.size == gray.size:
                            base8 = cv2.convertScaleAbs(base)
                            diff = cv2.absdiff(gray, base8)
                            # ROI-diff thresholding and state machine
                            cp = _category_params(getattr(s, "category", None))
                            thr_ratio = float(
                                cp.get("roi_diff_threshold")
                                or state.settings.roi_diff_threshold
                                or 0.02
                            )
                            t = int(max(1, min(255, round(255.0 * thr_ratio))))
                            _, mask = cv2.threshold(diff, t, 255, cv2.THRESH_BINARY)
                            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
                            changed_pixels = int(cv2.countNonZero(mask))
                            ratio = float(changed_pixels) / float(
                                mask.size if mask.size else 1
                            )
                            # State machine with hysteresis
                            spot_stable_ms = getattr(s, "stable_ms", None)
                            stable_ms = int(
                                spot_stable_ms
                                if spot_stable_ms is not None
                                else cp.get(
                                    "stable_ms", state.settings.phash_stable_ms or 0
                                )
                            )
                            min_pixels = int(
                                cp.get("roi_diff_min_pixels")
                                or state.settings.roi_diff_min_pixels
                                or 0
                            )
                            enter = ratio >= thr_ratio and changed_pixels >= max(
                                0, min_pixels
                            )
                            exit_thr = max(0.0, min(thr_ratio * 0.6, thr_ratio - 0.02))
                            leave = ratio <= exit_thr or changed_pixels < max(
                                0, int(min_pixels * 0.7)
                            )
                            state_key = str(rec.get("state") or "empty")
                            # track candidates
                            enter_since = float(rec.get("cand_enter_since") or 0)
                            exit_since = float(rec.get("cand_exit_since") or 0)
                            if state_key == "empty":
                                if enter:
                                    if not enter_since:
                                        rec["cand_enter_since"] = now
                                    elif ((now - enter_since) * 1000.0) >= stable_ms:
                                        # cooldown
                                        try:
                                            cd_ms = int(
                                                cp.get("cooldown_ms")
                                                or state.settings.roi_diff_cooldown_ms
                                                or 0
                                            )
                                        except Exception:
                                            cd_ms = 0
                                        last_evt_ts = float(
                                            rec.get("last_evt_ts") or 0.0
                                        )
                                        if not (
                                            cd_ms > 0
                                            and last_evt_ts
                                            and ((now - last_evt_ts) * 1000.0) < cd_ms
                                        ):
                                            # Optional flow requirement
                                            ok_flow = True
                                            try:
                                                req = float(
                                                    cp.get("flow_mag_min") or 0.0
                                                )
                                                if req > 0.0:
                                                    prev_g = rec.get("prev_gray")
                                                    if (
                                                        isinstance(prev_g, np.ndarray)
                                                        and prev_g.shape == gray.shape
                                                    ):
                                                        flow = cv2.calcOpticalFlowFarneback(
                                                            prev_g,
                                                            gray,
                                                            None,
                                                            0.5,
                                                            1,
                                                            15,
                                                            3,
                                                            5,
                                                            1.2,
                                                            0,
                                                        )
                                                        mag = np.sqrt(
                                                            flow[..., 0] ** 2
                                                            + flow[..., 1] ** 2
                                                        )
                                                        ok_flow = (
                                                            float(np.mean(mag)) >= req
                                                        )
                                            except Exception:
                                                ok_flow = True
                                            # Low-variance guard
                                            if ok_flow:
                                                try:
                                                    small = cv2.resize(
                                                        gray,
                                                        (64, 36),
                                                        interpolation=cv2.INTER_AREA,
                                                    )
                                                    mean = float(np.mean(small))
                                                    std = float(np.std(small))
                                                    if (mean < 6.0 or mean > 245.0) or (
                                                        std < 6.0
                                                    ):
                                                        ok_flow = False
                                                except Exception:
                                                    pass
                                            if ok_flow:
                                                # emit enter event
                                                try:
                                                    ts_ms = int(now * 1000)
                                                    base_name = f"evt_{ts_ms}_{s.id}"
                                                    full_path = (
                                                        state.frames_dir
                                                        / f"{base_name}_full.jpg"
                                                    )
                                                    crop_path = (
                                                        state.frames_dir
                                                        / f"{base_name}_crop.jpg"
                                                    )
                                                    thumb_path = (
                                                        state.frames_dir
                                                        / f"{base_name}_thumb.jpg"
                                                    )
                                                    params = [
                                                        int(cv2.IMWRITE_JPEG_QUALITY),
                                                        int(
                                                            state.settings.jpeg_quality
                                                            or 80
                                                        ),
                                                    ]
                                                    if state.settings.store_full_frames:
                                                        try:
                                                            cv2.imwrite(
                                                                str(full_path),
                                                                frame,
                                                                params,
                                                            )
                                                        except Exception:
                                                            pass
                                                    if state.settings.store_crops:
                                                        try:
                                                            cv2.imwrite(
                                                                str(crop_path),
                                                                crop,
                                                                params,
                                                            )
                                                        except Exception:
                                                            pass
                                                    if state.settings.store_thumbs:
                                                        try:
                                                            h_c, w_c = crop.shape[:2]
                                                            if w_c > 0 and h_c > 0:
                                                                new_w = 240
                                                                new_h = max(
                                                                    1,
                                                                    int(
                                                                        h_c
                                                                        * (new_w / w_c)
                                                                    ),
                                                                )
                                                                thumb = cv2.resize(
                                                                    crop,
                                                                    (new_w, new_h),
                                                                    interpolation=cv2.INTER_AREA,
                                                                )
                                                                cv2.imwrite(
                                                                    str(thumb_path),
                                                                    thumb,
                                                                    params,
                                                                )
                                                        except Exception:
                                                            pass
                                                    meta = {
                                                        "spot_id": s.id,
                                                        "ratio": ratio,
                                                        "method": "roi_diff",
                                                        "state": "enter",
                                                    }
                                                    try:
                                                        meta["has_prev"] = bool(_has_prev_event(s.id))
                                                        if not meta["has_prev"]:
                                                            meta["significant"] = False
                                                    except Exception:
                                                        pass
                                                    if (
                                                        state.settings.store_full_frames
                                                        and full_path.exists()
                                                    ):
                                                        meta["image_full"] = (
                                                            full_path.name
                                                        )
                                                    if (
                                                        state.settings.store_crops
                                                        and crop_path.exists()
                                                    ):
                                                        meta["image_crop"] = (
                                                            crop_path.name
                                                        )
                                                    if (
                                                        state.settings.store_thumbs
                                                        and thumb_path.exists()
                                                    ):
                                                        meta["image_thumb"] = (
                                                            thumb_path.name
                                                        )
                                                    state.events.add(
                                                        "spot_change", meta
                                                    )
                                                except Exception:
                                                    pass
                                                rec["state"] = "occupied"
                                                rec["since"] = now
                                                rec["last_evt_ts"] = now
                                                rec.pop("cand_enter_since", None)
                            else:  # occupied
                                if leave:
                                    if not exit_since:
                                        rec["cand_exit_since"] = now
                                    elif ((now - exit_since) * 1000.0) >= max(
                                        300, int(stable_ms * 0.7)
                                    ):
                                        # emit exit event
                                        try:
                                            ts_ms = int(now * 1000)
                                            base_name = f"evt_{ts_ms}_{s.id}"
                                            full_path = (
                                                state.frames_dir
                                                / f"{base_name}_full.jpg"
                                            )
                                            crop_path = (
                                                state.frames_dir
                                                / f"{base_name}_crop.jpg"
                                            )
                                            thumb_path = (
                                                state.frames_dir
                                                / f"{base_name}_thumb.jpg"
                                            )
                                            params = [
                                                int(cv2.IMWRITE_JPEG_QUALITY),
                                                int(state.settings.jpeg_quality or 80),
                                            ]
                                            if state.settings.store_full_frames:
                                                try:
                                                    cv2.imwrite(
                                                        str(full_path), frame, params
                                                    )
                                                except Exception:
                                                    pass
                                            if state.settings.store_crops:
                                                try:
                                                    cv2.imwrite(
                                                        str(crop_path), crop, params
                                                    )
                                                except Exception:
                                                    pass
                                            if state.settings.store_thumbs:
                                                try:
                                                    h_c, w_c = crop.shape[:2]
                                                    if w_c > 0 and h_c > 0:
                                                        new_w = 240
                                                        new_h = max(
                                                            1, int(h_c * (new_w / w_c))
                                                        )
                                                        thumb = cv2.resize(
                                                            crop,
                                                            (new_w, new_h),
                                                            interpolation=cv2.INTER_AREA,
                                                        )
                                                        cv2.imwrite(
                                                            str(thumb_path),
                                                            thumb,
                                                            params,
                                                        )
                                                except Exception:
                                                    pass
                                            meta = {
                                                "spot_id": s.id,
                                                "ratio": ratio,
                                                "method": "roi_diff",
                                                "state": "exit",
                                            }
                                            try:
                                                meta["has_prev"] = bool(_has_prev_event(s.id))
                                                if not meta["has_prev"]:
                                                    meta["significant"] = False
                                            except Exception:
                                                pass
                                            if (
                                                state.settings.store_full_frames
                                                and full_path.exists()
                                            ):
                                                meta["image_full"] = full_path.name
                                            if (
                                                state.settings.store_crops
                                                and crop_path.exists()
                                            ):
                                                meta["image_crop"] = crop_path.name
                                            if (
                                                state.settings.store_thumbs
                                                and thumb_path.exists()
                                            ):
                                                meta["image_thumb"] = thumb_path.name
                                            ev_id = state.events.add(
                                                "spot_change", meta
                                            )
                                            try:
                                                if state.llm and bool(meta.get("has_prev")):
                                                    state.llm.queue(int(ev_id))
                                            except Exception:
                                                pass
                                        except Exception:
                                            pass
                                        rec["state"] = "empty"
                                        rec["since"] = now
                                        rec["last_evt_ts"] = now
                                        rec.pop("cand_exit_since", None)
                            # update baseline after diff
                            try:
                                cv2.accumulateWeighted(
                                    gray,
                                    base,
                                    float(state.settings.roi_diff_alpha or 0.05),
                                )
                            except Exception:
                                pass
                            # keep last gray for optional flow checks
                            try:
                                rec["prev_gray"] = gray
                            except Exception:
                                pass
                    else:
                        # pHash detection path (previous behavior sans global-lighting suppression)
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        gray = cv2.GaussianBlur(gray, (3, 3), 0)
                        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
                        smallf = small.astype(np.float32)
                        dct = cv2.dct(smallf)
                        dct_low = dct[:8, :8].copy()
                        vals = dct_low.flatten()
                        vals_no_dc = vals[1:]
                        median = float(np.median(vals_no_dc))
                        bits = (vals > median).astype(np.uint8)
                        sig = "".join("1" if b else "0" for b in bits)
                        if sig is not None:
                            if not rec:
                                state.spot_tracker[s.id] = {"sig": sig, "since": now}
                                rec = state.spot_tracker[s.id]
                            elif rec.get("sig") != sig:
                                prev_sig = str(rec.get("sig") or "")
                                spot_min_bits = getattr(s, "min_bits", None)
                                spot_stable_ms = getattr(s, "stable_ms", None)
                                cp = _category_params(getattr(s, "category", None))
                                min_bits = int(
                                    spot_min_bits
                                    if spot_min_bits is not None
                                    else (
                                        cp.get("phash_min_bits")
                                        or state.settings.phash_min_bits
                                        or 0
                                    )
                                )
                                stable_ms = int(
                                    spot_stable_ms
                                    if spot_stable_ms is not None
                                    else (
                                        cp.get("stable_ms")
                                        or state.settings.phash_stable_ms
                                        or 0
                                    )
                                )
                                cand_sig = rec.get("cand_sig")
                                cand_since = float(rec.get("cand_since") or 0)
                                if cand_sig != sig:
                                    rec["cand_sig"] = sig
                                    rec["cand_since"] = now
                                else:
                                    stable = (
                                        (((now - cand_since) * 1000.0) >= stable_ms)
                                        if stable_ms > 0
                                        else True
                                    )
                                    delta_bits = _hamming(prev_sig, sig)
                                    if stable and delta_bits >= min_bits:
                                        # low-detail rejection
                                        try:
                                            small2 = cv2.resize(
                                                gray,
                                                (64, 36),
                                                interpolation=cv2.INTER_AREA,
                                            )
                                            mean = float(np.mean(small2))
                                            std = float(np.std(small2))
                                            if (mean < 6.0 or mean > 245.0) or (
                                                std < 6.0
                                            ):
                                                raise RuntimeError("lowvar")
                                        except Exception:
                                            pass
                                        try:
                                            ts_ms = int(now * 1000)
                                            base_name = f"evt_{ts_ms}_{s.id}"
                                            full_path = (
                                                state.frames_dir
                                                / f"{base_name}_full.jpg"
                                            )
                                            crop_path = (
                                                state.frames_dir
                                                / f"{base_name}_crop.jpg"
                                            )
                                            thumb_path = (
                                                state.frames_dir
                                                / f"{base_name}_thumb.jpg"
                                            )
                                            params = [
                                                int(cv2.IMWRITE_JPEG_QUALITY),
                                                int(state.settings.jpeg_quality or 80),
                                            ]
                                            if state.settings.store_full_frames:
                                                try:
                                                    cv2.imwrite(
                                                        str(full_path), frame, params
                                                    )
                                                except Exception:
                                                    pass
                                            if state.settings.store_crops:
                                                try:
                                                    cv2.imwrite(
                                                        str(crop_path), crop, params
                                                    )
                                                except Exception:
                                                    pass
                                            if state.settings.store_thumbs:
                                                try:
                                                    h_c, w_c = crop.shape[:2]
                                                    if w_c > 0 and h_c > 0:
                                                        new_w = 240
                                                        new_h = max(
                                                            1, int(h_c * (new_w / w_c))
                                                        )
                                                        thumb = cv2.resize(
                                                            crop,
                                                            (new_w, new_h),
                                                            interpolation=cv2.INTER_AREA,
                                                        )
                                                        cv2.imwrite(
                                                            str(thumb_path),
                                                            thumb,
                                                            params,
                                                        )
                                                except Exception:
                                                    pass
                                            meta = {
                                                "spot_id": s.id,
                                                "prev_sig": prev_sig,
                                                "new_sig": sig,
                                                "delta_bits": delta_bits,
                                            }
                                            try:
                                                meta["has_prev"] = bool(_has_prev_event(s.id))
                                                if not meta["has_prev"]:
                                                    meta["significant"] = False
                                            except Exception:
                                                pass
                                            if (
                                                state.settings.store_full_frames
                                                and full_path.exists()
                                            ):
                                                meta["image_full"] = full_path.name
                                            if (
                                                state.settings.store_crops
                                                and crop_path.exists()
                                            ):
                                                meta["image_crop"] = crop_path.name
                                            if (
                                                state.settings.store_thumbs
                                                and thumb_path.exists()
                                            ):
                                                meta["image_thumb"] = thumb_path.name
                                            ev_id = state.events.add("spot_change", meta)
                                            try:
                                                if state.llm and bool(meta.get("has_prev")):
                                                    state.llm.queue(int(ev_id))
                                            except Exception:
                                                pass
                                        except Exception:
                                            pass
                                        state.spot_tracker[s.id] = {
                                            "sig": sig,
                                            "since": now,
                                        }
                                        rec = state.spot_tracker[s.id]
            since = rec.get("since") if rec else None
            duration = (now - float(since)) if since else None
            stats.append(
                {
                    "id": s.id,
                    "name": s.name or s.id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "w": max(0, x2 - x1),
                    "h": max(0, y2 - y1),
                    "since": since,
                    "duration_sec": duration,
                    "sig": sig,
                }
            )
        return {"spots": stats, "ts": now}

    def apply_retention_sweep():
        import time

        # Time-based retention
        days = max(0, int(state.settings.retain_days or 0))
        now = time.time()
        deleted = 0
        if days > 0:
            cutoff = now - days * 86400
            old = state.events.older_than(cutoff, limit=5000)
            for e in old:
                meta = e.get("meta") or {}
                for k in ("image_full", "image_crop", "image_thumb"):
                    name = meta.get(k)
                    if name:
                        p = state.frames_dir / str(name)
                        if p.exists():
                            try:
                                p.unlink()
                            except Exception:
                                pass
                if state.events.delete(int(e["id"])):
                    deleted += 1
        # Event count cap
        total = state.events.count()
        max_e = max(100, int(state.settings.max_events or 5000))
        if total > max_e:
            overflow = total - max_e
            olds = state.events.oldest(overflow)
            for e in olds:
                meta = e.get("meta") or {}
                for k in ("image_full", "image_crop", "image_thumb"):
                    name = meta.get(k)
                    if name:
                        p = state.frames_dir / str(name)
                        if p.exists():
                            try:
                                p.unlink()
                            except Exception:
                                pass
                if state.events.delete(int(e["id"])):
                    deleted += 1
        # Soft storage cap (best-effort)
        try:
            cap = int(state.settings.max_storage_gb or 10) * (1024**3)
            total_bytes = 0
            if state.frames_dir.exists():
                for p in state.frames_dir.glob("*.jpg"):
                    try:
                        total_bytes += p.stat().st_size
                    except Exception:
                        pass
            while total_bytes > cap:
                batch = state.events.oldest(100)
                if not batch:
                    break
                for e in batch:
                    meta = e.get("meta") or {}
                    for k in ("image_full", "image_crop", "image_thumb"):
                        name = meta.get(k)
                        if name:
                            p = state.frames_dir / str(name)
                            if p.exists():
                                try:
                                    sz = p.stat().st_size
                                    p.unlink()
                                    total_bytes -= sz
                                except Exception:
                                    pass
                    if state.events.delete(int(e["id"])):
                        deleted += 1
                if len(batch) < 100:
                    break
        except Exception:
            pass
        return deleted

    @app.get("/api/event_image")
    async def event_image(id: int, kind: str = "thumb") -> Response:
        ev = state.events.get(int(id))
        if not ev:
            raise HTTPException(404, "Event not found")
        meta = ev.get("meta") or {}
        name = None
        if kind == "full":
            name = meta.get("image_full")
        elif kind == "crop":
            name = meta.get("image_crop")
        else:
            name = (
                meta.get("image_thumb")
                or meta.get("image_crop")
                or meta.get("image_full")
            )
        if not name:
            raise HTTPException(404, "No image for event")
        path = state.frames_dir / name
        if not path.exists():
            raise HTTPException(404, "Image not found")
        data = path.read_bytes()
        return Response(content=data, media_type="image/jpeg")

    # Events bulk clear (place before parameterized routes to avoid 422 conflicts)
    @app.post("/api/events/clear", response_class=JSONResponse)
    async def clear_events_compat():
        return await clear_events_impl()

    @app.post("/api/events/clear_all", response_class=JSONResponse)
    async def clear_events_api():
        return await clear_events_impl()

    async def clear_events_impl():
        # Remove all referenced images first, then clear DB
        try:
            ref = state.events.referenced_images()
            for name in list(ref):
                p = state.frames_dir / str(name)
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception:
            pass
        deleted = state.events.clear()
        # Also reset compare baseline so helper endpoints ignore older cache
        try:
            state.compare_baseline_ts = time.time()
        except Exception:
            pass
        return {"ok": True, "deleted": int(deleted)}

    # Spot-specific bulk clear (place before parameterized routes)
    @app.get("/api/events/clear_spot", response_class=JSONResponse)
    async def clear_spot_events_get(spot_id: str):
        sid = (spot_id or "").strip()
        if not sid:
            raise HTTPException(400, "spot_id required")
        return await _clear_spot_impl(sid)

    @app.post("/api/events/clear_spot", response_class=JSONResponse)
    async def clear_spot_events(payload: dict):
        spot_id = str(payload.get("spot_id") or "").strip()
        if not spot_id:
            raise HTTPException(400, "spot_id required")
        return await _clear_spot_impl(spot_id)

    async def _clear_spot_impl(spot_id: str):
        deleted = 0
        files_deleted = 0
        # Delete in batches to avoid loading everything at once
        while True:
            batch = state.events.events_with_spot(spot_id, limit=1000)
            if not batch:
                break
            for e in batch:
                # remove images
                try:
                    meta = e.get("meta") or {}
                    for k in ("image_full", "image_crop", "image_thumb"):
                        name = meta.get(k)
                        if name:
                            p = state.frames_dir / str(name)
                            if p.exists():
                                try:
                                    p.unlink()
                                    files_deleted += 1
                                except Exception:
                                    pass
                except Exception:
                    pass
                if state.events.delete(int(e["id"])):
                    deleted += 1
            # If we got fewer than limit, we're done
            if len(batch) < 1000:
                break
        # Best-effort orphan cleanup not strictly necessary, but cheap for consistency
        try:
            ref = state.events.referenced_images()
            if state.frames_dir.exists():
                for p in state.frames_dir.glob("*.jpg"):
                    if p.name not in ref:
                        try:
                            p.unlink()
                        except Exception:
                            pass
        except Exception:
            pass
        return {
            "ok": True,
            "deleted": int(deleted),
            "files_deleted": int(files_deleted),
        }

    # Generic events API for read/edit/delete
    @app.get("/api/events", response_class=JSONResponse)
    async def list_events(limit: int = 100):
        return {"items": state.events.recent(int(limit))}

    @app.get("/api/events/{event_id}", response_class=JSONResponse)
    async def get_event(event_id: int):
        ev = state.events.get(int(event_id))
        if not ev:
            raise HTTPException(404, "Event not found")
        return ev

    @app.post("/api/events/{event_id}", response_class=JSONResponse)
    async def update_event(event_id: int, payload: dict):
        kind = payload.get("kind")
        meta = payload.get("meta")
        ts = payload.get("ts")
        if meta is not None and not isinstance(meta, dict):
            raise HTTPException(400, "meta must be an object")
        ok = state.events.update(
            int(event_id),
            kind=kind,
            meta=meta,
            ts=float(ts) if ts is not None else None,
        )
        if not ok:
            raise HTTPException(404, "Event not found or nothing to update")
        ev = state.events.get(int(event_id))
        return {"ok": True, "event": ev}

    @app.delete("/api/events/{event_id}", response_class=JSONResponse)
    async def delete_event(event_id: int):
        # Try to delete associated image files first
        try:
            ev = state.events.get(int(event_id))
            if ev:
                meta = ev.get("meta") or {}
                for k in ("image_full", "image_crop", "image_thumb"):
                    name = meta.get(k)
                    if name:
                        p = state.frames_dir / str(name)
                        if p.exists():
                            try:
                                p.unlink()
                            except Exception:
                                pass
        except Exception:
            pass
        ok = state.events.delete(int(event_id))
        if not ok:
            raise HTTPException(404, "Event not found")
        return {"ok": True}

    # Duplicate clear_all route removed; handled by clear_events_api above.

    @app.get("/api/event_explain", response_class=JSONResponse)
    async def event_explain(id: int):
        e = state.events.get(int(id))
        if not e:
            raise HTTPException(404, "Event not found")
        meta = e.get("meta") or {}
        spot_id = (meta.get("spot_id") or "").strip()
        delta_bits = meta.get("delta_bits")
        try:
            delta_bits = int(delta_bits) if delta_bits is not None else None
        except Exception:
            delta_bits = None
        # Determine thresholds using category defaults, then apply per-spot overrides
        min_bits = int(state.settings.phash_min_bits or 0)
        stable_ms = int(state.settings.phash_stable_ms or 0)
        if spot_id:
            z = load_zones(state.zones_path)
            sp = next((s for s in z.spots if s.id == spot_id), None)
            if sp:
                try:
                    cat = (getattr(sp, "category", None) or "").strip().lower()
                    if cat == "gate":
                        stable_ms = int(
                            state.settings.category_gate_stable_ms or stable_ms
                        )
                        min_bits = int(
                            state.settings.category_gate_phash_min_bits or min_bits
                        )
                    elif cat in ("spot", "parking"):
                        stable_ms = int(
                            state.settings.category_parking_stable_ms or stable_ms
                        )
                        min_bits = int(
                            state.settings.category_parking_phash_min_bits or min_bits
                        )
                except Exception:
                    pass
                try:
                    if getattr(sp, "min_bits", None) is not None:
                        min_bits = int(sp.min_bits)
                except Exception:
                    pass
                try:
                    if getattr(sp, "stable_ms", None) is not None:
                        stable_ms = int(sp.stable_ms)
                except Exception:
                    pass
        meets = delta_bits is not None and delta_bits >= min_bits
        explanation = {
            "spot_id": spot_id,
            "delta_bits": delta_bits,
            "min_bits_used": min_bits,
            "stable_ms_config": stable_ms,
            "prev_sig": meta.get("prev_sig"),
            "new_sig": meta.get("new_sig"),
            "duration_prev": meta.get("duration_prev"),
            "meets_threshold": meets,
            "note": "Stable window enforced at trigger time; exact elapsed not stored.",
        }
        return {"ok": True, "explain": explanation}

    @app.get("/api/spot_recent", response_class=JSONResponse)
    async def spot_recent(per_spot: int = 2, scan_limit: int = 200):
        try:
            per_spot = max(1, int(per_spot))
            scan_limit = max(1, int(scan_limit))
        except Exception:
            per_spot, scan_limit = 2, 200
        # Collect recent spot_change events and group by spot
        baseline = float(state.compare_baseline_ts or 0.0)
        evs = [
            e
            for e in state.events.recent(scan_limit)
            if str(e.get("kind", "")).lower() == "spot_change"
            and (baseline <= 0.0 or float(e.get("ts", 0.0)) >= baseline)
        ]
        by_spot: dict[str, list[dict]] = {}
        for e in evs:
            sid = (e.get("meta") or {}).get("spot_id") or ""
            if not sid:
                continue
            by_spot.setdefault(sid, []).append(e)
            if len(by_spot[sid]) >= per_spot:
                continue
        out = []
        z = load_zones(state.zones_path)
        for sid, lst in by_spot.items():
            # lst is already newest-first from recent(); pick previous if available, else latest
            prev = lst[1] if len(lst) >= 2 else lst[0]
            meta = prev.get("meta") or {}
            # Build URLs for display convenience
            prev_id = prev["id"]
            # Prefer thumb then crop then full (smaller consistent width)
            prev_url = None
            if meta.get("image_thumb"):
                prev_url = f"/api/event_image?id={prev_id}&kind=thumb"
            elif meta.get("image_crop"):
                prev_url = f"/api/event_image?id={prev_id}&kind=crop"
            elif meta.get("image_full"):
                prev_url = f"/api/event_image?id={prev_id}&kind=full"
            prev_sig = meta.get("new_sig") or meta.get("prev_sig")
            # Filter using LLM significant flag when available; fallback to delta_bits threshold
            significant = bool(meta.get("significant"))
            if not significant:
                try:
                    dbits = int(meta.get("delta_bits") or 0)
                    # Use category-specific min_bits when possible
                    mb = int(state.settings.phash_min_bits or 0)
                    try:
                        sp = next((s for s in z.spots if s.id == sid), None)
                        if sp:
                            cat = (getattr(sp, "category", None) or "").strip().lower()
                            if cat == "gate":
                                mb = int(
                                    state.settings.category_gate_phash_min_bits or mb
                                )
                            elif cat in ("spot", "parking"):
                                mb = int(
                                    state.settings.category_parking_phash_min_bits or mb
                                )
                            if getattr(sp, "min_bits", None) is not None:
                                mb = int(sp.min_bits)
                    except Exception:
                        pass
                    significant = dbits >= (mb + 2)
                except Exception:
                    significant = False
            if not significant:
                continue
            out.append(
                {
                    "spot_id": sid,
                    "prev_event_id": prev_id,
                    "prev_url": prev_url,
                    "prev_sig": prev_sig,
                    "prev_ts": prev.get("ts"),
                }
            )
        return {"items": out}

    # Thumbnail and image helpers

    @app.get("/api/thumbnails", response_class=JSONResponse)
    async def thumbnails(limit: int = 60, significant_only: bool = False):
        items = []
        for e in state.events.recent(max(0, int(limit))):
            meta = e.get("meta") or {}
            thumb = (
                meta.get("image_thumb")
                or meta.get("image_crop")
                or meta.get("image_full")
            )
            if not thumb:
                continue
            if significant_only:
                try:
                    if not bool(meta.get("significant")):
                        continue
                except Exception:
                    continue
            items.append(
                {
                    "event_id": e["id"],
                    "spot_id": meta.get("spot_id", ""),
                    "ts": e["ts"],
                    "thumb": f"/api/event_image?id={e['id']}&kind=thumb",
                    "full": f"/api/event_image?id={e['id']}&kind=full",
                    "significant": bool(meta.get("significant") or False),
                }
            )
            if len(items) >= limit:
                break
        return {"items": items}

    @app.get("/api/images/summary", response_class=JSONResponse)
    async def images_summary():

        total_files = 0
        total_bytes = 0
        if state.frames_dir.exists():
            for p in state.frames_dir.glob("*.jpg"):
                try:
                    total_files += 1
                    total_bytes += p.stat().st_size
                except Exception:
                    pass
        ref = state.events.referenced_images()
        orphan = 0
        if state.frames_dir.exists():
            for p in state.frames_dir.glob("*.jpg"):
                if p.name not in ref:
                    orphan += 1
        return {
            "files": total_files,
            "bytes": total_bytes,
            "referenced": len(ref),
            "orphans": orphan,
        }

    @app.post("/api/images/cleanup", response_class=JSONResponse)
    async def images_cleanup():
        ref = state.events.referenced_images()
        deleted = 0
        if state.frames_dir.exists():
            for p in state.frames_dir.glob("*.jpg"):
                if p.name not in ref:
                    try:
                        p.unlink()
                        deleted += 1
                    except Exception:
                        pass
        return {"ok": True, "deleted": deleted}

    def _do_control(action: str):
        if action == "start":
            if not settings.rtsp_url:
                raise HTTPException(400, "RTSP_URL not configured")
            state.ensure_capture()
            return {"ok": True, "running": True}
        elif action == "stop":
            if state.capture:
                state.capture.stop()
                # Reset worker so a fresh start() creates a new thread cleanly
                try:
                    state.capture = None
                except Exception:
                    pass
            return {"ok": True, "running": False}
        else:
            raise HTTPException(400, "Unknown action")

    @app.post("/api/control", response_class=JSONResponse)
    async def api_control_post(action: str):
        return _do_control(action)

    @app.get("/api/control", response_class=JSONResponse)
    async def api_control_get(action: str):
        return _do_control(action)

    @app.get("/frame.jpg")
    async def frame() -> Response:
        if not state.capture:
            raise HTTPException(404, "No capture running")
        latest = state.capture.latest()
        if not latest:
            # Briefly wait for a fresh frame to avoid transient grey
            try:
                import asyncio

                for _ in range(3):
                    await asyncio.sleep(0.08)
                    latest = state.capture.latest()
                    if latest:
                        break
            except Exception:
                pass
            if not latest:
                # Serve last encoded JPEG if available to avoid flicker/grey
                if state._last_jpg is not None:
                    return Response(
                        content=state._last_jpg,
                        media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"},
                    )
                raise HTTPException(404, "No frame available")
        frame, _ts = latest
        # Encode JPEG at a limited rate and cache it to reduce CPU
        try:
            min_period = (
                1.0 / max(0.1, float(state.settings.frame_jpeg_fps or 0))
                if state.settings.frame_jpeg_fps
                else 0.0
            )
        except Exception:
            min_period = 0.0
        now = time.time()
        use_cache = False
        if min_period > 0.0 and state._last_jpg is not None:
            if (now - state._last_jpg_ts) < min_period:
                use_cache = True
        if use_cache and state._last_jpg is not None:
            return Response(
                content=state._last_jpg,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )
        # No grey-frame heuristic here; rely on capture-level resync and warmup.
        try:
            jpg = state.capture.encode_jpeg(
                frame, quality=int(state.settings.jpeg_quality or 80)
            )
        except Exception:
            # Fallback to last cached JPG on encoding error
            if state._last_jpg is not None:
                return Response(
                    content=state._last_jpg,
                    media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"},
                )
            raise
        state._last_jpg = jpg
        state._last_jpg_ts = now
        return Response(
            content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
        )

    # Removed grid cell preview endpoint for a simpler calibration UX
    # gate endpoint removed

    @app.get("/api/spot.jpg")
    async def spot_crop(id: str, w: int = 0) -> Response:
        if not state.capture:
            raise HTTPException(404, "No capture running")
        latest = state.capture.latest()
        if not latest:
            raise HTTPException(404, "No frame available")
        frame, _ts = latest
        zones = load_zones(state.zones_path)
        spot = next((s for s in zones.spots if s.id == id), None)
        if not spot:
            raise HTTPException(404, "Spot not found")
        # Compute bounding box of the polygon
        xs = [p[0] for p in spot.polygon]
        ys = [p[1] for p in spot.polygon]
        x1, y1 = max(0, int(round(min(xs)))), max(0, int(round(min(ys))))
        x2, y2 = int(round(max(xs))), int(round(max(ys)))
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            raise HTTPException(400, "Invalid spot polygon bounds")
        crop = frame[y1:y2, x1:x2].copy()
        # Optional scaling to a requested width (maintain aspect)
        if isinstance(w, int) and w and w > 0:
            import cv2

            h_c, w_c = crop.shape[:2]
            if w_c > 0 and h_c > 0 and w_c != w:
                new_w = int(w)
                new_h = max(1, int(h_c * (new_w / w_c)))
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        jpg = state.capture.encode_jpeg(
            crop, quality=int(state.settings.jpeg_quality or 80)
        )
        return Response(content=jpg, media_type="image/jpeg")

    # removed spot_suggest endpoint — manual rotation preferred

    @app.get("/api/zones", response_class=JSONResponse)
    async def get_zones():
        zones = load_zones(state.zones_path)
        return zones.to_dict()

    @app.post("/api/zones", response_class=JSONResponse)
    async def set_zones(payload: dict):
        # payload: { gate: {polygon:[[x,y],...]} | legacy formats | null, spots: [{id,name,polygon:[[x,y],...]}] }
        save_zones(state.zones_path, load_zones_from_payload(payload))
        return {"ok": True}

    # Analysis tasks/result endpoints
    @app.get("/api/analysis/tasks", response_class=JSONResponse)
    async def analysis_tasks(limit: int = 5):
        import json

        tasks_dir = state.data_dir / "analysis" / "tasks"
        items = []
        if tasks_dir.exists():
            files = sorted(
                tasks_dir.glob("*.json"), key=lambda p: p.name, reverse=True
            )[:limit]
            for p in files:
                try:
                    with p.open("r", encoding="utf-8") as f:
                        t = json.load(f)
                    t["image_url"] = f"/data/{t.get('image','')}"
                    items.append(t)
                except Exception:
                    continue
        return {"tasks": items}

    @app.post("/api/analysis/result", response_class=JSONResponse)
    async def analysis_result(payload: dict):
        import json

        tid = str(payload.get("id", ""))
        if not tid:
            raise HTTPException(400, "Missing id")
        out_dir = state.data_dir / "analysis" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"{tid}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        try:
            state.events.add("analysis", {"task": tid})
        except Exception:
            pass
        return {"ok": True}

    @app.get("/api/analysis/latest", response_class=JSONResponse)
    async def analysis_latest():
        import json

        res_dir = state.data_dir / "analysis" / "results"
        if not res_dir.exists():
            return {"result": None}
        files = sorted(res_dir.glob("*.json"), key=lambda p: p.name, reverse=True)
        if not files:
            return {"result": None}
        with files[0].open("r", encoding="utf-8") as f:
            return {"result": json.load(f)}

    # Simple config helpers to get/set RTSP_URL in .env
    @app.get("/api/config", response_class=JSONResponse)
    async def get_config():
        return {
            "RTSP_URL": settings.rtsp_url or "",
            "DATA_DIR": str(settings.data_dir),
            "DETECTOR_METHOD": settings.detector_method,
            "PHASH_MIN_BITS": int(settings.phash_min_bits or 0),
            "PHASH_STABLE_MS": int(settings.phash_stable_ms or 0),
            "ROI_DIFF_ALPHA": float(settings.roi_diff_alpha or 0.05),
            "ROI_DIFF_THRESHOLD": float(settings.roi_diff_threshold or 0.02),
            "ROI_DIFF_MIN_PIXELS": int(
                getattr(settings, "roi_diff_min_pixels", 0) or 0
            ),
            "ROI_DIFF_COOLDOWN_MS": int(
                getattr(settings, "roi_diff_cooldown_ms", 0) or 0
            ),
            # category defaults
            "CATEGORY_GATE_STABLE_MS": int(settings.category_gate_stable_ms or 0),
            "CATEGORY_GATE_PHASH_MIN_BITS": int(
                settings.category_gate_phash_min_bits or 0
            ),
            "CATEGORY_GATE_ROI_DIFF_THRESHOLD": float(
                settings.category_gate_roi_diff_threshold or 0.0
            ),
            "CATEGORY_PARKING_STABLE_MS": int(settings.category_parking_stable_ms or 0),
            "CATEGORY_PARKING_PHASH_MIN_BITS": int(
                settings.category_parking_phash_min_bits or 0
            ),
            "CATEGORY_PARKING_ROI_DIFF_THRESHOLD": float(
                settings.category_parking_roi_diff_threshold or 0.0
            ),
        }

    @app.post("/api/config", response_class=JSONResponse)
    async def set_config(payload: dict):
        # Optional values; only set if provided
        def _opt_set_int(key, attr):
            if key in payload and payload.get(key) is not None:
                try:
                    setattr(settings, attr, int(payload.get(key)))
                except Exception:
                    raise HTTPException(400, f"{key} must be an integer")

        def _opt_set_float(key, attr):
            if key in payload and payload.get(key) is not None:
                try:
                    setattr(settings, attr, float(payload.get(key)))
                except Exception:
                    raise HTTPException(400, f"{key} must be a number")

        if payload.get("RTSP_URL") is not None:
            rtsp_str = str(payload.get("RTSP_URL")).strip()
            if not rtsp_str:
                raise HTTPException(400, "RTSP_URL required")
            settings.rtsp_url = rtsp_str
        if payload.get("DETECTOR_METHOD") is not None:
            dm = str(payload.get("DETECTOR_METHOD")).strip().lower()
            if dm not in {"phash", "roi_diff"}:
                raise HTTPException(
                    400, "DETECTOR_METHOD must be 'phash' or 'roi_diff'"
                )
            settings.detector_method = dm
        _opt_set_int("PHASH_MIN_BITS", "phash_min_bits")
        _opt_set_int("PHASH_STABLE_MS", "phash_stable_ms")
        _opt_set_float("ROI_DIFF_ALPHA", "roi_diff_alpha")
        _opt_set_float("ROI_DIFF_THRESHOLD", "roi_diff_threshold")
        _opt_set_int("ROI_DIFF_MIN_PIXELS", "roi_diff_min_pixels")
        _opt_set_int("ROI_DIFF_COOLDOWN_MS", "roi_diff_cooldown_ms")
        # Category defaults
        _opt_set_int("CATEGORY_GATE_STABLE_MS", "category_gate_stable_ms")
        _opt_set_int("CATEGORY_GATE_PHASH_MIN_BITS", "category_gate_phash_min_bits")
        _opt_set_float(
            "CATEGORY_GATE_ROI_DIFF_THRESHOLD", "category_gate_roi_diff_threshold"
        )
        _opt_set_int("CATEGORY_PARKING_STABLE_MS", "category_parking_stable_ms")
        _opt_set_int(
            "CATEGORY_PARKING_PHASH_MIN_BITS", "category_parking_phash_min_bits"
        )
        _opt_set_float(
            "CATEGORY_PARKING_ROI_DIFF_THRESHOLD", "category_parking_roi_diff_threshold"
        )
        # Persist to project .env (best-effort)
        warn_msg = None
        try:
            project_root = Path(__file__).resolve().parents[1]
            env_path = project_root / ".env"
            # read existing (if any)
            lines = []
            if env_path.exists():
                lines = env_path.read_text(encoding="utf-8").splitlines()
            kv = {
                "RTSP_URL": settings.rtsp_url or "",
                "DETECTOR_METHOD": settings.detector_method,
                "PHASH_MIN_BITS": str(int(settings.phash_min_bits or 0)),
                "PHASH_STABLE_MS": str(int(settings.phash_stable_ms or 0)),
                "ROI_DIFF_ALPHA": str(float(settings.roi_diff_alpha or 0.05)),
                "ROI_DIFF_THRESHOLD": str(float(settings.roi_diff_threshold or 0.02)),
                "ROI_DIFF_MIN_PIXELS": str(
                    int(getattr(settings, "roi_diff_min_pixels", 0) or 0)
                ),
                "ROI_DIFF_COOLDOWN_MS": str(
                    int(getattr(settings, "roi_diff_cooldown_ms", 0) or 0)
                ),
                "CATEGORY_GATE_STABLE_MS": str(
                    int(settings.category_gate_stable_ms or 0)
                ),
                "CATEGORY_GATE_PHASH_MIN_BITS": str(
                    int(settings.category_gate_phash_min_bits or 0)
                ),
                "CATEGORY_GATE_ROI_DIFF_THRESHOLD": str(
                    float(settings.category_gate_roi_diff_threshold or 0.0)
                ),
                "CATEGORY_PARKING_STABLE_MS": str(
                    int(settings.category_parking_stable_ms or 0)
                ),
                "CATEGORY_PARKING_PHASH_MIN_BITS": str(
                    int(settings.category_parking_phash_min_bits or 0)
                ),
                "CATEGORY_PARKING_ROI_DIFF_THRESHOLD": str(
                    float(settings.category_parking_roi_diff_threshold or 0.0)
                ),
            }
            keys = set(kv.keys())
            out_lines = []
            seen = set()
            for ln in lines:
                k = ln.split("=", 1)[0].strip()
                if k in keys:
                    out_lines.append(f"{k}={kv[k]}")
                    seen.add(k)
                else:
                    out_lines.append(ln)
            for k in keys - seen:
                out_lines.append(f"{k}={kv[k]}")
            env_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        except Exception as e:
            warn_msg = f"Failed to write .env: {e}"
        cfg = await get_config()
        if warn_msg:
            try:
                cfg["warning"] = warn_msg
            except Exception:
                pass
        return cfg

    # Manual retention trigger
    @app.post("/api/retention/apply", response_class=JSONResponse)
    async def retention_apply():
        deleted = 0
        try:
            deleted = apply_retention_sweep()
        except Exception:
            pass
        return {"ok": True, "deleted": deleted}

    # LLM diagnostics and manual queueing
    @app.get("/api/llm/status", response_class=JSONResponse)
    async def llm_status():
        return {
            "enabled": bool(getattr(settings, "llm_enabled", False)),
            "provider": getattr(settings, "llm_provider", ""),
            "model_fast": getattr(settings, "llm_model_fast", ""),
            "running": bool(state.llm is not None),
        }

    @app.post("/api/llm/queue_last", response_class=JSONResponse)
    async def llm_queue_last(spot_id: str | None = None):
        # Find the most recent spot_change event (optionally for a specific spot)
        ev = None
        for e in state.events.recent(500):
            if str(e.get("kind", "")).lower() != "spot_change":
                continue
            if spot_id:
                meta = e.get("meta") or {}
                if str(meta.get("spot_id") or "") != spot_id:
                    continue
            ev = e
            break
        if not ev:
            raise HTTPException(404, "No spot_change event found")
        if not state.llm:
            raise HTTPException(503, "LLM worker not running")
        try:
            state.llm.queue(int(ev["id"]))
        except Exception:
            raise HTTPException(500, "Failed to enqueue")
        return {"ok": True, "queued_event_id": int(ev["id"])}

    @app.post("/api/llm/queue_event", response_class=JSONResponse)
    async def llm_queue_event(id: int):
        ev = state.events.get(int(id))
        if not ev or str(ev.get("kind", "")).lower() != "spot_change":
            raise HTTPException(404, "spot_change event not found")
        if not state.llm:
            raise HTTPException(503, "LLM worker not running")
        try:
            state.llm.queue(int(id))
        except Exception:
            raise HTTPException(500, "Failed to enqueue")
        return {"ok": True, "queued_event_id": int(id)}

    # Comparison baseline helpers
    @app.get("/api/compare/baseline", response_class=JSONResponse)
    async def compare_get_baseline():
        return {"baseline_ts": state.compare_baseline_ts}

    @app.post("/api/compare/baseline", response_class=JSONResponse)
    async def compare_set_baseline(payload: dict | None = None):
        import time as _t

        ts = None
        try:
            if payload and "ts" in payload:
                ts = float(payload.get("ts"))
        except Exception:
            ts = None
        if ts is None or ts <= 0:
            ts = _t.time()
        state.compare_baseline_ts = float(ts)
        return {"ok": True, "baseline_ts": state.compare_baseline_ts}

    @app.post("/api/compare/clear", response_class=JSONResponse)
    async def compare_clear_baseline():
        state.compare_baseline_ts = 0.0
        return {"ok": True, "baseline_ts": state.compare_baseline_ts}

    @app.get("/api/spot_history", response_class=JSONResponse)
    async def spot_history(
        spot_id: str, limit: int = 8, significant_only: bool = False
    ):
        evs = state.events.spot_events(spot_id, limit)
        if significant_only:
            try:
                evs = [e for e in evs if bool((e.get("meta") or {}).get("significant"))]
            except Exception:
                pass
        out = []
        for e in evs:
            out.append(
                {
                    "id": e["id"],
                    "ts": e["ts"],
                    "spot_id": spot_id,
                    "thumb": f"/api/event_image?id={e['id']}&kind=thumb",
                    "full": f"/api/event_image?id={e['id']}&kind=full",
                    "crop": f"/api/event_image?id={e['id']}&kind=crop",
                    "significant": bool(
                        ((e.get("meta") or {}).get("significant")) or False
                    ),
                }
            )
        return {"items": out}

    return app


def load_zones_from_payload(payload: dict):
    from .zones import Zones, Spot

    gate_p = payload.get("gate")
    gate_poly = None
    if gate_p:
        if isinstance(gate_p, dict) and gate_p.get("polygon"):
            gate_poly = [tuple(p) for p in gate_p["polygon"]]
        elif isinstance(gate_p, dict) and gate_p.get("center") is not None:
            cx, cy = gate_p["center"]
            w = float(gate_p.get("w", 0))
            h = float(gate_p.get("h", 0))
            x0, y0 = cx - w / 2.0, cy - h / 2.0
            gate_poly = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
        elif isinstance(gate_p, list):
            gate_poly = [tuple(p) for p in gate_p]
    spots_payload = payload.get("spots", [])
    spots = []
    for s in spots_payload:
        spots.append(
            Spot(
                id=s["id"],
                name=s.get("name", s.get("id", "")),
                polygon=[tuple(p) for p in s.get("polygon", [])],
                min_bits=s.get("min_bits"),
                stable_ms=s.get("stable_ms"),
                category=s.get("category"),
            )
        )
    return Zones(gate=gate_poly, spots=spots)


app = create_app()
