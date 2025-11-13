from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .capture import VideoCaptureWorker
from .storage import EventStore
from .analyzer import AnalysisWorker
from .zones import load_zones, save_zones


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
        # in-memory spot image signature tracker: { spot_id: {sig: str, since: float} }
        self.spot_tracker: dict[str, dict[str, float | str]] = {}

    def ensure_capture(self) -> None:
        if not self.settings.rtsp_url:
            return
        if self.capture is None:
            self.capture = VideoCaptureWorker(self.settings.rtsp_url)
        # start only if not running
        self.capture.start()


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
            worker.get_latest_frame = (lambda: state.capture.latest() if state.capture else None)
            def _get_spots():
                z = load_zones(state.zones_path)
                return [{"id": s.id, "name": s.name, "polygon": s.polygon} for s in z.spots]
            worker.get_spots = _get_spots
            worker.start()
            state.analysis = worker

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML

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

    @app.get("/api/spot_stats", response_class=JSONResponse)
    async def api_spot_stats():
        import numpy as np
        import cv2
        zones = load_zones(state.zones_path)
        now = time.time()
        latest = state.capture.latest() if state.capture else None
        frame = latest[0] if latest else None
        stats = []
        for s in zones.spots:
            # compute bbox and signature if we have a frame
            xs = [p[0] for p in s.polygon] or [0]
            ys = [p[1] for p in s.polygon] or [0]
            x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
            x2, y2 = int(max(xs)), int(max(ys))
            sig = None
            if frame is not None and x2 > x1 and y2 > y1:
                crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
                if crop.size > 0:
                    # downsample grayscale to 8x8 and threshold to get a simple hash
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
                    mean = float(np.mean(small))
                    bits = (small > mean).astype(np.uint8)
                    sig = ''.join('1' if b else '0' for b in bits.flatten())
            rec = state.spot_tracker.get(s.id)
            if sig is not None:
                if not rec:
                    state.spot_tracker[s.id] = { 'sig': sig, 'since': now }
                    rec = state.spot_tracker[s.id]
                elif rec.get('sig') != sig:
                    # Emit a spot_change event for previous state duration and save images
                    prev_sig = rec.get('sig')
                    prev_since = float(rec.get('since') or now)
                    duration_prev = max(0.0, now - prev_since)
                    try:
                        import cv2, os
                        ts_ms = int(now * 1000)
                        base = f"evt_{ts_ms}_{s.id}"
                        full_path = state.frames_dir / f"{base}_full.jpg"
                        crop_path = state.frames_dir / f"{base}_crop.jpg"
                        thumb_path = state.frames_dir / f"{base}_thumb.jpg"
                        # Save full frame
                        try:
                            cv2.imwrite(str(full_path), frame)
                        except Exception:
                            pass
                        # Save spot crop and a small thumbnail
                        if 'crop' in locals() and crop is not None and crop.size > 0:
                            try:
                                cv2.imwrite(str(crop_path), crop)
                                # thumbnail 240px width preserving aspect
                                h_c, w_c = crop.shape[:2]
                                if w_c > 0 and h_c > 0:
                                    new_w = 240
                                    new_h = max(1, int(h_c * (new_w / w_c)))
                                    thumb = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                                    cv2.imwrite(str(thumb_path), thumb)
                            except Exception:
                                pass
                        meta = {
                            'spot_id': s.id,
                            'prev_sig': prev_sig,
                            'new_sig': sig,
                            'duration_prev': duration_prev,
                            'image_full': full_path.name,
                            'image_crop': crop_path.name,
                            'image_thumb': thumb_path.name,
                        }
                        state.events.add('spot_change', meta)
                    except Exception:
                        pass
                    state.spot_tracker[s.id] = { 'sig': sig, 'since': now }
                    rec = state.spot_tracker[s.id]
            since = rec.get('since') if rec else None
            duration = (now - float(since)) if since else None
            stats.append({
                'id': s.id,
                'name': s.name or s.id,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'w': max(0, x2 - x1), 'h': max(0, y2 - y1),
                'since': since,
                'duration_sec': duration,
            })
        return { 'spots': stats, 'ts': now }

    @app.get("/api/event_image")
    async def event_image(id: int, kind: str = "thumb") -> Response:
        ev = state.events.get(int(id))
        if not ev:
            raise HTTPException(404, "Event not found")
        meta = ev.get('meta') or {}
        name = None
        if kind == 'full':
            name = meta.get('image_full')
        elif kind == 'crop':
            name = meta.get('image_crop')
        else:
            name = meta.get('image_thumb') or meta.get('image_crop') or meta.get('image_full')
        if not name:
            raise HTTPException(404, "No image for event")
        path = state.frames_dir / name
        if not path.exists():
            raise HTTPException(404, "Image not found")
        data = path.read_bytes()
        return Response(content=data, media_type="image/jpeg")

    @app.post("/api/control", response_class=JSONResponse)
    async def api_control(action: str):
        if action == "start":
            if not settings.rtsp_url:
                raise HTTPException(400, "RTSP_URL not configured")
            state.ensure_capture()
            return {"ok": True, "running": True}
        elif action == "stop":
            if state.capture:
                state.capture.stop()
            return {"ok": True, "running": False}
        else:
            raise HTTPException(400, "Unknown action")

    @app.get("/frame.jpg")
    async def frame() -> Response:
        if not state.capture:
            raise HTTPException(404, "No capture running")
        latest = state.capture.latest()
        if not latest:
            raise HTTPException(404, "No frame available")
        frame, _ts = latest
        jpg = state.capture.encode_jpeg(frame)
        return Response(content=jpg, media_type="image/jpeg")

    # Removed grid cell preview endpoint for a simpler calibration UX
    # gate endpoint removed

    @app.get("/api/spot.jpg")
    async def spot_crop(id: str) -> Response:
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
        jpg = state.capture.encode_jpeg(crop)
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
            x0, y0 = cx - w/2.0, cy - h/2.0
            gate_poly = [(x0, y0), (x0+w, y0), (x0+w, y0+h), (x0, y0+h)]
        elif isinstance(gate_p, list):
            gate_poly = [tuple(p) for p in gate_p]
    spots_payload = payload.get("spots", [])
    spots = [Spot(id=s["id"], name=s.get("name", s.get("id", "")), polygon=[tuple(p) for p in s.get("polygon", [])]) for s in spots_payload]
    return Zones(gate=gate_poly, spots=spots)

    # Analysis tasks/result endpoints
    @app.get("/api/analysis/tasks", response_class=JSONResponse)
    async def analysis_tasks(limit: int = 5):
        import json
        tasks_dir = state.data_dir / "analysis" / "tasks"
        items = []
        if tasks_dir.exists():
            files = sorted(tasks_dir.glob("*.json"), key=lambda p: p.name, reverse=True)[:limit]
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


INDEX_HTML = """
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Backyard Monitor</title>
    <style>
      body { font-family: -apple-system, system-ui, sans-serif; margin: 20px; }
      .row { display:flex; gap:16px; align-items:flex-start; }
      .card { border:1px solid #ddd; padding:12px; border-radius:8px; }
      #canvasWrap { position: relative; display:inline-block; }
      canvas { border-radius:8px; border:1px solid #ddd; display:block; }
      table { border-collapse: collapse; }
      td, th { border: 1px solid #ddd; padding: 4px 8px; }
      .controls button { margin-right: 8px; }
      .muted { color:#666; }
      label { display:block; margin-top:6px; }
      input[type=number] { width: 6em; }
      .tooltip { position:absolute; background:#fff; border:1px solid #ccc; border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.12); padding:6px 8px; display:none; z-index:10; }
      .tooltip button { margin-right:6px; }
    </style>
  </head>
  <body>
    <h1>Backyard Monitor</h1>
    <div class="controls">
      <button onclick="control('start')">Start</button>
      <button onclick="control('stop')">Stop</button>
      <span class="muted">Spots: click to add; click to select; arrows to nudge; Delete to remove.</span>
    </div>
    <div class="row" style="margin-top:12px;">
      <div class="card">
        <div><strong>Live Frame</strong></div>
        <div id="canvasWrap">
          <canvas id="canvas" width="640" height="360"></canvas>
          <div id="spotToolbar" class="tooltip"></div>
        </div>
        <div class="muted" id="status"></div>
        <div style="margin-top:8px; display:flex; gap:8px; align-items:center; flex-wrap: wrap;">
          <button onclick="saveZones()">Save Zones</button>
          <button onclick="clearSpots()">Clear Spots</button>
          <button id="sizeBtn" onclick="toggleSize()">Shrink</button>
        </div>
        
      </div>
      <div class="card">
        <div style="margin-top:12px;"><strong>Spot Preview</strong>
          <label>Spot <select id="spot_select"></select></label>
          <button onclick="previewSpot()">Preview</button>
          <div><img id="spotimg" class="thumb" style="margin-top:6px;"/></div>
        </div>
        <div style="margin-top:12px;"><strong>Spot Events (last 10)</strong></div>
        <table id="events"></table>
      </div>
      <div class="card">
        <div><strong>Spots</strong></div>
        <table id="spot_stats"><tr><th>Spot</th><th>Center</th><th>Size</th><th>Present For</th><th>Preview</th></tr></table>
      </div>
    </div>
    <script>
      let spots = []; // [{id,name,polygon:[[x,y],...]}]
      let selectedSpotId = null;
      const DEFAULT_SPOT = { w: 120, h: 200 };
      const canvas = document.getElementById('canvas');
      const ctx = canvas.getContext('2d');
      const canvasWrap = document.getElementById('canvasWrap');
      const spotToolbar = document.getElementById('spotToolbar');
      let baseW = 0, baseH = 0; // native frame size

      async function control(action) {
        const res = await fetch('/api/control?action=' + action, { method: 'POST' });
        const j = await res.json();
        console.log(j);
        await refresh();
      }
      async function refresh() {
        const status = await (await fetch('/api/status')).json();
        document.getElementById('status').innerText = `running=${status.running} last_ts=${status.last_ts} size=${status.width}x${status.height}`;
        if (status.width && status.height) {
          baseW = status.width; baseH = status.height;
          // keep current scale or default to 1.0 first time
          const curScale = (canvas.width && baseW) ? (canvas.width / baseW) : 1.0;
          const s = (curScale > 0 ? curScale : 1.0);
          canvas.width = Math.round(baseW * s);
          canvas.height = Math.round(baseH * s);
          const btn = document.getElementById('sizeBtn');
          if (btn) btn.textContent = (s === 1.0) ? 'Shrink' : 'Expand';
        }
        const events = (status.events || []).filter(e => String(e.kind||'').toLowerCase() === 'spot_change').slice(0,10);
        const et = document.getElementById('events');
        et.innerHTML = events.map(e => {
          const ts = new Date(e.ts*1000).toLocaleString();
          const spot = (e.meta && e.meta.spot_id) ? e.meta.spot_id : '';
          return `<div style="display:inline-block; margin:6px; text-align:center;">`
                 + `<a href="/api/event_image?id=${e.id}&kind=full" target="_blank">`
                 + `<img class="thumb" src="/api/event_image?id=${e.id}&kind=thumb"/>`
                 + `</a>`
                 + `<div class="muted" style="max-width:240px;">${ts} • ${spot}</div>`
                 + `</div>`;
        }).join('');
      }

      function rectPolygon(cx, cy, w, h) {
        const x0 = cx - w/2, y0 = cy - h/2;
        return [ [x0,y0], [x0+w,y0], [x0+w,y0+h], [x0,y0+h] ];
      }
      function pointInPoly(pt, poly) {
        let [x, y] = pt; let inside = false;
        for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
          const xi = poly[i][0], yi = poly[i][1];
          const xj = poly[j][0], yj = poly[j][1];
          const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-9) + xi);
          if (intersect) inside = !inside;
        }
        return inside;
      }

      function deleteSelected(){
        if (!selectedSpotId) return;
        spots = spots.filter(sp => sp.id !== selectedSpotId);
        selectedSpotId = null;
        updateToolbar();
        draw();
      }

      function rotateSelected(deg){
        if (!selectedSpotId) return;
        const s = spots.find(sp => sp.id === selectedSpotId);
        if (!s || !s.polygon || s.polygon.length < 3) return;
        const xs = s.polygon.map(p=>p[0]); const ys = s.polygon.map(p=>p[1]);
        const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
        const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
        const ang = (deg * Math.PI) / 180;
        const ca = Math.cos(ang), sa = Math.sin(ang);
        s.polygon = s.polygon.map(([px,py]) => {
          const dx = px - cx, dy = py - cy;
          const rx = dx*ca - dy*sa; const ry = dx*sa + dy*ca;
          return [cx + rx, cy + ry];
        });
        updateToolbar();
        draw();
      }

      function updateSpotName(name){
        if (!selectedSpotId) return;
        const s = spots.find(sp => sp.id === selectedSpotId);
        if (s) { s.name = (name || '').trim() || s.id; updateToolbar(); draw(); }
      }

      function selectedBBox(){
        const s = spots.find(sp => sp.id === selectedSpotId);
        if (!s || !s.polygon || s.polygon.length < 3) return null;
        const xs = s.polygon.map(p=>p[0]); const ys = s.polygon.map(p=>p[1]);
        const x1 = Math.min(...xs), x2 = Math.max(...xs), y1 = Math.min(...ys), y2 = Math.max(...ys);
        return { x: x1, y: y1, w: x2-x1, h: y2-y1, name: s.name || s.id };
      }

      function updateToolbar(){
        const box = selectedBBox();
        if (!box){ spotToolbar.style.display = 'none'; return; }
        spotToolbar.style.display = 'block';
        const pad = 8;
        const sx = baseW ? (canvas.width / baseW) : 1;
        const sy = baseH ? (canvas.height / baseH) : 1;
        let tx = Math.max(pad, Math.min(canvas.width - 220, Math.round(box.x * sx)));
        let ty = Math.max(pad, Math.min(canvas.height - 44, Math.round(box.y * sy) - 36));
        spotToolbar.style.left = `${tx}px`;
        spotToolbar.style.top = `${ty}px`;
        const safe = (s) => String(s||'').replace(/[&<>]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]||c));
        spotToolbar.innerHTML = `
          <div style="display:flex; align-items:center; gap:6px;">
            <input id="tb_name" type="text" value="${safe(box.name)}" style="max-width:120px;"/>
            <button id="tb_save">Save</button>
            <button id="tb_rn">-10°</button>
            <button id="tb_rp">+10°</button>
            <button id="tb_del" style="color:#a00;">Delete</button>
          </div>`;
        document.getElementById('tb_save').onclick = () => {
          const v = document.getElementById('tb_name').value; updateSpotName(v);
        };
        document.getElementById('tb_rn').onclick = () => rotateSelected(-10);
        document.getElementById('tb_rp').onclick = () => rotateSelected(10);
        document.getElementById('tb_del').onclick = () => deleteSelected();
      }
      window.addEventListener('resize', updateToolbar);

      function toggleSize(){
        if (!baseW || !baseH) return;
        const curScale = canvas.width / baseW;
        const nextScale = curScale > 0.75 ? 0.5 : 1.0;
        canvas.width = Math.round(baseW * nextScale);
        canvas.height = Math.round(baseH * nextScale);
        const btn = document.getElementById('sizeBtn');
        if (btn) btn.textContent = (nextScale === 1.0) ? 'Shrink' : 'Expand';
        draw();
      }

      async function draw() {
        // draw latest frame
        const img = new Image();
        img.onload = () => {
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          // overlay spots as rectangles
          if (spots && spots.length) {
            ctx.strokeStyle = '#44aaff'; ctx.lineWidth = 2; ctx.fillStyle = 'rgba(68,170,255,0.12)';
            for (const s of spots) {
              if (!s.polygon || s.polygon.length < 4) continue;
              const sx = baseW ? (canvas.width / baseW) : 1;
              const sy = baseH ? (canvas.height / baseH) : 1;
              const p0 = s.polygon[0];
              ctx.beginPath(); ctx.moveTo(p0[0]*sx, p0[1]*sy);
              for (let i=1;i<s.polygon.length;i++){ const p = s.polygon[i]; ctx.lineTo(p[0]*sx, p[1]*sy); }
              ctx.closePath(); ctx.stroke(); ctx.fill();
              if (s.id === selectedSpotId) {
                ctx.strokeStyle = '#ffaa00'; ctx.lineWidth = 2; ctx.setLineDash([6,4]);
                ctx.stroke(); ctx.setLineDash([]); ctx.strokeStyle = '#44aaff';
              }
            }
          }
          updateToolbar();
        };
        img.src = '/frame.jpg?ts=' + Date.now();
      }

      canvas.addEventListener('click', async (e) => {
        const rect = canvas.getBoundingClientRect();
        const sx = baseW ? (baseW / canvas.width) : 1;
        const sy = baseH ? (baseH / canvas.height) : 1;
        const x = Math.round((e.clientX - rect.left) * sx);
        const y = Math.round((e.clientY - rect.top) * sy);
        const hit = spots.find(s => pointInPoly([x,y], s.polygon));
        if (hit) {
          selectedSpotId = hit.id;
          updateToolbar();
        } else {
          const id = 'spot_' + (spots.length + 1);
          let poly = rectPolygon(x, y, DEFAULT_SPOT.w, DEFAULT_SPOT.h);
          spots.push({ id, name: id, polygon: poly });
          selectedSpotId = id;
          updateToolbar();
        }
        renderZonesView();
        draw();
      });

      document.addEventListener('keydown', (e) => {
        if (!selectedSpotId) return;
        const s = spots.find(sp => sp.id === selectedSpotId);
        if (!s) return;
        const step = (e.shiftKey ? 10 : 5);
        if (e.key === 'Delete' || e.key === 'Backspace') {
          spots = spots.filter(sp => sp.id !== selectedSpotId);
          selectedSpotId = null; renderZonesView(); draw(); e.preventDefault(); return;
        }
        let dx = 0, dy = 0;
        if (e.key === 'ArrowLeft') dx = -step;
        else if (e.key === 'ArrowRight') dx = step;
        else if (e.key === 'ArrowUp') dy = -step;
        else if (e.key === 'ArrowDown') dy = step;
        if (dx || dy) {
          s.polygon = s.polygon.map(p => [p[0] + dx, p[1] + dy]);
          renderZonesView(); draw(); e.preventDefault();
        }
      });

      async function saveZones(){
        const payload = { spots };
        await fetch('/api/zones', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        alert('Saved');
      }

      async function loadZones(){
        const res = await fetch('/api/zones');
        const z = await res.json();
        spots = z.spots || [];
        renderZonesView();
      }

      function clearSpots(){ spots = []; selectedSpotId = null; draw(); }
      function renderZonesView(){ /* we keep homepage minimal; could add JSON preview if desired */ }

      setInterval(() => { draw(); }, 1000);
      async function updatePanels(){
        try {
          const [zones, status, spotStats] = await Promise.all([
            fetch('/api/zones').then(r=>r.json()),
            fetch('/api/status').then(r=>r.json()),
            fetch('/api/spot_stats').then(r=>r.json())
          ]);
          const sel = document.getElementById('spot_select');
          if (sel) {
            const cur = sel.value;
            sel.innerHTML = (zones.spots || []).map(s => `<option value="${s.id}">${s.name || s.id}</option>`).join('');
            if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
          }
          const tbl = document.getElementById('spot_stats');
          const spotsList = zones.spots || [];
          const statMap = Object.fromEntries((spotStats.spots||[]).map(x => [x.id, x]));
          tbl.innerHTML = '<tr><th>Spot</th><th>Center</th><th>Size</th><th>Present For</th><th>Preview</th></tr>' + spotsList.map(s => {
            const xs=s.polygon.map(p=>p[0]); const ys=s.polygon.map(p=>p[1]);
            const x1=Math.min(...xs), x2=Math.max(...xs), y1=Math.min(...ys), y2=Math.max(...ys);
            const center = isFinite(x1+x2) ? `${Math.round((x1+x2)/2)},${Math.round((y1+y2)/2)}` : '-';
            const size = isFinite(x2-x1) ? `${Math.round(x2-x1)}×${Math.round(y2-y1)}` : '-';
            const stat = statMap[s.id] || {}; const secs = stat.duration_sec || 0;
            const dur = secs ? fmtDuration(secs) : '-';
            const img = `<img class=\"thumb\" src=\"/api/spot.jpg?id=${encodeURIComponent(s.id)}&ts=${Date.now()}\"/>`;
            const safe = (t) => String(t||'').replace(/[&<>]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]||c));
            return `<tr>`+
              `<td><input id=\"name_${s.id}\" type=\"text\" value=\"${safe(s.name||s.id)}\" style=\"max-width:120px;\"/>`+
              ` <button onclick=\"renameSpot('${s.id}')\">Save</button></td>`+
              `<td>${center}</td><td>${size}</td><td>${dur}</td><td>${img}</td></tr>`;
          }).join('');
          draw();
        } catch {}
      }
      function fmtDuration(secs){
        secs = Math.floor(secs);
        const h = Math.floor(secs/3600); secs -= h*3600;
        const m = Math.floor(secs/60); secs -= m*60;
        const s = secs;
        const parts = [];
        if (h) parts.push(h+'h'); if (m) parts.push(m+'m'); parts.push(s+'s');
        return parts.join(' ');
      }
      setInterval(updatePanels, 1500);
      function previewSpot(){
        const sel = document.getElementById('spot_select');
        if (!sel || !sel.value) return;
        document.getElementById('spotimg').src = `/api/spot.jpg?id=${encodeURIComponent(sel.value)}&ts=` + Date.now();
      }
      async function renameSpot(id){
        try {
          const input = document.getElementById('name_'+id);
          const newName = input ? input.value : '';
          // pull current zones
          const z = await (await fetch('/api/zones')).json();
          const spotsZ = z.spots || [];
          for (const sp of spotsZ){ if (sp.id === id){ sp.name = newName || sp.id; break; } }
          await fetch('/api/zones', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ spots: spotsZ }) });
          // refresh panels after save
          updatePanels();
        } catch (e) { console.error(e); }
      }
      refresh();
      loadZones();
    </script>
  </body>
</html>
"""


app = create_app()
