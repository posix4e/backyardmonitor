from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass
import os
from pathlib import Path
from queue import Queue, Empty
from typing import Optional


@dataclass
class LLMConfig:
    provider: str
    model_fast: str
    timeout_sec: int


class LLMWorker:
    """Background worker that enriches spot_change events using an LLM.

    Flow:
      1) queue(event_id) adds an investigation_started event and enqueues id
      2) worker loads the spot_change event, reads crop/thumb image, calls LLM
      3) updates original event meta with significant + reason and emits deduced event
    """

    def __init__(self, frames_dir: Path, events, cfg: LLMConfig):
        self.frames_dir = frames_dir
        self.events = events
        self.cfg = cfg
        self._q: Queue[int] = Queue(maxsize=1024)
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def start(self):
        if self._t and self._t.is_alive():
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True, name="LLMWorker")
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            try:
                self._t.join(timeout=2)
            except Exception:
                pass

    def queue(self, event_id: int):
        try:
            # Create a lightweight investigation-started event
            try:
                self.events.add(
                    "spot_investigation",
                    {"event_id": int(event_id), "status": "started"},
                )
            except Exception:
                pass
            self._q.put_nowait(int(event_id))
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            try:
                try:
                    ev_id = self._q.get(timeout=0.5)
                except Empty:
                    continue
                self._handle(ev_id)
            except Exception:
                # Swallow and continue
                pass

    def _read_image_b64(self, meta: dict) -> tuple[Optional[str], Optional[str]]:
        for key in ("image_crop", "image_thumb", "image_full"):
            name = meta.get(key)
            if not name:
                continue
            p = self.frames_dir / str(name)
            if p.exists():
                try:
                    data = p.read_bytes()
                    b64 = base64.b64encode(data).decode("ascii")
                    return f"data:image/jpeg;base64,{b64}", key
                except Exception:
                    continue
        return None, None

    def _find_prev_event(self, ev: dict) -> Optional[dict]:
        """Return the most recent prior spot_change event for the same spot."""
        try:
            meta = ev.get("meta") or {}
            sid = meta.get("spot_id")
            if not sid:
                return None
            # Scan recent events for this spot and pick the first older than this event
            lst = self.events.events_with_spot(str(sid), limit=50)
            cur_id = int(ev.get("id") or 0)
            cur_ts = float(ev.get("ts") or 0.0)
            for e in lst:
                if int(e.get("id") or 0) == cur_id:
                    continue
                if str(e.get("kind", "")).lower() != "spot_change":
                    continue
                ets = float(e.get("ts") or 0.0)
                if cur_ts and ets and ets >= cur_ts:
                    continue
                return e
        except Exception:
            return None
        return None

    def _handle(self, event_id: int):
        import logging

        log = logging.getLogger("uvicorn.error")
        ev = self.events.get(int(event_id))
        if not ev or str(ev.get("kind", "")).lower() != "spot_change":
            return
        meta = ev.get("meta") or {}
        # Idempotence: skip if already done
        if str(meta.get("llm_status", "")).lower() == "done":
            return
        # Prepare input
        img_b64, img_src_key = self._read_image_b64(meta)
        prev_ev = self._find_prev_event(ev)
        prev_b64, prev_src_key = (None, None)
        if prev_ev is not None:
            prev_b64, prev_src_key = self._read_image_b64(prev_ev.get("meta") or {})
        prompt = (
            "You are an assistant that classifies whether a changed region of a fixed camera is a meaningful change. "
            "You may be given two images: image 1 is the PREVIOUS scene and image 2 is the CURRENT scene. "
            "Identify if there is a meaningful change (e.g., new/removed person/vehicle/animal) vs trivial lighting/noise. "
            "Return strict JSON with the following keys: significant (boolean), reason (short string), "
            "and bbox (array [x1,y1,x2,y2] of normalized coordinates 0..1 in the CURRENT image). If unsure, use [0,0,0,0]."
        )
        significant = False
        reason = ""
        provider = (self.cfg.provider or "openai").strip().lower()
        model = self.cfg.model_fast or "gpt-4o-mini"
        txt = ""
        error_msg: Optional[str] = None
        # Try provider; gracefully skip if client not available
        try:
            if provider == "openai":
                try:
                    from openai import OpenAI  # type: ignore
                except Exception:
                    raise RuntimeError("openai_client_unavailable")
                client = OpenAI()
                # Prepare messages, include previous then current when available
                content = [{"type": "text", "text": prompt}]
                if prev_b64:
                    content.append({"type": "image_url", "image_url": {"url": prev_b64}})
                if img_b64:
                    content.append({"type": "image_url", "image_url": {"url": img_b64}})
                # Execute
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.2,
                    max_tokens=80,
                    timeout=self.cfg.timeout_sec,
                )
                txt = (resp.choices[0].message.content or "").strip()
                # Try to parse JSON from the response
                meta_out_bbox = None
                try:
                    # Allow for fenced JSON
                    if txt.startswith("```"):
                        txt = txt.strip("` ")
                        if txt.lower().startswith("json"):
                            txt = txt[4:].strip()
                    data = json.loads(txt)
                    significant = bool(data.get("significant", False))
                    reason = str(data.get("reason", "")).strip()
                    bb = data.get("bbox")
                    if (
                        isinstance(bb, (list, tuple))
                        and len(bb) == 4
                        and all(isinstance(v, (int, float)) for v in bb)
                    ):
                        meta_out_bbox = [max(0.0, min(1.0, float(v))) for v in bb]
                except Exception:
                    # Heuristic fallback
                    significant = ("true" in txt.lower()) and (
                        "false" not in txt.lower()
                    )
                    reason = txt[:180]
                    meta_out_bbox = None
            elif provider == "openrouter":
                import requests
                api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError("openrouter_api_key_missing")
                url = "https://openrouter.ai/api/v1/chat/completions"
                content = [{"type": "text", "text": prompt}]
                if prev_b64:
                    content.append({"type": "image_url", "image_url": {"url": prev_b64}})
                if img_b64:
                    content.append({"type": "image_url", "image_url": {"url": img_b64}})
                payload = {
                    "model": model or "google/gemini-2.5-flash",
                    "messages": [
                        {"role": "user", "content": content},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 80,
                }
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                r = requests.post(url, json=payload, headers=headers, timeout=max(5, int(self.cfg.timeout_sec or 20)))
                if r.status_code >= 400:
                    raise RuntimeError(f"openrouter_http_{r.status_code}")
                data = r.json()
                msg = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                txt = (msg or "").strip()
                meta_out_bbox = None
                try:
                    if txt.startswith("```"):
                        txt = txt.strip("` ")
                        if txt.lower().startswith("json"):
                            txt = txt[4:].strip()
                    d = json.loads(txt)
                    significant = bool(d.get("significant", False))
                    reason = str(d.get("reason", "")).strip()
                    bb = d.get("bbox")
                    if (
                        isinstance(bb, (list, tuple))
                        and len(bb) == 4
                        and all(isinstance(v, (int, float)) for v in bb)
                    ):
                        meta_out_bbox = [max(0.0, min(1.0, float(v))) for v in bb]
                except Exception:
                    significant = ("true" in txt.lower()) and ("false" not in txt.lower())
                    reason = txt[:180]
                    meta_out_bbox = None
        except Exception as e:
            # Leave defaults; mark failure below
            error_msg = f"{type(e).__name__}: {e}"
        # Update event meta
        try:
            meta_out = dict(meta)
            meta_out["llm_attempted"] = True
            meta_out["llm_status"] = "done" if error_msg is None else "error"
            meta_out["llm_provider"] = provider
            meta_out["llm_model"] = model
            meta_out["llm_ts"] = time.time()
            meta_out["significant"] = bool(significant)
            if reason:
                meta_out["llm_reason"] = reason
            # Include a concise conversation trace (no images)
            meta_out["llm_prompt"] = prompt
            if txt:
                meta_out["llm_response"] = txt[:2000]
            if img_src_key:
                meta_out["llm_image_source"] = img_src_key
            if prev_ev is not None:
                try:
                    meta_out["llm_prev_event_id"] = int(prev_ev.get("id"))
                except Exception:
                    pass
            if prev_src_key:
                meta_out["llm_prev_image_source"] = prev_src_key
            if error_msg:
                meta_out["llm_error"] = error_msg
            try:
                if meta_out_bbox:
                    meta_out["llm_bbox_norm"] = meta_out_bbox
            except Exception:
                pass
            self.events.update(int(event_id), meta=meta_out)
            try:
                log.info(
                    "llm annotate event_id=%s status=%s significant=%s error=%s",
                    event_id,
                    meta_out.get("llm_status"),
                    meta_out.get("significant"),
                    error_msg,
                )
            except Exception:
                pass
        except Exception:
            pass
        # Emit a deduced event for visibility
        try:
            payload = {
                "event_id": int(ev["id"]),
                "status": "deduced" if error_msg is None else "error",
                "significant": bool(significant),
                "model": model,
            }
            if error_msg:
                payload["error"] = error_msg
            self.events.add("spot_investigation", payload)
        except Exception:
            pass
