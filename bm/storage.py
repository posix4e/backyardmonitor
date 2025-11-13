from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List
from typing import Set


class EventStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts REAL NOT NULL,
                  kind TEXT NOT NULL,
                  meta TEXT
                )
                """
            )

    def add(self, kind: str, meta: Dict[str, Any] | None = None) -> int:
        ts = time.time()
        meta_json = json.dumps(meta or {})
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute(
                "INSERT INTO events(ts, kind, meta) VALUES (?, ?, ?)",
                (ts, kind, meta_json),
            )
            return int(cur.lastrowid)

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT id, ts, kind, meta FROM events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            meta = json.loads(r["meta"]) if r["meta"] else {}
            out.append({"id": r["id"], "ts": r["ts"], "kind": r["kind"], "meta": meta})
        return out

    def get(self, event_id: int) -> Dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            r = con.execute(
                "SELECT id, ts, kind, meta FROM events WHERE id = ?", (event_id,)
            ).fetchone()
        if not r:
            return None
        meta = json.loads(r["meta"]) if r["meta"] else {}
        return {"id": r["id"], "ts": r["ts"], "kind": r["kind"], "meta": meta}

    def spot_events(self, spot_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        # Simple LIKE on JSON blob for portability (avoid requiring SQLite JSON1)
        like = f'%"spot_id":"{spot_id}"%'
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT id, ts, kind, meta FROM events WHERE kind = 'spot_change' AND meta LIKE ? ORDER BY ts DESC LIMIT ?",
                (like, int(limit)),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            meta = json.loads(r["meta"]) if r["meta"] else {}
            out.append({"id": r["id"], "ts": r["ts"], "kind": r["kind"], "meta": meta})
        return out

    def update(
        self,
        event_id: int,
        kind: str | None = None,
        meta: Dict[str, Any] | None = None,
        ts: float | None = None,
    ) -> bool:
        sets = []
        args: list[Any] = []
        if ts is not None:
            sets.append("ts = ?")
            args.append(float(ts))
        if kind is not None:
            sets.append("kind = ?")
            args.append(str(kind))
        if meta is not None:
            sets.append("meta = ?")
            args.append(json.dumps(meta))
        if not sets:
            return False
        args.append(int(event_id))
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute(f"UPDATE events SET {', '.join(sets)} WHERE id = ?", args)
            return cur.rowcount > 0

    def delete(self, event_id: int) -> bool:
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute("DELETE FROM events WHERE id = ?", (int(event_id),))
            return cur.rowcount > 0

    def referenced_images(self) -> Set[str]:
        """Return set of image filenames referenced by any event meta."""
        out: Set[str] = set()
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            for r in con.execute("SELECT meta FROM events"):
                try:
                    meta = json.loads(r["meta"]) if r["meta"] else {}
                    for k in ("image_full", "image_crop", "image_thumb"):
                        v = meta.get(k)
                        if v and isinstance(v, str):
                            out.add(v)
                except Exception:
                    continue
        return out

    def all(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT id, ts, kind, meta FROM events ORDER BY id ASC").fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            meta = json.loads(r["meta"]) if r["meta"] else {}
            out.append({"id": r["id"], "ts": r["ts"], "kind": r["kind"], "meta": meta})
        return out

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as con:
            r = con.execute("SELECT COUNT(*) FROM events").fetchone()
            return int(r[0]) if r else 0

    def older_than(self, ts_cutoff: float, limit: int = 1000) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT id, ts, kind, meta FROM events WHERE ts < ? ORDER BY ts ASC LIMIT ?",
                (float(ts_cutoff), int(limit)),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            meta = json.loads(r["meta"]) if r["meta"] else {}
            out.append({"id": r["id"], "ts": r["ts"], "kind": r["kind"], "meta": meta})
        return out

    def oldest(self, limit: int) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT id, ts, kind, meta FROM events ORDER BY ts ASC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            meta = json.loads(r["meta"]) if r["meta"] else {}
            out.append({"id": r["id"], "ts": r["ts"], "kind": r["kind"], "meta": meta})
        return out
