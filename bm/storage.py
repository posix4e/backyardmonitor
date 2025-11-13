from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List


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
            cur = con.execute("INSERT INTO events(ts, kind, meta) VALUES (?, ?, ?)", (ts, kind, meta_json))
            return int(cur.lastrowid)

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT id, ts, kind, meta FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            meta = json.loads(r["meta"]) if r["meta"] else {}
            out.append({"id": r["id"], "ts": r["ts"], "kind": r["kind"], "meta": meta})
        return out

    def get(self, event_id: int) -> Dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            r = con.execute("SELECT id, ts, kind, meta FROM events WHERE id = ?", (event_id,)).fetchone()
        if not r:
            return None
        meta = json.loads(r["meta"]) if r["meta"] else {}
        return {"id": r["id"], "ts": r["ts"], "kind": r["kind"], "meta": meta}
