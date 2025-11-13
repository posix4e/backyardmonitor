from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any


Point = Tuple[float, float]


@dataclass
class Spot:
    id: str
    name: str
    polygon: List[Point]


@dataclass
class Zones:
    gate: List[Point] | None  # polygon
    spots: List[Spot]

    @staticmethod
    def empty() -> "Zones":
        return Zones(gate=None, spots=[])

    def to_dict(self) -> Dict[str, Any]:
        gate_out: Any = None
        if self.gate:
            gate_out = {"polygon": [list(p) for p in self.gate]}
        return {"gate": gate_out, "spots": [asdict(s) for s in self.spots]}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Zones":
        gate_in = d.get("gate")
        # Support: {polygon:[...]}, legacy {center,w,h}, legacy {a,b}, or direct list
        gate_poly: List[Point] | None = None
        if gate_in:
            if isinstance(gate_in, dict) and "polygon" in gate_in:
                gate_poly = [tuple(p) for p in gate_in.get("polygon", [])]
            elif isinstance(gate_in, list) and gate_in and isinstance(gate_in[0], (list, tuple)):
                gate_poly = [tuple(p) for p in gate_in]
            elif isinstance(gate_in, dict) and "center" in gate_in:
                cx, cy = gate_in["center"]
                w = float(gate_in.get("w", 0))
                h = float(gate_in.get("h", 0))
                x0, y0 = cx - w/2.0, cy - h/2.0
                gate_poly = [(x0, y0), (x0+w, y0), (x0+w, y0+h), (x0, y0+h)]
            elif isinstance(gate_in, dict) and "a" in gate_in and "b" in gate_in:
                ax, ay = gate_in["a"]
                bx, by = gate_in["b"]
                cx, cy = ((ax+bx)/2.0, (ay+by)/2.0)
                w = abs(bx-ax) or 2.0
                h = abs(by-ay) or 20.0
                x0, y0 = cx - w/2.0, cy - h/2.0
                gate_poly = [(x0, y0), (x0+w, y0), (x0+w, y0+h), (x0, y0+h)]
        spots = [Spot(id=s["id"], name=s.get("name", s["id"]), polygon=[tuple(p) for p in s.get("polygon", [])]) for s in d.get("spots", [])]
        # Ignore any legacy 'grid' key if present to reduce confusion
        return Zones(gate=gate_poly, spots=spots)


def load_zones(path: Path) -> Zones:
    if not path.exists():
        return Zones.empty()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Zones.from_dict(data)


def save_zones(path: Path, zones: Zones) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(zones.to_dict(), f, indent=2)
