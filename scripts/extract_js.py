#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def extract_first_script(html_path: Path, js_out: Path, src_href: str) -> bool:
    txt = html_path.read_text(encoding="utf-8")
    # Match the first non-empty <script>...</script> block (not src)
    pat = re.compile(r"<script>(.*?)</script>", re.S | re.I)
    m = pat.search(txt)
    if not m:
        return False
    js_code = m.group(1).strip()
    js_out.parent.mkdir(parents=True, exist_ok=True)
    js_out.write_text(js_code + "\n", encoding="utf-8")
    # Replace the full block with a src tag
    new_txt = txt[: m.start()] + f'<script src="{src_href}"></script>' + txt[m.end() :]
    html_path.write_text(new_txt, encoding="utf-8")
    return True


def main() -> None:
    changed = False
    # index.html -> js/index.js, use absolute path since index is served at '/'
    idx_html = ROOT / "bm/static/index.html"
    if idx_html.exists():
        changed |= extract_first_script(
            idx_html, ROOT / "bm/static/js/index.js", "/static/js/index.js"
        )
    # spot.html -> js/spot.js; served under /static so relative works too; use absolute for consistency
    spot_html = ROOT / "bm/static/spot.html"
    if spot_html.exists():
        changed |= extract_first_script(
            spot_html, ROOT / "bm/static/js/spot.js", "/static/js/spot.js"
        )
    if not changed:
        print("No inline <script> blocks found or already extracted.")


if __name__ == "__main__":
    main()
