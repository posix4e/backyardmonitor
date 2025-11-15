#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:  # nosec B310
        data = r.read()
        return json.loads(data.decode("utf-8"))


def gist_update(gist_id: str, token: str, filename: str, content: str) -> None:
    api = f"https://api.github.com/gists/{gist_id}"
    payload = {"files": {filename: {"content": content}}}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api,
        data=data,
        headers={
            "Authorization": f"token {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "User-Agent": "backyardmonitor-badge-updater",
        },
        method="PATCH",
    )
    with urllib.request.urlopen(req, timeout=10) as r:  # nosec B310
        if r.status >= 300:
            raise RuntimeError(f"Gist update failed: HTTP {r.status}")


def main() -> int:
    gist_id = os.getenv("GIST_ID")
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GIST_TOKEN")
    api_url = os.getenv("API_URL", "http://localhost:8080/api/version")
    label = os.getenv("BADGE_LABEL", "deployed")
    color = os.getenv("BADGE_COLOR", "blue")
    filename = os.getenv("BADGE_FILENAME", "badge.json")

    if not gist_id or not token:
        print("Missing GIST_ID or GITHUB_TOKEN env vars", file=sys.stderr)
        return 2

    try:
        v = http_get_json(api_url)
    except Exception as e:
        print(f"Failed to fetch version from {api_url}: {e}", file=sys.stderr)
        return 3

    ver = str(v.get("version") or "unknown")
    build = str(v.get("build") or "").strip()
    msg = f"v{ver}"
    if build:
        msg = f"{msg} ({build[:7]})"
    badge = {"schemaVersion": 1, "label": label, "message": msg, "color": color}

    try:
        gist_update(gist_id, token, filename, json.dumps(badge))
    except Exception as e:
        print(f"Failed to update gist: {e}", file=sys.stderr)
        return 4

    print(f"Updated badge gist {gist_id} with message: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
