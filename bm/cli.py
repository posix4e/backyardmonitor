from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn


def _load_env_file(path: str | None) -> None:
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "=" not in s:
                continue
            k, v = s.split("=", 1)
            os.environ[k.strip()] = v.strip()
    except Exception:
        # Best-effort; ignore parse errors
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="backyardmonitor", description="Run the Backyard Monitor server"
    )
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument(
        "--env-file", default=os.getenv("ENV_FILE", ".env"), help="Path to .env file"
    )
    args = parser.parse_args()

    _load_env_file(args.env_file)

    uvicorn.run("bm.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
