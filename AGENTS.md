# Repository Guidelines

## Project Structure & Module Organization
- `bm/`: FastAPI app and core modules (`app.py`, `capture.py`, `storage.py`, `zones.py`, `config.py`, `analyzer.py`).
- `bm/static/`: Single‑page UI assets (entry: `index.html`).
- `data/`: Runtime data (`events.db`, `frames/`, `analysis/`) created at run‑time; do not commit.
- `.env` / `.env.example`: Runtime configuration.
- Entry point: `bm/cli.py` exposed as `backyardmonitor`.

## Build, Test, and Development Commands
- Install (uv):
  ```bash
  uv sync
  ```
- Install (pip alt):
  ```bash
  python -m venv .venv && . .venv/bin/activate && pip install -e .
  ```
- Run server (hot reload):
  ```bash
  uv run backyardmonitor --reload --port 8080 --env-file .env
  # Alt:
  uv run uvicorn bm.app:app --reload --port 8080 --env-file .env
  ```
- Environment: copy `.env.example` to `.env`, set `RTSP_URL`; optional `DATA_DIR`, `AUTO_START`, `PHASH_MIN_BITS`, `PHASH_STABLE_MS`, and retention knobs.

## Coding Style & Naming Conventions
- Python 3.9+; follow PEP 8 with 4‑space indents.
- Use type hints and `@dataclass` where appropriate; keep modules small and focused.
- Naming: modules/files `snake_case`; classes `CapWords`; functions/vars `snake_case`.
- API JSON keys: lower_snake_case; keep responses minimal and documented.
- Prefer standard library and existing patterns; avoid heavy frameworks.

## Testing Guidelines
- Framework: `pytest` (no suite yet). Place tests under `tests/` as `test_*.py` (e.g., `tests/test_storage.py`).
- Run tests:
  ```bash
  uv run pytest
  ```
- Keep tests hermetic; do not depend on live camera/RTSP. Use temp dirs/db files.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise; optional scope prefix (e.g., `chore:`, `feat:`).
- PRs: include description, local run steps, screenshots for UI changes, and linked issues.
- Ensure `.env` and `data/` contents are not committed. Document any new env vars in README and this guide.

## Security & Configuration Tips
- Required: `RTSP_URL`. Optional storage/retention: `STORE_FULL_FRAMES`, `STORE_CROPS`, `STORE_THUMBS`, `JPEG_QUALITY`, `RETAIN_DAYS`, `MAX_EVENTS`, `MAX_STORAGE_GB`.
- Defaults favor safety (thumbnails on, full‑frame off). To test pruning, call:
  ```bash
  curl -X POST http://localhost:8080/api/retention/apply
  ```
