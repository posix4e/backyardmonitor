# Repository Guidelines

## Project Structure & Module Organization
- `bm/`: FastAPI app and core modules (`app.py`, `capture.py`, `storage.py`, `zones.py`, `config.py`, `analyzer.py`).
- `bm/static/`: Single-page UI assets (`index.html`).
- `data/`: Runtime data (SQLite `events.db`, `frames/`, `analysis/`). Created at run-time; do not commit.
- `.env` / `.env.example`: Runtime configuration.
- Entry point: `bm/cli.py` exposed as `backyardmonitor`.

## Build, Test, and Development Commands
- Install (uv): `uv sync`
- Install (pip): `python -m venv .venv && . .venv/bin/activate && pip install -e .`
- Run server: `uv run backyardmonitor --reload --port 8080 --env-file .env`
- Alt run: `uv run uvicorn bm.app:app --reload --port 8080 --env-file .env`
- Environment: copy `.env.example` to `.env` and set `RTSP_URL`; optional: `DATA_DIR`, `AUTO_START`, `PHASH_MIN_BITS`, `PHASH_STABLE_MS`, retention knobs.

## Coding Style & Naming Conventions
- Python 3.9+; follow PEP 8 with 4‑space indents.
- Use type hints and `@dataclass` where appropriate; small, focused modules.
- Names: modules/files `snake_case`; classes `CapWords`; functions/vars `snake_case`.
- API JSON keys lower_snake_case; keep responses minimal and documented.
- Avoid introducing heavy frameworks; prefer standard library and existing patterns.

## Testing Guidelines
- No test suite yet. If adding tests, use `pytest` with files under `tests/` named `test_*.py`.
- Example: `tests/test_storage.py` for `EventStore` (use temp dirs/db files).
- Run: `uv run pytest` (or `pytest` inside the venv).
- Keep tests hermetic; do not depend on camera/RTSP in unit tests.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise; optional scope prefix (e.g., `chore:`, `feat:`).
- PRs: include description, steps to run locally, screenshots for UI changes, and linked issues.
- Check that `.env` and `data/` contents are not committed; document new env vars in README and here if needed.

## Security & Configuration Tips
- Required: `RTSP_URL`. Optional knobs: storage/retention (`STORE_FULL_FRAMES`, `STORE_CROPS`, `STORE_THUMBS`, `JPEG_QUALITY`, `RETAIN_DAYS`, `MAX_EVENTS`, `MAX_STORAGE_GB`).
- Defaults aim to be safe: thumbnails on, full-frame off.
- Use `POST /api/retention/apply` to prune when testing storage behavior.
