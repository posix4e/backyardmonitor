#!/usr/bin/env bash
set -euo pipefail

paths=("bm")
if [[ -d tests ]]; then
  paths+=("tests")
fi

echo "[lint] Pyflakes on: ${paths[*]}"
if ! command -v pyflakes >/dev/null 2>&1; then
  echo "[lint] pyflakes not installed. Run: uv sync --group dev" >&2
  exit 1
fi

pyflakes "${paths[@]}"
echo "[lint] Done"

