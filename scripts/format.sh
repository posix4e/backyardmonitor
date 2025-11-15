#!/usr/bin/env bash
set -euo pipefail

MODE="fix"
if [[ "${1:-}" == "--check" ]]; then
  MODE="check"
fi

echo "[fmt] Python (black) — ${MODE}"
if [[ "$MODE" == "check" ]]; then
  black --check . || echo "[fmt] Warning: black check had issues" >&2
else
  black . || echo "[fmt] Warning: black formatting failed on some files" >&2
fi

echo "[fmt] HTML (djlint) — ${MODE}"
HTML_DIR="bm/static"
if command -v djlint >/dev/null 2>&1; then
  if [[ -d "$HTML_DIR" ]]; then
    if [[ "$MODE" == "check" ]]; then
      djlint "$HTML_DIR" --check
    else
      djlint "$HTML_DIR" --reformat
    fi
  else
    echo "[fmt] Skipped: ${HTML_DIR} not found"
  fi
else
  echo "[fmt] Skipped: djlint not installed (uv sync --group dev)" >&2
fi

echo "[fmt] JavaScript (js-beautify) — ${MODE}"
JS_DIR="bm/static/js"
if [[ -d "$JS_DIR" ]]; then
  if command -v js-beautify >/dev/null 2>&1; then
    if [[ "$MODE" == "check" ]]; then
      tmpdir=$(mktemp -d)
      rsync -a --exclude='*.min.js' "$JS_DIR/" "$tmpdir/"
      find "$tmpdir" -name "*.js" -type f -print0 | xargs -0 js-beautify -r --end-with-newline -n
      if ! diff -ru "$JS_DIR" "$tmpdir" >/dev/null; then
        echo "[fmt] JS formatting differences found" >&2
        diff -ru "$JS_DIR" "$tmpdir" | sed -n '1,200p' || true
        rm -rf "$tmpdir"
        exit 2
      fi
      rm -rf "$tmpdir"
    else
      find "$JS_DIR" -name "*.js" -type f -print0 | xargs -0 js-beautify -r --end-with-newline -n
    fi
  else
    # Fallback to Python API
    if python - <<'PY'
import sys, os
try:
    import jsbeautifier
except Exception:
    sys.exit(1)
opts = jsbeautifier.default_options()
opts.end_with_newline = True
for root, _, files in os.walk(sys.argv[1]):
    for f in files:
        if f.endswith('.js'):
            p = os.path.join(root, f)
            with open(p, 'r', encoding='utf-8') as fh:
                src = fh.read()
            out = jsbeautifier.beautify(src, opts)
            if out != src:
                with open(p, 'w', encoding='utf-8') as fh:
                    fh.write(out)
PY
    "$JS_DIR"; then
      echo "[fmt] JS formatted via Python fallback"
    else
      echo "[fmt] Skipped: js-beautify/jsbeautifier not installed (uv sync --group dev)" >&2
    fi
  fi
else
  echo "[fmt] Skipped: ${JS_DIR} not found"
fi

echo "[fmt] Done"
