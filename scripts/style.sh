#!/usr/bin/env bash
set -euo pipefail

# Unified style helper: format + lint
# Usage:
#   scripts/style.sh           # fix mode (formats, then lints)
#   scripts/style.sh --check   # check mode (no writes), fails on issues

MODE="fix"
if [[ "${1:-}" == "--check" ]]; then
  MODE="check"
fi

echo "[style] Mode: ${MODE}"

# 1) Format (Python, HTML, JS)
if [[ "$MODE" == "check" ]]; then
  bash "$(dirname "$0")/format.sh" --check
else
  bash "$(dirname "$0")/format.sh"
fi

# 2) Lint (Python)
bash "$(dirname "$0")/lint.sh"

echo "[style] Done"

