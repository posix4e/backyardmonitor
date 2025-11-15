.PHONY: format format-check lint style style-check

format:
	uv run bash scripts/format.sh

format-check:
	uv run bash scripts/format.sh --check

lint:
	uv run bash scripts/lint.sh

# Unified formatter + linter
style:
	uv run bash scripts/style.sh

style-check:
	uv run bash scripts/style.sh --check
