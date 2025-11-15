.PHONY: format format-check lint

format:
	uv run bash scripts/format.sh

format-check:
	uv run bash scripts/format.sh --check

lint:
	uv run bash scripts/lint.sh

