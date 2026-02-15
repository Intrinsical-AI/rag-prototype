# Simple developer helpers (uv-first).

.PHONY: venv sync lint type test sec docker-build compose-up compose-down

# Keep uv cache local to the repo so it's always writable (and it's already ignored).
UV_CACHE_DIR ?= .uv-cache
UV := UV_CACHE_DIR=$(UV_CACHE_DIR) uv

venv:
	$(UV) venv .venv

sync: venv
	$(UV) sync --frozen --extra test --extra lint

lint: sync
	$(UV) run --active --no-sync ruff check .
	$(UV) run --active --no-sync black --check .
	$(UV) run --active --no-sync isort --check-only .

type: sync
	$(UV) run --active --no-sync mypy .

test: sync
	$(UV) run --active --no-sync pytest -q

sec:
	- $(UV) pip install bandit safety
	- $(UV) run bandit -r src/ -q
	- $(UV) run safety check -q

docker-build:
	docker build -t intrinsical-rag-prototype:latest --target production .

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down
