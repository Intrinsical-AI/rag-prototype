# Simple developer helpers (uv-first).

.PHONY: venv sync lint type test sec clean docker-build compose-up compose-down

# Keep uv cache local to the repo so it's always writable (and it's already ignored).
UV_CACHE_DIR ?= .uv-cache
UV := UV_CACHE_DIR=$(UV_CACHE_DIR) uv

venv:
	$(UV) venv .venv

sync: venv
	$(UV) sync --frozen --extra test --extra lint

lint: sync
	$(UV) run --active --no-sync ruff check .
	$(UV) run --active --no-sync ruff format --check .

type: sync
	$(UV) run --active --no-sync mypy .

test: sync
	$(UV) run --active --no-sync pytest -q

sec:
	- $(UV) pip install bandit safety
	- $(UV) run bandit -r src/ -q
	- $(UV) run safety check -q

clean:
	rm -rf \
		.pytest_cache .mypy_cache .ruff_cache \
		htmlcov .coverage coverage.xml pytest-results.xml \
		bandit-report.json safety-report.json
	find src tests -type d -name "__pycache__" -prune -exec rm -rf {} +
	find src tests -type f -name "*.py[cod]" -delete
	rm -rf src/*.egg-info

docker-build:
	docker build -t intrinsical/rag-prototype:latest --target production .

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down
