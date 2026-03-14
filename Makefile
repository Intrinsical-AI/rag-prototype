# Simple developer helpers (uv-first).

.PHONY: help venv sync sync-sec lint type test test-architecture sec sec-run sec-hard sec-soft clean clean-all docker-build compose-up compose-down

# Keep uv cache local to the repo so it's always writable (and it's already ignored).
UV_CACHE_DIR ?= .uv_cache
UV := UV_CACHE_DIR=$(UV_CACHE_DIR) uv
IMAGE_NAME ?= intrinsical/rag-prototype
IMAGE_TAG ?= latest

.venv/.python-stamp:
	$(UV) venv .venv
	@touch $@

.venv/.uv-sync-stamp: .venv/.python-stamp pyproject.toml uv.lock
	$(UV) sync --frozen --group test --group lint --extra server --no-default-groups
	@touch $@

.venv/.uv-sec-stamp: .venv/.python-stamp pyproject.toml uv.lock
	$(UV) sync --frozen --group test --group lint --group sec --extra server --no-default-groups
	@touch $@

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  %-14s %s\n", $$1, $$2}'

venv: .venv/.python-stamp ## Create local virtual environment

sync: .venv/.uv-sync-stamp ## Sync locked test and lint dependencies

sync-sec: .venv/.uv-sec-stamp ## Sync locked security tooling dependencies

lint: sync ## Run Ruff lint and format checks
	$(UV) run --active --no-sync ruff check .
	$(UV) run --active --no-sync ruff format --check .

type: sync ## Run mypy type checking
	DEBUG=false $(UV) run --no-sync mypy --python-executable .venv/bin/python src/local_rag_backend

test: sync ## Run test suite
	$(UV) run --active --no-sync pytest -q

test-architecture: sync ## Run architecture guardrail tests only
	DEBUG=false $(UV) run --active --no-sync pytest -q -o addopts='' tests/unit/http/test_architecture_*.py

sec: sec-hard ## Run strict security checks

sec-run: sync-sec
	$(SEC_IGNORE)$(UV) run bandit -r src/ -ll -ii
	$(SEC_IGNORE)@if [ -n "$(SAFETY_API_KEY)" ]; then \
		$(UV) run safety check --full-report --key "$(SAFETY_API_KEY)"; \
	else \
		$(UV) run safety check --full-report; \
	fi

sec-hard: SEC_IGNORE=
sec-hard: sec-run ## Run security checks and fail on findings

sec-soft: SEC_IGNORE=-
sec-soft: sec-run ## Run security checks without failing the target

clean: ## Remove cache, coverage, and Python build artifacts
	rm -rf \
		.pytest_cache .mypy_cache .ruff_cache \
		htmlcov .coverage coverage.xml pytest-results.xml \
		bandit-report.json safety-report.json
	find src tests -type d -name "__pycache__" -prune -exec rm -rf {} +
	find src tests -type f -name "*.py[cod]" -delete
	rm -rf src/*.egg-info

clean-all: clean ## Also remove local virtualenv and uv cache
	rm -rf .venv .uv_cache

docker-build: ## Build production Docker image (IMAGE_NAME/IMAGE_TAG overridable)
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) --target production .

compose-up: ## Start docker compose stack
	docker compose up -d --build

compose-down: ## Stop docker compose stack
	docker compose down
