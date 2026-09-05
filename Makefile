# Simple developer helpers (uv-first).

.PHONY: help venv sync sync-dense-st sync-sec format format-check lint lint-imports
.PHONY: type test test-architecture check pre-commit build
.PHONY: contract-check
.PHONY: smoke-embedding-api-wheel sec sec-run sec-hard sec-soft clean clean-all
.PHONY: docker-build compose-up compose-down

# Keep uv cache local to the repo so it's always writable (and it's already ignored).
VENV_DIR ?= .venv
UV_CACHE_DIR ?= .uv_cache
UV := UV_CACHE_DIR=$(UV_CACHE_DIR) UV_PROJECT_ENVIRONMENT=$(VENV_DIR) uv
PRE_COMMIT_HOME ?= .pre-commit-cache
VENV_PYTHON_STAMP := $(VENV_DIR)/.python-stamp
VENV_SYNC_STAMP := $(VENV_DIR)/.uv-sync-stamp
VENV_DENSE_ST_STAMP := $(VENV_DIR)/.uv-sync-dense-st-stamp
VENV_SEC_STAMP := $(VENV_DIR)/.uv-sec-stamp
IMAGE_NAME ?= intrinsical/rag-prototype
IMAGE_TAG ?= latest
WORKSPACE_CONTROL_ROOT ?= $(abspath ../../../workspace-control)
WORKSPACE_CONTROL_CACHE ?= /tmp/uv-cache

$(VENV_PYTHON_STAMP):
	@if [ -x "$(VENV_DIR)/bin/python" ] || [ -f "$(VENV_DIR)/Scripts/python.exe" ]; then \
		:; \
	elif [ -f "$(VENV_DIR)/pyvenv.cfg" ]; then \
		echo "Recreating unusable generated environment at $(VENV_DIR)"; \
		$(UV) venv --clear "$(VENV_DIR)"; \
	elif [ -e "$(VENV_DIR)" ]; then \
		echo "Refusing to replace existing non-virtualenv directory: $(VENV_DIR)" >&2; \
		exit 1; \
	else \
		$(UV) venv "$(VENV_DIR)"; \
	fi
	@touch "$@"

$(VENV_SYNC_STAMP): $(VENV_PYTHON_STAMP) pyproject.toml uv.lock
	$(UV) sync --frozen --group test --group lint --extra server --no-default-groups
	@touch "$@"

$(VENV_DENSE_ST_STAMP): $(VENV_PYTHON_STAMP) pyproject.toml uv.lock
	$(UV) sync --frozen --group test --group lint --extra server --extra dense-st --no-default-groups
	@touch "$@"

$(VENV_SEC_STAMP): $(VENV_PYTHON_STAMP) pyproject.toml uv.lock
	$(UV) sync --frozen --group test --group lint --group sec --extra server --no-default-groups
	@touch "$@"

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  %-14s %s\n", $$1, $$2}'

venv: $(VENV_PYTHON_STAMP) ## Create local virtual environment

sync: $(VENV_SYNC_STAMP) ## Sync locked test and lint dependencies

sync-dense-st: $(VENV_DENSE_ST_STAMP) ## Sync test/lint plus heavy SentenceTransformers dependencies

sync-sec: $(VENV_SEC_STAMP) ## Sync locked security tooling dependencies

format: sync ## Apply Ruff formatting and safe fixes
	$(UV) run --active --no-sync ruff check --fix src tests
	$(UV) run --active --no-sync ruff format src tests

format-check: sync ## Check Ruff formatting without modifying files
	$(UV) run --active --no-sync ruff format --check src tests

lint: sync ## Run Ruff without modifying files
	$(UV) run --active --no-sync ruff check --no-fix src tests

lint-imports: sync ## Run import-linter architecture contracts
	PYTHONPATH=src $(UV) run --active --no-sync lint-imports

type: sync ## Run mypy type checking
	DEBUG=false $(UV) run --no-sync mypy --python-executable $(VENV_DIR)/bin/python src/local_rag_backend

test: sync ## Run test suite
	$(UV) run --active --no-sync pytest -q

test-architecture: sync ## Run architecture guardrail tests only
	PYTHONPATH=src $(UV) run --active --no-sync lint-imports
	DEBUG=false $(UV) run --active --no-sync pytest -q -o addopts='' tests/architecture/test_*.py

check: format-check lint lint-imports type test ## Run local non-security gates

contract-check: ## Validate the workspace repository contract and README block.
	UV_CACHE_DIR=$(WORKSPACE_CONTROL_CACHE) uv run --project "$(WORKSPACE_CONTROL_ROOT)" --frozen --group dev workspace-control contract-check \
		--repo-root "$(CURDIR)" --repo-id repo-agentic-loopings-agentic-rag-prototype

pre-commit: sync ## Run all repository hooks
	PRE_COMMIT_HOME=$(PRE_COMMIT_HOME) $(UV) run --active --no-sync pre-commit run --all-files

build: sync ## Build wheel and source distributions
	$(UV) build

smoke-embedding-api-wheel: sync ## Build/install wheel and smoke public embedding API outside checkout
	UV_CACHE_DIR=$(abspath $(UV_CACHE_DIR)) UV_PROJECT_ENVIRONMENT=$(VENV_DIR) \
		uv run --no-sync python scripts/smoke_embedding_api_wheel.py

sec: sec-hard ## Run strict security checks

sec-run: sync-sec
	$(SEC_IGNORE)$(UV) run bandit -r src/ -ll -ii
	$(SEC_IGNORE)@requirements_file="$$(mktemp)"; \
	trap 'rm -f "$$requirements_file"' EXIT; \
	$(UV) export --frozen --no-default-groups --extra server --no-emit-project \
		--format requirements-txt --output-file "$$requirements_file" >/dev/null; \
	if [ -n "$(SAFETY_API_KEY)" ]; then \
		$(UV) run safety check --file "$$requirements_file" --full-report --key "$(SAFETY_API_KEY)"; \
	else \
		$(UV) run safety check --file "$$requirements_file" --full-report; \
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
	rm -rf $(VENV_DIR) .uv_cache $(PRE_COMMIT_HOME)

docker-build: ## Build production Docker image (IMAGE_NAME/IMAGE_TAG overridable)
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) --target production .

compose-up: ## Start docker compose stack
	docker compose up -d --build

compose-down: ## Stop docker compose stack
	docker compose down
