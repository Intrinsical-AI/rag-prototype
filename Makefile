# Simple developer helpers

.PHONY: venv sync lint type test sec docker-build compose-up compose-down

venv:
	uv venv .venv

sync: venv
	uv sync --frozen --extra test --extra lint

lint:
	$(MAKE) sync
	.venv/bin/ruff check .
	.venv/bin/black --check .
	.venv/bin/isort --check-only .

type:
	$(MAKE) sync
	.venv/bin/mypy .

test:
	$(MAKE) sync
	.venv/bin/python -m pytest -q

sec:
	- bandit -r src/ -q
	- safety check -q

docker-build:
	docker build -t intrinsical-rag-prototype:latest --target production .

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down
