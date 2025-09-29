# Simple developer helpers

.PHONY: lint type test sec docker-build compose-up compose-down

lint:
	uv run ruff check .
	uv run black --check .
	uv run isort --check-only .

type:
	uv run mypy .

test:
	uv run python -m pytest -q

sec:
	- bandit -r src/ -q
	- safety check -q

docker-build:
	docker build -t intrinsical-rag-prototype:latest --target production .

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down
