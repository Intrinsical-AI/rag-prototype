.PHONY: build run test clean

build:
	docker compose build

run:
	docker compose up

run-with-ollama:
	docker compose --profile with-ollama up

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

summary:
	python scripts/concatenate_src.py
	python scripts/concatenate_tests.py