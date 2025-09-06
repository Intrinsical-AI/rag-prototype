.PHONY: build run test clean


build-docker:
	docker compose build

run-docker:
	docker compose up

run-docker-with-ollama:
	docker compose --profile with-ollama up


run:
	uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name htmlcov -exec rm -rf {} +
	find . -type f -name .ruff_cache -exec rm -rf {} +

summary:
	python scripts/concatenate_src.py
	python scripts/concatenate_tests.py


lint:
	black src tests;
	sleep 1;
	isort src tests;
	sleep 1;
	ruff check src tests;

