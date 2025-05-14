.PHONY: build run test clean

build:
	docker compose build

run:
	docker compose up

run-with-ollama:
	docker compose --profile with-ollama up

test:
	pytest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
