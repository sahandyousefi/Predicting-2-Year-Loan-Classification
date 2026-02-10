.PHONY: install install-all lint format test train serve docker-build docker-run clean

CONFIG ?= configs/default.yaml

install:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[dev,gbm,tune,explain,track,serve,viz,nn]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

test:
	pytest -q

train:
	credit-risk train --config $(CONFIG)

serve:
	uvicorn credit_risk.serving.api:create_app --factory --reload

docker-build:
	docker build -t credit-risk:latest .

docker-run:
	docker run --rm -p 8000:8000 -v $(PWD)/artifacts:/app/artifacts credit-risk:latest

clean:
	rm -rf artifacts .pytest_cache .ruff_cache build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
