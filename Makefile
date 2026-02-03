.PHONY: install install-all lint format test train clean

CONFIG ?= configs/default.yaml

install:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[dev,gbm,tune,explain,track,serve,viz]"

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

clean:
	rm -rf artifacts .pytest_cache .ruff_cache build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
