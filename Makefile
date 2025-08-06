# Makefile for ECD-to-SNAP project

.PHONY: help install test test-unit test-integration test-benchmarks test-quick clean format lint

help:
	@echo "ECD-to-SNAP Project Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install          Install project dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  test            Run all tests"
	@echo "  test-unit       Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-benchmarks  Run benchmark tests"
	@echo "  test-quick      Run quick validation tests"
	@echo "  clean           Remove Python cache files"
	@echo "  format          Format code with black and isort"
	@echo "  lint            Run code quality checks"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	python tests/run_tests.py all

test-unit:
	python tests/run_tests.py unit

test-integration:
	python tests/run_tests.py integration

test-benchmarks:
	python tests/run_tests.py benchmarks

test-quick:
	python tests/run_tests.py quick

test-pytest:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

format:
	@command -v black >/dev/null 2>&1 && black src/ tests/ scripts/ || echo "black not installed, skipping"
	@command -v isort >/dev/null 2>&1 && isort src/ tests/ scripts/ || echo "isort not installed, skipping"

lint:
	@command -v flake8 >/dev/null 2>&1 && flake8 src/ tests/ scripts/ --max-line-length=100 || echo "flake8 not installed, skipping"
	@command -v mypy >/dev/null 2>&1 && mypy src/ --ignore-missing-imports || echo "mypy not installed, skipping"

run-cli:
	python scripts/cli.py

optimize-identity:
	python scripts/cli.py optimize --target-type identity --layers 4 --truncation 6

optimize-linear:
	python scripts/cli.py optimize --target-type linear --target-param 0.1 --layers 6 --truncation 8

compare-strategies:
	python scripts/cli.py compare-strategies