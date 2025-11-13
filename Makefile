.PHONY: help install run test lint clean

help:
	@echo "StockMate - Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make run        - Run the API server"
	@echo "  make test       - Run test suite"
	@echo "  make lint       - Run linters"
	@echo "  make clean      - Clean up generated files"

install:
	pip install -r requirements.txt

run:
	python run.py

test:
	pytest

test-cov:
	pytest --cov=app --cov-report=html --cov-report=term

lint:
	@echo "Running linters..."
	@command -v black >/dev/null 2>&1 && black --check app tests || echo "black not installed"
	@command -v isort >/dev/null 2>&1 && isort --check-only app tests || echo "isort not installed"
	@command -v flake8 >/dev/null 2>&1 && flake8 app tests || echo "flake8 not installed"

format:
	@echo "Formatting code..."
	@command -v black >/dev/null 2>&1 && black app tests || echo "black not installed"
	@command -v isort >/dev/null 2>&1 && isort app tests || echo "isort not installed"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
