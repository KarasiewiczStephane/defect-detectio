.PHONY: install test lint clean run docker docker-build docker-run docker-compose-up docker-compose-down

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/ --fix
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m src.main

serve:
	python -m src.main serve --host 0.0.0.0 --port 8000

docker-build:
	docker build -t defect-detection:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/models:/app/models:ro defect-detection:latest

docker-compose-up:
	docker compose up -d

docker-compose-down:
	docker compose down

pre-commit:
	pre-commit run --all-files
