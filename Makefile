.PHONY: install run test lint bench-cache bench-load bench-outage eval up down

install:
	pip install -e ".[dev]"

run:
	python -m gateway

test:
	pytest --cov=gateway --cov-report=term-missing

lint:
	ruff check .

bench-cache:
	python bench/cache_benchmark.py --requests 10000 --out docs/cache_benchmark.json

bench-load:
	python bench/loadtest.py --requests 5000 --concurrency 100 --unique --out docs/loadtest_uncached.json
	python bench/loadtest.py --requests 5000 --concurrency 100 --pool 200 --out docs/loadtest_mixed.json

bench-outage:
	python bench/outage_sim.py --provider mock-alpha --phase 10 --out docs/outage_sim.json

eval:
	python evals/run_eval.py --models auto,gamma-pro

up:
	docker compose up --build -d

down:
	docker compose down
