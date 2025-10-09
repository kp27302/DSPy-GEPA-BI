.PHONY: help install init-data etl eval gepa app clean

help:
	@echo "BI-DSPy-GEPA Makefile Commands:"
	@echo ""
	@echo "  make install      - Install dependencies"
	@echo "  make init-data    - Generate synthetic data"
	@echo "  make etl          - Run ETL pipeline"
	@echo "  make eval         - Evaluate SQL synthesis"
	@echo "  make gepa         - Run GEPA optimization"
	@echo "  make app          - Launch Streamlit dashboard"
	@echo "  make clean        - Clean generated files"
	@echo ""

install:
	pip install -r requirements.txt

init-data:
	python scripts/init_data.py

etl:
	python -m src.scripts.run_etl

eval:
	python -m src.scripts.eval_sql

gepa:
	python -m src.scripts.run_gepa_search

app:
	python -m src.scripts.run_app

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf data/warehouse/*.duckdb
	rm -rf data/warehouse/*.parquet
	rm -rf eval/results/*
	@echo "Cleaned generated files"

