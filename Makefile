.PHONY: install download graphs train backtest dashboard test docker-build docker-run

install:
	pip install -e .

download:
	python scripts/download_data.py

graphs:
	python scripts/build_graphs.py

train:
	python scripts/train_model.py

backtest:
	python scripts/run_backtest.py

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

docker-build:
	docker build -t smart-portfolio-gnn .

docker-run:
	docker run -p 8501:8501 smart-portfolio-gnn
