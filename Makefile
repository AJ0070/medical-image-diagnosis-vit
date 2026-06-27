.PHONY: help install train evaluate infer tune compare export api dashboard test lint format docker-build

help:
	@echo "Medical Image Diagnosis ViT — available commands:"
	@echo "  install        Install dependencies"
	@echo "  train          Train a model (CONFIG=configs/chest_xray.yaml)"
	@echo "  evaluate       Evaluate model (CONFIG=... CKPT=...)"
	@echo "  infer          Run inference (CONFIG=... CKPT=... INPUT=...)"
	@echo "  tune           Optuna hyperparameter search"
	@echo "  compare        Compare ViT and EfficientNet architectures"
	@echo "  export         Export to ONNX and TorchScript"
	@echo "  api            Start FastAPI server"
	@echo "  dashboard      Start Streamlit dashboard"
	@echo "  test           Run test suite"
	@echo "  lint           Run linters"
	@echo "  format         Auto-format code"
	@echo "  docker-build   Build Docker image"

install:
	pip install -r requirements.txt && pip install -e . --no-deps

train:
	python scripts/train.py --config $(CONFIG)

evaluate:
	python scripts/evaluate.py --config $(CONFIG) --checkpoint $(CKPT)

infer:
	python scripts/infer.py --config $(CONFIG) --checkpoint $(CKPT) --input $(INPUT)

tune:
	python scripts/tune.py --config $(CONFIG)

compare:
	python scripts/compare_models.py --config $(CONFIG) --epochs 15

export:
	python scripts/export.py --config $(CONFIG) --checkpoint $(CKPT)

api:
	CONFIG_PATH=$(CONFIG) CHECKPOINT_PATH=$(CKPT) uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v --tb=short

lint:
	flake8 --max-line-length=100 --extend-ignore=E203,W503 .
	mypy . --ignore-missing-imports

format:
	black .
	isort .

docker-build:
	docker build -f docker/Dockerfile -t medical-image-diagnosis-vit:latest .
