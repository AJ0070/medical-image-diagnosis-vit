# Medical Image Diagnosis — Vision Transformers

> An AI system for classifying medical images across four tasks using Vision Transformers (ViT) and EfficientNet, with Grad-CAM explainability, a FastAPI REST API, Streamlit dashboard, and Docker support.

---

## Supported Tasks

| Task | Input | Classes | Dataset |
|------|-------|---------|---------|
| Chest X-Ray Pneumonia | Chest X-Ray | Normal, Pneumonia | [Kaggle CXR](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| Brain Tumor MRI | Brain MRI | Glioma, Meningioma, Pituitary, No Tumor | [Kaggle Brain MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) |
| Skin Cancer | Dermoscopy | Benign, Malignant | [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| Diabetic Retinopathy | Fundus | 0–4 Severity | [APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection) |

---

## Architecture

```
medical-image-diagnosis-vit/
├── configs/                  # YAML configs per dataset + base
├── preprocessing/            # Dataset, augmentation, transforms
├── models/                   # ViT, EfficientNet, losses, metrics, trainer, evaluator
├── explainability/           # Grad-CAM++, ViT attention rollout
├── inference/                # End-to-end predictor, TTA, ONNX export
├── api/                      # FastAPI REST API
├── dashboard/                # Streamlit interactive dashboard
├── scripts/                  # CLI: train, evaluate, infer, tune, compare, export
├── tests/                    # pytest unit + integration tests
├── docker/                   # Dockerfile + docker-compose
└── .github/workflows/        # CI/CD with GitHub Actions
```

### Model Architectures

| Model | Params | Image Size | Notes |
|-------|--------|------------|-------|
| ViT-B/16 | ~86M | 224×224 | Patch size 16, custom head |
| ViT-L/16 | ~307M | 224×224 | Larger backbone |
| EfficientNet-B0 | ~5M | 224×224 | Lightweight baseline |
| EfficientNet-B3 | ~12M | 300×300 | Better accuracy |

All models use **ImageNet pretrained weights** via [timm](https://github.com/huggingface/pytorch-image-models).

---

## Quickstart

### 1. Install dependencies

```bash
git clone https://github.com/your-username/medical-image-diagnosis-vit
cd medical-image-diagnosis-vit
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare dataset

Download from Kaggle and place in `datasets/<task_name>/`:

```
datasets/chest_xray/
    Normal/   (*.jpg images)
    Pneumonia/ (*.jpg images)
```

### 3. Train

```bash
# Chest X-Ray with ViT-B/16
python scripts/train.py --config configs/chest_xray.yaml

# Brain Tumor with EfficientNet-B3, override batch size
python scripts/train.py \
    --config configs/brain_tumor.yaml \
    --model efficientnet_b3 \
    --batch-size 16 \
    --epochs 30

# Resume from checkpoint
python scripts/train.py \
    --config configs/chest_xray.yaml \
    --resume checkpoints/chest-xray-pneumonia/checkpoint_epoch_010.pth
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --config configs/chest_xray.yaml \
    --checkpoint checkpoints/chest-xray-pneumonia/best_model.pth \
    --output-dir results/chest_xray
```

Outputs: `metrics.csv`, `confusion_matrix.png`, `roc_curves.png`, `pr_curves.png`

### 5. Hyperparameter Search

```bash
python scripts/tune.py \
    --config configs/chest_xray.yaml \
    --n-trials 50 \
    --storage sqlite:///optuna_study.db
```

### 6. Inference

```bash
# Single image
python scripts/infer.py \
    --config configs/chest_xray.yaml \
    --checkpoint checkpoints/chest-xray-pneumonia/best_model.pth \
    --input data/test_xray.png \
    --output-dir results/inference

# Batch directory
python scripts/infer.py \
    --config configs/chest_xray.yaml \
    --checkpoint checkpoints/chest-xray-pneumonia/best_model.pth \
    --input datasets/chest_xray/test/ \
    --output-dir results/inference
```

### 7. Export Model

```bash
# Export to ONNX and TorchScript
python scripts/export.py \
    --config configs/chest_xray.yaml \
    --checkpoint checkpoints/chest-xray-pneumonia/best_model.pth \
    --format both
```

### 8. Compare Architectures

```bash
python scripts/compare_models.py \
    --config configs/chest_xray.yaml \
    --epochs 15 \
    --output results/model_comparison.csv
```

---

## FastAPI

```bash
# Start server
CONFIG_PATH=configs/chest_xray.yaml \
CHECKPOINT_PATH=checkpoints/chest-xray-pneumonia/best_model.pth \
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Or directly
python api/main.py
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Classify image (multipart upload) |
| GET | `/model-info` | Model architecture details |
| GET | `/metrics` | Last evaluation metrics |
| GET | `/docs` | Interactive Swagger UI |

**Example:**

```bash
curl -X POST "http://localhost:8000/predict?gradcam=true" \
  -H "accept: application/json" \
  -F "file=@chest_xray.png;type=image/png"
```

```json
{
  "predicted_class": 1,
  "predicted_label": "Pneumonia",
  "confidence": 0.934,
  "probabilities": {"Normal": 0.066, "Pneumonia": 0.934},
  "model_name": "vit_b16",
  "dataset": "chest-xray-pneumonia",
  "has_gradcam": true,
  "gradcam_b64": "<base64-encoded-PNG>"
}
```

---

## Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard features:
- Image upload and real-time prediction
- Probability bar chart
- Grad-CAM / attention overlay visualization
- Confusion matrix and ROC curves
- Model comparison table

---

## Docker

```bash
# Build and run API
docker compose -f docker/docker-compose.yml up api

# Build and run dashboard
docker compose -f docker/docker-compose.yml up dashboard

# GPU support (requires nvidia-docker)
DEVICE=cuda docker compose -f docker/docker-compose.yml up
```

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_PATH` | `configs/chest_xray.yaml` | Dataset config |
| `CHECKPOINT_PATH` | `checkpoints/.../best_model.pth` | Model weights |
| `DEVICE` | `cpu` | `cpu` or `cuda` |

---

## Training Features

| Feature | Detail |
|---------|--------|
| Mixed Precision (AMP) | `torch.cuda.amp` — 2× speedup on modern GPUs |
| Gradient Accumulation | Simulate large batches on limited memory |
| Early Stopping | Monitor val F1, configurable patience |
| LR Schedulers | Cosine, Step, ReduceLROnPlateau, OneCycleLR |
| Linear Warmup | Stabilize early ViT training |
| Freeze/Unfreeze | Warm-start with frozen backbone, then fine-tune |
| WeightedRandomSampler | Handle class imbalance |
| Weighted CE / Focal Loss | Additional imbalance handling |
| TensorBoard + W&B | Full experiment tracking |
| Checkpoint Resume | Continue interrupted training |
| Multi-GPU | `torch.nn.DataParallel` |
| Random Seed | Full reproducibility |

---

## Evaluation Metrics

Accuracy · Precision · Recall · Specificity · Sensitivity · F1 · ROC-AUC · PR-AUC ·
Cohen's Kappa · Matthews Correlation Coefficient · Confusion Matrix

---

## Explainability

### Grad-CAM++ (EfficientNet)
Highlights regions the model attended to when making its decision.

### Attention Rollout (ViT)
Propagates attention weights through all transformer layers to produce a spatial heatmap.

### Output Per Prediction
- Original image
- Heatmap overlay
- Confidence score
- Class probabilities

---

## Hyperparameter Tuning (Optuna)

```yaml
# configs/hparam_search.yaml
search_space:
  lr: {type: float, low: 1e-5, high: 1e-2, log: true}
  batch_size: {type: categorical, choices: [16, 32, 64]}
  dropout: {type: float, low: 0.0, high: 0.5}
  optimizer: {type: categorical, choices: [adam, adamw, sgd]}
  model: {type: categorical, choices: [vit_b16, efficientnet_b0, efficientnet_b3]}
```

Results saved to `results/hparam_search_results.csv` with optimization history plots.

---

## Model Comparison Results

> Example (Chest X-Ray, 30 epochs, V100):

| Model | Params | Train Time | Infer (ms) | Accuracy | F1 | ROC-AUC |
|-------|--------|-----------|------------|----------|----|---------|
| ViT-B/16 | 86M | 24 min | 12.3 | 0.965 | 0.963 | 0.991 |
| EfficientNet-B0 | 5M | 8 min | 4.1 | 0.948 | 0.946 | 0.982 |
| EfficientNet-B3 | 12M | 11 min | 5.8 | 0.957 | 0.955 | 0.988 |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Coverage report
pytest tests/ --cov=. --cov-report=html

# Specific suite
pytest tests/test_losses_metrics.py -v
```

---

## Code Quality

```bash
# Format
black .
isort .

# Lint
flake8 .

# Type check
mypy .

# Pre-commit (runs all of the above)
pre-commit install
pre-commit run --all-files
```

---

## Project Structure

```
configs/
    base.yaml                  # Base config (all shared defaults)
    chest_xray.yaml            # Chest X-Ray task
    brain_tumor.yaml           # Brain Tumor task
    skin_cancer.yaml           # Skin Cancer task
    diabetic_retinopathy.yaml  # Diabetic Retinopathy task
    hparam_search.yaml         # Optuna search space

preprocessing/
    dataset.py      # MedicalImageDataset + DataModule
    transforms.py   # Image loading, denormalize, CLAHE helpers
    augmentation.py # Albumentations pipelines + TTA

models/
    vit.py          # ViT-B/16 and ViT-L/16 classifiers
    efficientnet.py # EfficientNet-B0 and B3 classifiers
    losses.py       # CE, Weighted CE, Focal, Class-Balanced
    metrics.py      # Full metrics: AUC, Kappa, MCC, specificity…
    trainer.py      # Full training loop (AMP, grad accum, logging)
    evaluator.py    # Evaluation + plot generation

explainability/
    gradcam.py         # GradCAM + GradCAM++
    attention_maps.py  # ViT attention rollout

inference/
    predictor.py   # End-to-end inference + ONNX/TorchScript export

api/
    main.py              # FastAPI app
    dependencies.py      # Singleton model loading
    routes/predict.py    # POST /predict
    routes/health.py     # GET /health
    routes/model_info.py # GET /model-info, /metrics
    schemas/prediction.py

dashboard/
    app.py   # Streamlit dashboard

scripts/
    train.py           # Training CLI
    evaluate.py        # Evaluation CLI
    infer.py           # Inference CLI
    tune.py            # Optuna HPO CLI
    compare_models.py  # Architecture comparison
    export.py          # ONNX / TorchScript export

docker/
    Dockerfile
    docker-compose.yml

tests/
    conftest.py
    test_models.py
    test_losses_metrics.py
    test_dataset.py
    test_api.py
    test_inference.py
```

---

## Future Improvements

- [ ] Multi-label classification support
- [ ] DICOM file format support
- [ ] Federated learning for privacy-preserving training
- [ ] Knowledge distillation (ViT → EfficientNet)
- [ ] Active learning pipeline
- [ ] Automatic radiology report generation
- [ ] Model registry with versioning (MLflow)
- [ ] Kubernetes deployment manifests

---

## License

MIT
