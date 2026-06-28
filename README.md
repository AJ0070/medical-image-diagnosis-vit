# Medical Image Diagnosis — Vision Transformers

Deep learning pipeline for medical image classification across four tasks, demonstrating transfer learning, data augmentation, model comparison, hyperparameter tuning, and Grad-CAM interpretability.

---

## Tasks

| Task | Classes | Dataset |
|------|---------|---------|
| Chest X-Ray Pneumonia | Normal, Pneumonia | [Kaggle CXR](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| Brain Tumor MRI | Glioma, Meningioma, Pituitary, No Tumor | [Kaggle Brain MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) |
| Skin Cancer | Benign, Malignant | [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| Diabetic Retinopathy | 5-class severity (0–4) | [APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection) |

---

## Models

| Model | Params | Notes |
|-------|--------|-------|
| ViT-B/16 | 86M | Vision Transformer, patch size 16 |
| ViT-L/16 | 307M | Larger ViT backbone |
| EfficientNet-B0 | 5M | Lightweight CNN baseline |
| EfficientNet-B3 | 12M | Higher-accuracy CNN |

All models use **ImageNet pretrained weights** via [timm](https://github.com/huggingface/pytorch-image-models).

---

## Project Structure

```
configs/               # YAML configs per dataset + base
preprocessing/         # Dataset, augmentation pipeline (Albumentations), transforms
models/                # ViT, EfficientNet, loss functions, metrics, trainer, evaluator
explainability/        # Grad-CAM and Grad-CAM++ visualization
scripts/               # CLI: train, evaluate, infer, tune (Optuna), compare models
notebooks/             # Demo notebook
```

---

## Setup

```bash
git clone https://github.com/your-username/medical-image-diagnosis-vit
cd medical-image-diagnosis-vit

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

---

## Dataset Preparation

Download from Kaggle and organize as class-labelled folders:

```
datasets/chest_xray/
    Normal/      *.jpg
    Pneumonia/   *.jpg
```

Or use the download helper (requires Kaggle API key):

```bash
python scripts/download_datasets.py --dataset chest_xray
```

---

## Training

```bash
# Train ViT-B/16 on chest X-ray
python scripts/train.py --config configs/chest_xray.yaml

# Train EfficientNet-B3 on brain tumor MRI
python scripts/train.py --config configs/brain_tumor.yaml --model efficientnet_b3

# Quick smoke test — verifies full pipeline in ~2 min on CPU
python scripts/train.py --config configs/chest_xray.yaml --model efficientnet_b0 --fast-dev-run

# Resume from checkpoint
python scripts/train.py --config configs/chest_xray.yaml --resume checkpoints/chest-xray-pneumonia/best_model.pth
```

### Training features
- Transfer learning with frozen backbone → gradual unfreeze
- Mixed precision (AMP) on GPU
- Gradient accumulation
- Early stopping on val F1
- Cosine LR scheduler with linear warmup
- Weighted sampling for class imbalance
- TensorBoard logging

---

## Evaluation

```bash
python scripts/evaluate.py \
    --config configs/chest_xray.yaml \
    --checkpoint checkpoints/chest-xray-pneumonia/best_model.pth
```

Outputs: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Cohen's Kappa, confusion matrix, ROC curves.

---

## Hyperparameter Tuning (Optuna)

```bash
python scripts/tune.py --config configs/chest_xray.yaml --n-trials 50
```

Searches over: learning rate, weight decay, batch size, dropout, optimizer, scheduler, model architecture, image size.

---

## Model Comparison

```bash
python scripts/compare_models.py --config configs/chest_xray.yaml --epochs 15
```

Generates `results/model_comparison.csv` comparing ViT-B/16, EfficientNet-B0, and EfficientNet-B3 on accuracy, F1, ROC-AUC, params, and inference time.

---

## Inference with Grad-CAM

```bash
python scripts/infer.py \
    --config configs/chest_xray.yaml \
    --checkpoint checkpoints/chest-xray-pneumonia/best_model.pth \
    --input path/to/xray.png \
    --output-dir results/inference
```

Outputs the predicted class, confidence scores, and a Grad-CAM++ overlay image showing which regions influenced the prediction.

---

## Loss Functions

| Loss | Use case |
|------|----------|
| Cross-Entropy | Balanced datasets |
| Weighted Cross-Entropy | Class imbalance |
| Focal Loss | Severe imbalance (skin cancer, DR) |
| Label Smoothing CE | Regularization |

---

## Augmentation Pipeline (Albumentations)

Horizontal/vertical flip · Random rotation · Brightness/contrast jitter · CLAHE · Gaussian noise · ShiftScaleRotate · CoarseDropout

Configured per dataset in `configs/<task>.yaml`.
