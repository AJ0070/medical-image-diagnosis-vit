You are a senior Machine Learning Engineer and AI Research Engineer.

I want to build a production-quality Medical Image Diagnosis System using Vision Transformers that is worthy of an AI/ML Engineer resume.

The project should look like something built by an ML engineer at Google, Microsoft, NVIDIA, OpenAI, or a medical AI startup.

The objective is NOT just training a model.

It should demonstrate the complete machine learning lifecycle including:

• Dataset preparation
• Data preprocessing
• Data augmentation
• Transfer Learning
• Model experimentation
• Explainable AI
• Hyperparameter tuning
• Evaluation
• Deployment
• API development
• Dashboard
• Dockerization
• Reproducibility

The codebase should follow professional software engineering practices with clean architecture.

======================================================
PROJECT OBJECTIVE
======================================================

Build an AI system capable of classifying multiple medical imaging tasks.

The project should support independent training for the following datasets:

1. Chest X-Ray Pneumonia Detection

Input:
Chest X-Ray

Output:
Normal
Pneumonia

Dataset:
Kaggle Chest X-Ray Pneumonia Dataset

--------------------------------------

2. Brain Tumor Classification

Classes:

Glioma
Meningioma
Pituitary
No Tumor

Dataset:
Kaggle Brain MRI Dataset

--------------------------------------

3. Skin Cancer Classification

Classes:

Benign
Malignant

Dataset:
ISIC or HAM10000

--------------------------------------

4. Diabetic Retinopathy Detection

Classes:

0
1
2
3
4

Dataset:
APTOS 2019 Blindness Detection

The project should be modular so new diseases can easily be added.

======================================================
TECH STACK
======================================================

Python

PyTorch

Torchvision

Vision Transformer (ViT)

EfficientNet

OpenCV

Albumentations

Scikit-learn

NumPy

Pandas

Matplotlib

Seaborn

TensorBoard

Weights & Biases

Grad-CAM

FastAPI

Streamlit

Docker

GitHub Actions

======================================================
PROJECT STRUCTURE
======================================================

Create a production-ready structure.

medical-image-diagnosis-vit/

│

├── configs/

├── datasets/

├── models/

│ ├── vit.py

│ ├── efficientnet.py

│ ├── trainer.py

│ ├── evaluator.py

│ ├── losses.py

│ ├── metrics.py

│

├── preprocessing/

│ ├── transforms.py

│ ├── augmentation.py

│ ├── dataset.py

│

├── explainability/

│ ├── gradcam.py

│ ├── attention_maps.py

│

├── inference/

│ ├── predictor.py

│

├── api/

│ ├── FastAPI

│

├── dashboard/

│ ├── Streamlit

│

├── notebooks/

├── scripts/

├── checkpoints/

├── results/

├── docker/

├── tests/

├── docs/

└── README.md

======================================================
MODEL REQUIREMENTS
======================================================

Implement

Vision Transformer

ViT-B/16

ViT-L

EfficientNet-B0

EfficientNet-B3

Compare both architectures.

Support transfer learning using pretrained ImageNet weights.

Allow freezing and unfreezing layers.

======================================================
DATA PIPELINE
======================================================

Implement

Dataset downloading instructions

Automatic train-validation-test split

Custom PyTorch Dataset

Image normalization

Resize

Random crop

Horizontal flip

Vertical flip

Random rotation

Gaussian noise

Brightness adjustment

Contrast adjustment

CLAHE for medical images

Albumentations pipeline

Support configurable augmentation.

======================================================
TRAINING PIPELINE
======================================================

Training script should support

Mixed Precision (AMP)

Gradient accumulation

Early stopping

Learning rate scheduler

Warmup scheduler

Checkpoint saving

Resume training

Automatic logging

TensorBoard

W&B integration

Multi-GPU support

Random seed

Configuration file

======================================================
LOSS FUNCTIONS
======================================================

Support

Cross Entropy

Weighted Cross Entropy

Focal Loss

Class-balanced loss

======================================================
OPTIMIZERS
======================================================

Support

Adam

AdamW

SGD

RMSProp

======================================================
HYPERPARAMETER TUNING
======================================================

Implement

Optuna

Search over

Learning rate

Weight decay

Batch size

Dropout

Optimizer

Scheduler

Image size

Number of epochs

Generate optimization plots.

======================================================
EVALUATION
======================================================

Calculate

Accuracy

Precision

Recall

Specificity

Sensitivity

F1 Score

ROC-AUC

PR-AUC

Confusion Matrix

Classification Report

Cohen's Kappa

Matthews Correlation Coefficient

======================================================
MODEL COMPARISON
======================================================

Generate comparison tables for

Vision Transformer

EfficientNet

Compare

Training time

Inference time

Accuracy

F1

ROC-AUC

Model size

Number of parameters

Memory usage

======================================================
EXPLAINABILITY
======================================================

Implement Grad-CAM.

For every prediction generate

Original image

Heatmap

Overlay

Confidence score

Predicted class

Ground truth

Generate attention visualization for Vision Transformers.

Explain why the model predicted a disease.

======================================================
INFERENCE
======================================================

Create an inference pipeline.

User uploads a medical image.

Pipeline should

Preprocess image

Run inference

Generate probability scores

Generate Grad-CAM

Generate attention map

Return prediction

Confidence

Disease explanation

======================================================
FASTAPI
======================================================

Build REST API.

Endpoints

/health

/predict

/model-info

/metrics

/docs

======================================================
STREAMLIT DASHBOARD
======================================================

Dashboard should include

Image upload

Prediction

Probability chart

Grad-CAM visualization

Attention map

Model comparison

Training metrics

Confusion matrix

ROC curve

======================================================
EXPERIMENT TRACKING
======================================================

Track

Loss

Accuracy

Learning rate

GPU usage

Training time

Validation metrics

Best model

======================================================
DOCKER
======================================================

Create

Dockerfile

docker-compose.yml

GPU support

======================================================
TESTING
======================================================

Unit tests

Dataset tests

Inference tests

API tests

======================================================
README
======================================================

Create an exceptional README including

Project overview

Architecture diagram

Dataset description

Training instructions

Evaluation

Results

Model comparison

Screenshots

API examples

Docker usage

Future improvements

======================================================
BONUS FEATURES
======================================================

Implement

ONNX export

TorchScript export

Quantization

Knowledge Distillation

Test-Time Augmentation

Model Ensemble

Automatic report generation

Batch inference

CSV prediction export

CLI interface

Model registry

======================================================
CODE QUALITY
======================================================

Use

Type hints

Docstrings

Logging

Black

isort

flake8

pytest

Pre-commit hooks

Professional modular code

Everything should be production-ready, scalable, and follow ML engineering best practices.

======================================================
GIT WORKFLOW
======================================================

Develop this project as if it were a real production project.

Make Git commits throughout development instead of completing everything in one large commit.

Commit whenever a logical unit of work is completed, such as:

- Project initialization
- Dataset pipeline implementation
- Data preprocessing
- Data augmentation
- Vision Transformer implementation
- EfficientNet implementation
- Training pipeline
- Evaluation metrics
- Explainability features
- FastAPI endpoints
- Streamlit dashboard
- Docker support
- Testing
- Documentation
- CI/CD configuration
- Bug fixes
- Refactoring
- Performance improvements

Every commit should represent one meaningful change.

Commit messages must:

- Follow Conventional Commits.
- Be concise and descriptive.
- Describe the actual change made.
- Never reference development stages or implementation order.
- Never use messages such as "Phase 1", "Step 2", "Progress", "Continue working", "WIP", or similar.
- Never bundle unrelated changes into a single commit.

Examples of good commit messages:

feat: implement Vision Transformer training pipeline

feat: add EfficientNet transfer learning support

feat: add Grad-CAM visualization for predictions

feat: create FastAPI inference endpoint

feat: build Streamlit prediction dashboard

feat: implement Optuna hyperparameter optimization

feat: export trained models to ONNX

perf: optimize dataloader with pinned memory

refactor: simplify dataset preprocessing pipeline

refactor: modularize model configuration

fix: correct class mapping during inference

fix: prevent invalid image uploads

test: add unit tests for prediction pipeline

test: add API endpoint integration tests

docs: document training and inference workflow

ci: add GitHub Actions for automated testing

At the end of every completed feature, immediately create an appropriate commit before moving on to the next logical change.

The Git history should resemble that of an experienced software engineer working on a professional open-source project, with clean, atomic, and meaningful commits.