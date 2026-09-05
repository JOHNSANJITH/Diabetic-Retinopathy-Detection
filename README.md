# Diabetic Retinopathy Detection — Explainable Computer Vision

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![Pytest](https://img.shields.io/badge/Tests-Pytest-0A9EDC.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A production-oriented computer vision pipeline for **5-class diabetic retinopathy severity classification** from retinal fundus images using Xception, Grad-CAM, FastAPI and Docker.

**Portfolio / research prototype only.** This project is not a medical device and must not be used for clinical diagnosis or treatment decisions.

---

## Overview

This project demonstrates an end-to-end computer vision workflow covering:

- Transfer learning with Xception
- Reproducible dataset preparation
- Stratified train/validation/test splitting
- Retinal-image preprocessing and augmentation
- Class-imbalance-aware training
- Model evaluation and error analysis
- Grad-CAM visual explanations
- Single-image inference
- FastAPI REST inference
- Dockerized deployment
- Automated testing
- Configuration-driven execution

The emphasis is on building an **engineered ML system**, not only training a neural network.

---

## System Capabilities

| Capability | Implementation |
|---|---|
| Image Classification | 5-class diabetic retinopathy severity |
| Backbone | Xception |
| Transfer Learning | ImageNet pretrained weights |
| Preprocessing | Resize + Xception preprocessing |
| Augmentation | Training-only augmentation |
| Imbalance Handling | Training-set class weights |
| Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC |
| Explainability | Grad-CAM |
| Inference | CLI + REST API |
| API | FastAPI |
| Deployment | Docker |
| Testing | Pytest |
| Configuration | JSON + CLI |
| Reproducibility | Deterministic splits and configuration |

---

## Architecture

```text
                         ┌──────────────────────┐
                         │    Fundus Image      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Dataset Preparation  │
                         │ Validation + Splits  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Image Preprocessing  │
                         │ Resize + Normalize   │
                         └──────────┬───────────┘
                                    │
                              Training only
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Data Augmentation    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │           Xception            │
                    │     ImageNet Transfer         │
                    │          Learning             │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ 5-Class Classifier   │
                         └──────────┬───────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
           Prediction          Evaluation           Grad-CAM
                │                   │                   │
                └───────────────────┼───────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI         │
                         │   Inference API      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       Docker         │
                         └──────────────────────┘
```

---

## Classification

| Label | Severity |
|---:|---|
| `0` | No DR |
| `1` | Mild DR |
| `2` | Moderate DR |
| `3` | Severe DR |
| `4` | Proliferative DR |

The task is implemented as multi-class classification while recognizing that diabetic-retinopathy severity is ordinal.

---

## Project Structure

```text
.
├── api/
│   └── main.py
│
├── configs/
│   └── default.json
│
├── scripts/
│   ├── evaluate.py
│   ├── predict.py
│   ├── prepare_dataset.py
│   └── train.py
│
├── src/
│   └── drd/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── explainability.py
│       ├── inference.py
│       ├── metrics.py
│       ├── model.py
│       └── preprocessing.py
│
├── tests/
│   ├── test_metrics.py
│   └── test_preprocessing.py
│
├── data/
├── models/
├── artifacts/
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Dataset

The original project was built around the **Diabetic Retinopathy Resized** dataset and its five severity labels.

The dataset and trained weights are intentionally not included in this repository.

Expected prepared layout:

```text
data/processed/
├── train/
│   ├── no_DR/
│   ├── mild_DR/
│   ├── moderate_DR/
│   ├── severe_DR/
│   └── proliferative_DR/
│
├── validation/
│   ├── no_DR/
│   ├── mild_DR/
│   ├── moderate_DR/
│   ├── severe_DR/
│   └── proliferative_DR/
│
└── test/
    ├── no_DR/
    ├── mild_DR/
    ├── moderate_DR/
    ├── severe_DR/
    ├── proliferative_DR/
```

For a Kaggle-style dataset containing an image directory and `trainLabels.csv`:

```bash
python scripts/prepare_dataset.py \
  --images-dir /path/to/resized_train/resized_train \
  --labels-csv /path/to/trainLabels.csv \
  --output-dir data/processed
```

The preparation script:

- validates the CSV structure
- checks available images
- reports missing files
- performs deterministic stratified splitting
- reports class distributions
- copies images instead of moving them

No user-specific Windows paths are embedded in the source code.

---

## Installation

Python 3.10–3.12 is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Training

Run:

```bash
python scripts/train.py --config configs/default.json
```

The training pipeline:

1. Loads the prepared dataset
2. Applies training augmentation
3. Applies Xception preprocessing
4. Calculates class weights from the training split
5. Loads ImageNet-pretrained Xception
6. Freezes the convolutional backbone initially
7. Trains the classification head
8. Monitors validation performance
9. Applies early stopping
10. Saves the best validation checkpoint

Expected model output:

```text
models/best.keras
```

Fine-tuning can be enabled through configuration after the classifier has stabilized.

---

## Evaluation

Run:

```bash
python scripts/evaluate.py \
  --model models/best.keras \
  --data-dir data/processed/test \
  --output-dir artifacts/evaluation
```

The evaluation pipeline produces:

| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness |
| Macro Precision | Average precision across classes |
| Macro Recall | Average recall across classes |
| Macro F1 | Balanced class-level performance |
| Weighted F1 | Support-weighted F1 |
| Per-Class Metrics | Class-specific performance |
| Confusion Matrix | Error distribution |
| ROC-AUC | One-vs-rest analysis when supported |

No benchmark numbers are hard-coded into this README.

Reported performance should always come from an actual training and evaluation run.

---

## Inference

Run prediction on an individual image:

```bash
python scripts/predict.py \
  --model models/best.keras \
  --image /path/to/fundus.jpeg
```

Generate a Grad-CAM visualization:

```bash
python scripts/predict.py \
  --model models/best.keras \
  --image /path/to/fundus.jpeg \
  --gradcam-output artifacts/gradcam.png
```

Example response format:

```json
{
  "predicted_class": "moderate_DR",
  "confidence": 0.87,
  "probabilities": {
    "no_DR": 0.01,
    "mild_DR": 0.05,
    "moderate_DR": 0.87,
    "severe_DR": 0.05,
    "proliferative_DR": 0.02
  }
}
```

The values above are illustrative and do not represent benchmark performance.

---

## REST API

Start the API:

```bash
uvicorn api.main:app --reload
```

API:

```text
http://localhost:8000
```

Interactive documentation:

```text
http://localhost:8000/docs
```

### Health

```http
GET /health
```

### Prediction

```http
POST /predict
Content-Type: multipart/form-data
file=<image>
```

Example:

```bash
curl -X POST \
  "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@fundus.jpeg"
```

The API loads the model from `MODEL_PATH` and returns:

- predicted class
- confidence
- class probability distribution
- optional Grad-CAM output

---

## Docker

Build:

```bash
docker build -t diabetic-retinopathy-cv .
```

Run:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/models:/app/models" \
  diabetic-retinopathy-cv
```

Open:

```text
http://localhost:8000/docs
```

Trained weights are mounted separately rather than baked into the container image.

---

## Explainability

Grad-CAM is implemented against the final convolutional feature map.

```text
Fundus Image
     │
     ▼
Xception
     │
     ▼
Target Prediction
     │
     ▼
Gradient Computation
     │
     ▼
Convolutional Feature Map
     │
     ▼
Grad-CAM Heatmap
     │
     ▼
Visualization
```

Grad-CAM is intended for:

- model debugging
- qualitative error analysis
- inspecting model attention
- investigating unexpected predictions

It should not be interpreted as lesion segmentation or clinical evidence.

---

## Testing

Run:

```bash
pytest -q
```

Current tests cover:

- preprocessing output contracts
- metric-generation behavior
- inference-related contracts

Test layout:

```text
tests/
├── test_metrics.py
└── test_preprocessing.py
```

---

## Configuration

Training configuration is stored in:

```text
configs/default.json
```

Example:

```json
{
  "data_dir": "data/processed",
  "model_path": "models/best.keras",
  "image_size": [299, 299],
  "batch_size": 16,
  "epochs": 15,
  "learning_rate": 0.0001,
  "validation_split": 0.2,
  "test_split": 0.1,
  "seed": 42,
  "fine_tune": false,
  "fine_tune_layers": 30,
  "dropout": 0.35
}
```

Configuration keeps model and training parameters separate from the implementation.

---

## Engineering Decisions

### Xception

Xception is retained from the original project as the primary transfer-learning baseline for high-resolution image classification.

### Transfer Learning

ImageNet-pretrained weights provide a useful initialization compared with training a high-capacity CNN from scratch.

### Class Weights

Class weights are calculated using the training split to reduce majority-class dominance.

### Evaluation Isolation

Validation and test data do not receive random training augmentation.

### Configuration-Driven Execution

Paths and training parameters are supplied through configuration and CLI arguments rather than hard-coded developer-specific paths.

### Image-Based Inference

The old webcam workflow is not part of the primary inference pipeline because generic webcam frames are not equivalent to properly acquired retinal fundus photographs.

### Grad-CAM

Grad-CAM is treated as an interpretability and debugging mechanism rather than a diagnostic guarantee.

### API Separation

Model logic is separated from the HTTP layer so that inference can be used through both CLI and REST interfaces.

---

# Development Workflow

This repository follows a feature-branch workflow.

```text
main
 │
 ├── feature/model-improvements
 │
 ├── feature/gradcam
 │
 ├── feature/api-inference
 │
 └── fix/preprocessing
          │
          ▼
      Pull Request
          │
          ▼
         main
```

## Create a Branch

```bash
git checkout main
git pull --ff-only origin main

git checkout -b feature/improve-gradcam
```

## Inspect Changes

```bash
git status
git diff
```

## Run Tests

```bash
pytest -q
```

## Commit

Use focused commits:

```bash
git add .
git commit -m "Improve Grad-CAM visualization"
```

Examples:

```text
Add stratified dataset splitting
Improve Xception preprocessing
Add Grad-CAM visualization
Add FastAPI prediction endpoint
Improve evaluation metrics
Fix dataset validation
Add Docker inference configuration
```

## Push

```bash
git push -u origin feature/improve-gradcam
```

Open a Pull Request against `main`.

## Synchronize With Main

```bash
git fetch origin
git rebase origin/main
```

If conflicts occur:

```bash
git status
```

Resolve the affected files and continue:

```bash
git add .
git rebase --continue
```

If the branch was already pushed:

```bash
git push --force-with-lease
```

`--force-with-lease` is preferred over `--force` because it helps prevent accidental overwrites of newer remote work.

---

# Branch Naming

Use descriptive branch names:

```text
feature/model-improvements
feature/gradcam
feature/api-inference
feature/evaluation
fix/preprocessing
fix/dataset-splitting
docs/update-readme
```

Avoid vague branch names:

```text
test
new
changes
final
stuff
update
```

---

# Commit Convention

Prefer commits that describe one logical change.

Examples:

```text
Add stratified dataset preparation
Improve class weighting calculation
Add Grad-CAM inference output
Add FastAPI prediction endpoint
Add preprocessing tests
Improve Docker health check
Update evaluation metrics
```

Avoid:

```text
update
changes
final version
fixed things
new code
```

Focused commits make debugging, code review and rollback easier.

---

# Repository Quality Gates

Before merging a change into `main`:

```text
Code
 │
 ├── Tests pass
 │
 ├── No hard-coded local paths
 │
 ├── Configuration updated if required
 │
 ├── Documentation updated
 │
 ├── Generated artifacts excluded
 │
 └── Commit has a clear message
```

---

# Limitations

- Dataset shift can materially affect performance across cameras, populations and acquisition settings.
- Five-class severity labels are ordinal and adjacent grades may be difficult to distinguish.
- Accuracy alone is insufficient for an imbalanced medical classification problem.
- Grad-CAM can be unstable and should not be interpreted as lesion segmentation.
- Model confidence does not guarantee correctness.
- External validation is not established by this repository.
- Clinical deployment would require external validation, calibration, safety evaluation, regulatory review and prospective testing.

---

# Responsible Use

This repository is a portfolio and research prototype.

It is not intended for:

- Clinical diagnosis
- Treatment decisions
- Patient management
- Autonomous medical decision-making

Any real-world medical deployment would require substantially more validation, clinical oversight, safety engineering, calibration, regulatory assessment and prospective evaluation.

---

# Dataset and License

The original project was built around the **Diabetic Retinopathy Resized** dataset and its five severity labels.

The dataset and trained weights are intentionally not included in this repository.

Dataset licensing and redistribution terms must be checked separately at the original dataset source.

The repository retains the original project's `LICENSE` file.

---

# Portfolio Positioning

This project demonstrates the Computer Vision and Deep Learning side of an AI engineering portfolio.

The engineering workflow covers:

```text
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Training
   │
   ▼
Evaluation
   │
   ▼
Explainability
   │
   ▼
Inference
   │
   ▼
REST API
   │
   ▼
Docker
   │
   ▼
Testing
   │
   ▼
Git Workflow
```

The project is designed to demonstrate both **machine-learning capability and software-engineering discipline**.

---

# Project Status

| Component | Status |
|---|:---:|
| Dataset preparation | Complete |
| Preprocessing | Complete |
| Xception model | Complete |
| Class weighting | Complete |
| Training pipeline | Complete |
| Evaluation pipeline | Complete |
| Grad-CAM | Complete |
| CLI inference | Complete |
| FastAPI inference | Complete |
| Docker | Complete |
| Automated tests | Complete |
| Documentation | Complete |
| Real benchmark training | Complete |

> Benchmark metrics will be added only after an actual training and evaluation run on the selected dataset.

---

# Author

**John**

AI / ML Engineer

Python · Computer Vision · Deep Learning · LLM / RAG Systems
