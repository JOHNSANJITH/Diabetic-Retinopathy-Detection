# Diabetic Retinopathy Detection — Explainable Computer Vision

A production-oriented computer vision pipeline for **5-class diabetic retinopathy severity classification** from retinal fundus images.

> **Portfolio / research prototype only.** This project is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## What this project demonstrates

- Transfer learning with Xception
- Reproducible dataset preparation and stratified splitting
- Retinal-image preprocessing and augmentation
- Class-imbalance-aware training
- Evaluation with accuracy, macro F1, per-class recall and confusion matrix
- Grad-CAM visual explanations
- FastAPI image inference
- Dockerized API deployment
- Automated tests for preprocessing and prediction contracts
- Configuration-driven, platform-independent paths

## Architecture

```text
Fundus image
    │
    ▼
Dataset / preprocessing
    │
    ▼
Augmentation (training only)
    │
    ▼
Xception transfer-learning classifier
    │
    ├── No DR
    ├── Mild DR
    ├── Moderate DR
    ├── Severe DR
    └── Proliferative DR
    │
    ├───────────────┐
    ▼               ▼
Evaluation       Grad-CAM
    │               │
    └───────┬───────┘
            ▼
        FastAPI API
            │
            ▼
          Docker
```

## Project structure

```text
.
├── api/
│   └── main.py
├── configs/
│   └── default.json
├── scripts/
│   ├── evaluate.py
│   ├── predict.py
│   ├── prepare_dataset.py
│   └── train.py
├── src/drd/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── explainability.py
│   ├── inference.py
│   ├── metrics.py
│   ├── model.py
│   └── preprocessing.py
├── tests/
│   ├── test_metrics.py
│   └── test_preprocessing.py
├── data/
├── models/
├── artifacts/
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .env.example
├── pyproject.toml
└── README.md
```

## Dataset

The original project was built around the **Diabetic Retinopathy Resized** dataset and its five severity labels. This repository intentionally does **not** include the dataset or trained weights.

Expected prepared layout:

```text
data/processed/
├── train/
│   ├── no_DR/
│   ├── mild_DR/
│   ├── moderate_DR/
│   ├── severe_DR/
│   └── proliferative_DR/
├── validation/
└── test/
```

Place the dataset under a local path and pass that path to the preparation script. No user-specific Windows paths are embedded in the code.

## Installation

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -e .
```

## Prepare a CSV-labelled dataset

For a Kaggle-style dataset containing an image folder and `trainLabels.csv`:

```bash
python scripts/prepare_dataset.py \
  --images-dir /path/to/resized_train/resized_train \
  --labels-csv /path/to/trainLabels.csv \
  --output-dir data/processed
```

The script copies images rather than moving them, creates deterministic train/validation/test splits, and reports missing files and class counts.

## Train

```bash
python scripts/train.py --config configs/default.json
```

Training uses ImageNet-pretrained Xception, freezes the convolutional base for the initial phase, computes class weights from the training split, and saves the best validation checkpoint. Fine-tuning can be enabled in the configuration after the classifier has stabilized.

## Evaluate

```bash
python scripts/evaluate.py \
  --model models/best.keras \
  --data-dir data/processed/test \
  --output-dir artifacts/evaluation
```

The evaluation artifact contains:

- accuracy
- macro F1
- weighted F1
- macro precision
- macro recall
- per-class precision / recall / F1 / support
- confusion matrix
- one-vs-rest ROC-AUC when every class is represented

**No benchmark numbers are hard-coded into this README.** Reported metrics should come from an actual run on the selected dataset split.

## Single-image inference

```bash
python scripts/predict.py \
  --model models/best.keras \
  --image /path/to/fundus.jpeg \
  --gradcam-output artifacts/gradcam.png
```

Example response:

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

## API

Start the service:

```bash
uvicorn api.main:app --reload
```

Health check:

```text
GET /health
```

Prediction:

```text
POST /predict
Content-Type: multipart/form-data
file=<image>
```

The API loads the model from `MODEL_PATH` and returns the predicted class, confidence, and full probability distribution. If Grad-CAM is requested, an explanation image can also be generated.

## Docker

Build:

```bash
docker build -t diabetic-retinopathy-cv .
```

Run with a trained model mounted into the container:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/models:/app/models" \
  diabetic-retinopathy-cv
```

Then open the FastAPI documentation at `/docs`.

## Explainability

Grad-CAM is implemented against the final convolutional feature map and produces a heatmap over the input image. It is intended as a debugging and interpretability aid, **not as proof that a highlighted region is a clinically meaningful lesion**.

## Testing

Run:

```bash
pytest -q
```

Tests cover preprocessing output contracts and metric generation without requiring a trained model or dataset.

## Engineering decisions

1. **Xception is retained from the original project** because it is a strong transfer-learning baseline for high-resolution image classification.
2. **ImageNet preprocessing is applied consistently** for training, evaluation, and inference.
3. **Class weights are calculated from the training split** to reduce the effect of imbalanced severity labels.
4. **Validation/test data receive no random augmentation.**
5. **Metrics are generated from real predictions** rather than manually entered benchmark values.
6. **Paths are configuration/CLI driven**, so the repository is portable across Windows, Linux, Docker and CI.
7. **The old webcam workflow is removed from the primary pipeline** because a webcam frame is not equivalent to a calibrated retinal fundus photograph.
8. **Grad-CAM is treated as an explanation/debugging tool**, not a diagnostic guarantee.

## Limitations

- Dataset shift can materially affect performance across cameras, populations and acquisition settings.
- Five-class severity labels are ordinal and may be difficult to separate at adjacent grades.
- Accuracy alone is insufficient for an imbalanced medical classification problem.
- Grad-CAM can be unstable and should not be interpreted as lesion segmentation.
- Clinical deployment would require external validation, calibration, safety evaluation, regulatory review and prospective testing.

## License

The repository retains the original project's license file. Dataset licensing and terms must be checked separately at the dataset source before redistribution.
