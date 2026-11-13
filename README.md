# CrimeWatch — End-to-End MLOps Pipeline for Firearm Detection

An automated firearm detection system built on **YOLOv8** with a fully reproducible
MLOps pipeline using **DVC** (data versioning) and **MLflow** (experiment tracking),
hosted on **DagsHub**.

---

## Project Links

| Resource | Link |
|---|---|
| GitHub Repository | `https://github.com/<your-org>/crimewatch` |
| DagsHub / MLflow | `https://dagshub.com/<your-user>/crimewatch` |

---

## What It Does

CrimeWatch processes surveillance footage to detect:
- Handguns and machine guns (custom-trained YOLOv8s)
- Persons carrying firearms (IoU-based association)
- Vehicles and license plates linked to armed suspects
- Getaway vehicles with evidence extraction

---

## Folder Structure

```
crimewatch/
├── data/
│   ├── raw/                  # Raw dataset (DVC-tracked, not in git)
│   │   ├── Handgun/
│   │   ├── Machine_Gun/
│   │   └── No_Gun/
│   └── processed/            # Extracted YOLO frames (DVC-tracked)
├── models/                   # Trained weights (DVC-tracked, not in git)
│   ├── baseline_best.pt      # YOLOv8n baseline
│   ├── main_best.pt          # YOLOv8s main model
│   ├── vehicle_detection.pt
│   └── license_plate.pt
├── src/
│   ├── data/
│   │   ├── ingest.py         # Stage 1: verify & manifest raw data
│   │   ├── validate.py       # Stage 2: schema & quality checks
│   │   └── prepare.py        # Stage 3: frame extraction + YOLO labels
│   ├── train/
│   │   ├── train_baseline.py # Stage 4a: YOLOv8n + MLflow logging
│   │   └── train_main.py     # Stage 4b: YOLOv8s + MLflow + registry
│   └── evaluate/
│       └── evaluate.py       # Stage 5: compare models, write metrics.json
├── templates/                # Flask HTML templates
├── app.py                    # Flask web app
├── main.py                   # Inference pipeline (video processing)
├── dvc.yaml                  # DVC pipeline definition
├── params.yaml               # All hyperparameters (single source of truth)
├── metrics.json              # DVC-tracked evaluation output
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/<your-org>/crimewatch.git
cd crimewatch
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure DVC remote (DagsHub)

```bash
dvc remote add origin https://dagshub.com/<your-user>/crimewatch.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <dagshub_username>
dvc remote modify origin --local password <dagshub_token>
```

### 3. Configure MLflow remote (DagsHub)

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<your-user>/crimewatch.mlflow
export MLFLOW_TRACKING_USERNAME=<dagshub_username>
export MLFLOW_TRACKING_PASSWORD=<dagshub_token>
```

Add these three lines to your `.env` file or shell profile to avoid
re-exporting every session.

### 4. Pull versioned data and models

```bash
dvc pull
```

---

## How to Run

### Run the full pipeline end-to-end

```bash
dvc repro
```

This executes all five stages in order, skipping any stage whose
inputs have not changed since the last run.

### Run individual stages

```bash
# Stage 1 — Ingest
python src/data/ingest.py --dataset_root data/raw

# Stage 2 — Validate
python src/data/validate.py

# Stage 3 — Prepare dataset
python src/data/prepare.py

# Stage 4a — Train baseline
python src/train/train_baseline.py

# Stage 4b — Train main model
python src/train/train_main.py

# Stage 5 — Evaluate both models
python src/evaluate/evaluate.py
```

### View the DVC pipeline graph

```bash
dvc dag
```

### Compare metrics across runs

```bash
dvc metrics show
dvc metrics diff
```

### Launch the Flask web app

```bash
python app.py
# Open http://localhost:5000
```

---

## DVC Pipeline Stages

| Stage | Script | Inputs | Outputs |
|---|---|---|---|
| `ingest` | `src/data/ingest.py` | `data/raw/` | `data/manifest.json` |
| `validate` | `src/data/validate.py` | `data/manifest.json` | `data/validation_report.json` |
| `prepare` | `src/data/prepare.py` | `data/raw/` | `data/processed/` |
| `train_baseline` | `src/train/train_baseline.py` | `data/processed/` | `models/baseline_best.pt` |
| `train_main` | `src/train/train_main.py` | `data/processed/` | `models/main_best.pt` |
| `evaluate` | `src/evaluate/evaluate.py` | both models | `metrics.json` |

---

## MLflow Experiment Tracking

All training runs are logged to DagsHub MLflow with:
- Hyperparameters: model size, epochs, batch size, patience, confidence threshold
- Metrics: mAP50, mAP50-95, precision, recall, per-class AP50
- Artifacts: best model checkpoint

The main YOLOv8s model is registered in the **MLflow Model Registry**
under `crimewatch-gun-detector` with status **Staging**.

---

## Models

| Model | Architecture | Role |
|---|---|---|
| Baseline | YOLOv8n | Lightweight reference model |
| Main | YOLOv8s | Primary firearm detector |
| Person tracker | YOLOv8n (pretrained) | Person detection + tracking |
| Vehicle | YOLOv8n (fine-tuned) | Vehicle detection |
| License plate | YOLOv8n (fine-tuned) | Plate localization |

---

## Group Members

| Name | GitHub |
|---|---|
| Member 1 | @username |
| Member 2 | @username |
| Member 3 | @username |
