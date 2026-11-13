# DagsHub Setup Guide

Follow these steps ONCE to connect your repo to DagsHub for DVC and MLflow.

---

## Step 1 — Create a DagsHub repo

1. Go to https://dagshub.com and sign in (or sign up).
2. Click **New Repository**.
3. Choose **Connect a repository** and link your existing GitHub repo
   (`crimewatch`).
4. DagsHub will mirror your GitHub repo and give you a DVC remote + MLflow
   tracking server automatically.

---

## Step 2 — Get your DagsHub token

1. Go to https://dagshub.com/user/settings/tokens
2. Click **Generate new token**, name it `crimewatch`, copy it.

---

## Step 3 — Configure DVC remote

Run these commands inside your project folder:

```bash
dvc init                          # only needed if first time

dvc remote add origin https://dagshub.com/<your-dagshub-user>/crimewatch.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <your-dagshub-user>
dvc remote modify origin --local password <your-dagshub-token>

dvc remote default origin
```

Verify:
```bash
dvc remote list
# origin    https://dagshub.com/<user>/crimewatch.dvc
```

---

## Step 4 — Add data and models to DVC

```bash
# Track raw data
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "dvc: track raw dataset"

# Track model weights
dvc add models/vehicle_detection.pt models/license_plate.pt models/yolov8n.pt
git add models/*.dvc .gitignore
git commit -m "dvc: track pretrained model weights"

# Push data to DagsHub
dvc push
```

---

## Step 5 — Set MLflow environment variables

Add these to your shell (`~/.bashrc`, `~/.zshrc`) or a `.env` file:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<your-dagshub-user>/crimewatch.mlflow
export MLFLOW_TRACKING_USERNAME=<your-dagshub-user>
export MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

Then reload:
```bash
source ~/.bashrc    # or source ~/.zshrc
```

Verify MLflow can connect:
```python
import mlflow, os
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
print(mlflow.get_tracking_uri())
```

---

## Step 6 — Run the full pipeline

```bash
dvc repro
```

After it finishes, push the DVC lock file and metrics:

```bash
dvc push
git add dvc.lock metrics.json
git commit -m "pipeline: first full run"
git push
```

---

## Step 7 — Verify on DagsHub

- **DVC**: Open your DagsHub repo → **Data** tab. You should see `data/raw`
  and `models/` listed.
- **MLflow**: Open your DagsHub repo → **Experiments** tab. You should see
  the `crimewatch-firearm-detection` experiment with at least 3 runs
  (baseline, main, evaluation).
- **Model Registry**: Click **Models** in the MLflow UI. You should see
  `crimewatch-gun-detector` version 1 in **Staging**.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `dvc push` returns 401 | Re-check token and username in `dvc remote modify` |
| `MLFLOW_TRACKING_URI not set` | Run `export` commands or source your `.env` |
| `dvc repro` skips all stages | A dependency changed — run `dvc status` to see what |
| `model.train()` crashes on GPU | Set `device: "cpu"` in `params.yaml` for testing |
