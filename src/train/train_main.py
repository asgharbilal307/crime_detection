"""
Stage 4b — Main Model Training (YOLOv8s)
Trains the full YOLOv8s as the primary model.
Logs all hyperparameters and metrics to MLflow (DagsHub remote).

Usage:
    python src/train/train_main.py
    python src/train/train_main.py --params params.yaml
"""

import argparse
import os
import shutil
from pathlib import Path

import mlflow
import mlflow.pyfunc
import yaml
from ultralytics import YOLO


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def setup_mlflow(params: dict):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise EnvironmentError(
            "MLFLOW_TRACKING_URI is not set.\n"
            "Export it before running:\n"
            "  export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow\n"
            "  export MLFLOW_TRACKING_USERNAME=<dagshub_user>\n"
            "  export MLFLOW_TRACKING_PASSWORD=<dagshub_token>"
        )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    print(f"  MLflow tracking URI : {tracking_uri}")
    print(f"  Experiment          : {params['mlflow']['experiment_name']}")


def train(params: dict) -> Path:
    mp = params["main"]
    dp = params["data"]
    mlflow_params = params["mlflow"]

    yaml_path = Path(dp["output_root"]) / "gun_detection.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at {yaml_path}. Run prepare.py first."
        )

    checkpoint_dir = Path("outputs/checkpoints/main")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("MAIN MODEL TRAINING — YOLOv8s")
    print("=" * 60)
    print(f"  Model     : {mp['model_size']}")
    print(f"  Epochs    : {mp['epochs']}")
    print(f"  Img size  : {mp['img_size']}")
    print(f"  Batch     : {mp['batch_size']}")
    print(f"  Device    : {mp['device']}")
    print(f"  YAML      : {yaml_path}\n")

    with mlflow.start_run(run_name="main-yolov8s") as run:
        # Log all hyperparameters
        mlflow.log_params({
            "model_size": mp["model_size"],
            "epochs": mp["epochs"],
            "img_size": mp["img_size"],
            "batch_size": mp["batch_size"],
            "patience": mp["patience"],
            "conf_threshold": mp["conf_threshold"],
            "frame_skip": dp["frame_skip"],
            "train_split": dp["train_split"],
            "model_type": "main",
        })

        mlflow.set_tag("model_type", "main")
        mlflow.set_tag("architecture", "yolov8s")

        last_checkpoint = checkpoint_dir / "gun_detection" / "weights" / "last.pt"
        if last_checkpoint.exists():
            print(f"  Resuming from checkpoint: {last_checkpoint}\n")
            model = YOLO(str(last_checkpoint))
            results = model.train(resume=True)
        else:
            model = YOLO(f"{mp['model_size']}.pt")
            results = model.train(
                data=str(yaml_path),
                epochs=mp["epochs"],
                imgsz=mp["img_size"],
                batch=mp["batch_size"],
                patience=mp["patience"],
                project=str(checkpoint_dir),
                name="gun_detection",
                save_period=10,
                device=mp["device"],
            )

        # --- Validation metrics ---
        metrics = model.val()
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)

        mlflow.log_metrics({
            "val/mAP50": map50,
            "val/mAP50-95": map50_95,
            "val/precision": precision,
            "val/recall": recall,
        })

        class_names = ["Handgun", "Machine_Gun"]
        for i, name in enumerate(class_names):
            mlflow.log_metric(f"val/AP50_{name}", float(metrics.box.ap50[i]))

        print(f"\n  mAP50       : {map50:.4f}")
        print(f"  mAP50-95    : {map50_95:.4f}")
        print(f"  Precision   : {precision:.4f}")
        print(f"  Recall      : {recall:.4f}")

        # --- Save best model ---
        best_pt = checkpoint_dir / "gun_detection" / "weights" / "best.pt"
        final_dir = Path("models")
        final_dir.mkdir(exist_ok=True)
        final_path = final_dir / "main_best.pt"

        if best_pt.exists():
            shutil.copy(best_pt, final_path)
            mlflow.log_artifact(str(final_path), artifact_path="model")
            print(f"\n  Best model saved : {final_path}")

        # --- Register model in MLflow Model Registry ---
        model_uri = f"runs:/{run.info.run_id}/model"
        registered = mlflow.register_model(
            model_uri=model_uri,
            name=mlflow_params["registered_model_name"],
        )

        # Transition to Staging
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=mlflow_params["registered_model_name"],
            version=registered.version,
            stage="Staging",
        )

        print(f"\n  Model registered  : {mlflow_params['registered_model_name']}")
        print(f"  Version           : {registered.version}")
        print(f"  Stage             : Staging")
        print(f"  MLflow Run ID     : {run.info.run_id}")

    return final_path


def main():
    parser = argparse.ArgumentParser(description="CrimeWatch — Main Model Training")
    parser.add_argument("--params", type=str, default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    setup_mlflow(params)
    train(params)

    print("\n[OK] Main model training complete.")


if __name__ == "__main__":
    main()
