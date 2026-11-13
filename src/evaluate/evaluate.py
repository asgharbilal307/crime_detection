"""
Stage 5 — Model Evaluation
Loads both baseline (YOLOv8n) and main (YOLOv8s) models,
runs validation, logs a side-by-side comparison to MLflow,
and writes a JSON report for DVC metrics tracking.

Usage:
    python src/evaluate/evaluate.py
    python src/evaluate/evaluate.py --params params.yaml
"""

import argparse
import json
import os
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def setup_mlflow(params: dict):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise EnvironmentError(
            "MLFLOW_TRACKING_URI is not set. See README for setup instructions."
        )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])


def evaluate_model(model_path: Path, yaml_path: Path, label: str) -> dict:
    """Run YOLO validation and return a metrics dict."""
    print(f"\n  Evaluating {label} ({model_path.name})...")
    if not model_path.exists():
        print(f"  [SKIP] Model not found: {model_path}")
        return {}

    model = YOLO(str(model_path))
    metrics = model.val(data=str(yaml_path), verbose=False)

    result = {
        "model": label,
        "model_path": str(model_path),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "AP50_Handgun": round(float(metrics.box.ap50[0]), 4),
        "AP50_Machine_Gun": round(float(metrics.box.ap50[1]), 4),
    }

    print(f"    mAP50          : {result['mAP50']}")
    print(f"    mAP50-95       : {result['mAP50_95']}")
    print(f"    Precision      : {result['precision']}")
    print(f"    Recall         : {result['recall']}")
    print(f"    AP50 Handgun   : {result['AP50_Handgun']}")
    print(f"    AP50 MachineGun: {result['AP50_Machine_Gun']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="CrimeWatch — Evaluation")
    parser.add_argument("--params", type=str, default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    setup_mlflow(params)

    dp = params["data"]
    yaml_path = Path(dp["output_root"]) / "gun_detection.yaml"

    baseline_path = Path("models/baseline_best.pt")
    main_path = Path("models/main_best.pt")

    print("=" * 60)
    print("STAGE 5 — MODEL EVALUATION")
    print("=" * 60)

    with mlflow.start_run(run_name="evaluation-comparison"):
        baseline_metrics = evaluate_model(baseline_path, yaml_path, "baseline-yolov8n")
        main_metrics = evaluate_model(main_path, yaml_path, "main-yolov8s")

        # Log comparison metrics to MLflow
        if baseline_metrics:
            for k, v in baseline_metrics.items():
                if isinstance(v, float):
                    mlflow.log_metric(f"baseline/{k}", v)

        if main_metrics:
            for k, v in main_metrics.items():
                if isinstance(v, float):
                    mlflow.log_metric(f"main/{k}", v)

        # Log improvement delta
        if baseline_metrics and main_metrics:
            delta_map50 = main_metrics["mAP50"] - baseline_metrics["mAP50"]
            delta_map50_95 = main_metrics["mAP50_95"] - baseline_metrics["mAP50_95"]
            mlflow.log_metric("improvement/delta_mAP50", round(delta_map50, 4))
            mlflow.log_metric("improvement/delta_mAP50_95", round(delta_map50_95, 4))

            print(f"\n  Improvement (main vs baseline):")
            print(f"    mAP50 delta    : {delta_map50:+.4f}")
            print(f"    mAP50-95 delta : {delta_map50_95:+.4f}")

        # Write DVC metrics file
        report = {
            "baseline": baseline_metrics,
            "main": main_metrics,
        }
        metrics_path = Path("metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(report, f, indent=2)

        mlflow.log_artifact(str(metrics_path))
        print(f"\n  Metrics written to : {metrics_path}")

    print("\n[OK] Evaluation complete.")


if __name__ == "__main__":
    main()
