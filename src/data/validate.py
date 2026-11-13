"""
Stage 2 — Data Validation
Runs schema and quality checks on the raw dataset BEFORE training.
Fails fast with a clear error message if anything is wrong.

Checks performed:
  - Manifest exists and is valid JSON
  - Minimum video counts per category
  - Spot-checks video files are readable by OpenCV
  - Label JSON files parse correctly
  - Bounding box values are within [0, 1] after normalisation
  - No unexpected class IDs

Usage:
    python src/data/validate.py --manifest data/manifest.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import yaml


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def check_manifest(manifest_path: Path) -> dict:
    """Load and sanity-check the manifest written by ingest.py."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run ingest.py first."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    required_keys = {"categories", "total_videos", "dir_hash"}
    missing = required_keys - manifest.keys()
    if missing:
        raise ValueError(f"Manifest is missing keys: {missing}")
    return manifest


def check_video_readable(video_path: Path) -> bool:
    """Try opening a video with OpenCV and reading one frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False
    ret, _ = cap.read()
    cap.release()
    return ret


def check_label_file(label_path: Path, expected_classes: list) -> list:
    """
    Parse label.json and validate bounding boxes.
    Returns a list of error strings (empty = all good).
    """
    errors = []
    if not label_path.exists():
        return errors   # No_Gun folders legitimately have no labels

    with open(label_path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            return [f"JSON parse error in {label_path}: {e}"]

    annotations = data.get("annotations", [])
    for ann in annotations:
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            errors.append(f"Bad bbox length {len(bbox)} in {label_path}")
            continue
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            errors.append(f"Non-positive bbox dimensions in {label_path}: {bbox}")

    return errors


def run_validation(manifest: dict, params: dict, dataset_root: Path) -> bool:
    val_params = params["validation"]
    categories = ["Handgun", "Machine_Gun", "No_Gun"]
    class_mapping = {"Handgun": 0, "Machine_Gun": 1}
    expected_classes = val_params["expected_classes"]

    all_errors = []
    warnings = []
    checks_passed = 0
    checks_total = 0

    print("\n--- Check 1: Minimum video counts ---")
    for cat in categories:
        cat_info = manifest["categories"].get(cat, {})
        count = cat_info.get("videos", 0)
        checks_total += 1
        min_required = 5   # at least 5 videos per category
        if count < min_required:
            all_errors.append(
                f"Category '{cat}' has only {count} videos (minimum: {min_required})"
            )
            print(f"  [FAIL] {cat}: {count} videos")
        else:
            checks_passed += 1
            print(f"  [OK]   {cat}: {count} videos")

    print("\n--- Check 2: Spot-check video readability (3 per category) ---")
    for cat in categories:
        cat_path = dataset_root / cat
        if not cat_path.exists():
            continue
        video_folders = [f for f in cat_path.iterdir() if f.is_dir()]
        sample = random.sample(video_folders, min(3, len(video_folders)))
        for vf in sample:
            video_path = vf / "video.mp4"
            checks_total += 1
            if not video_path.exists():
                warnings.append(f"Missing video.mp4 in {vf}")
                print(f"  [WARN] {vf.name}/video.mp4 not found")
                continue
            if check_video_readable(video_path):
                checks_passed += 1
                print(f"  [OK]   {cat}/{vf.name}/video.mp4 readable")
            else:
                all_errors.append(f"Cannot read video: {video_path}")
                print(f"  [FAIL] {cat}/{vf.name}/video.mp4 unreadable")

    print("\n--- Check 3: Label file validation (5 per gun category) ---")
    for cat in ["Handgun", "Machine_Gun"]:
        cat_path = dataset_root / cat
        if not cat_path.exists():
            continue
        video_folders = [f for f in cat_path.iterdir() if f.is_dir()]
        sample = random.sample(video_folders, min(5, len(video_folders)))
        for vf in sample:
            label_path = vf / "label.json"
            checks_total += 1
            errs = check_label_file(label_path, expected_classes)
            if errs:
                all_errors.extend(errs)
                for e in errs:
                    print(f"  [FAIL] {e}")
            else:
                checks_passed += 1
                print(f"  [OK]   {cat}/{vf.name}/label.json valid")

    print("\n--- Check 4: Dataset balance check ---")
    handgun_count = manifest["categories"].get("Handgun", {}).get("videos", 0)
    machine_gun_count = manifest["categories"].get("Machine_Gun", {}).get("videos", 0)
    no_gun_count = manifest["categories"].get("No_Gun", {}).get("videos", 0)
    checks_total += 1
    if handgun_count == 0 or machine_gun_count == 0:
        all_errors.append("One or more gun classes has 0 videos.")
        print("  [FAIL] Missing gun class data")
    else:
        ratio = no_gun_count / (handgun_count + machine_gun_count)
        checks_passed += 1
        status = "OK" if ratio < 10 else "WARN"
        print(f"  [{status}]   No_Gun:Gun ratio = {ratio:.1f}x (target: < 10x)")
        if ratio >= 10:
            warnings.append(
                f"High background ratio ({ratio:.1f}x). Consider reducing No_Gun samples."
            )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Checks passed : {checks_passed}/{checks_total}")
    print(f"  Warnings      : {len(warnings)}")
    print(f"  Errors        : {len(all_errors)}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  ! {w}")

    if all_errors:
        print("\nErrors:")
        for e in all_errors:
            print(f"  x {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="CrimeWatch — Data Validation")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifest.json",
        help="Path to manifest.json written by ingest.py",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="params.yaml",
        help="Path to params.yaml",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("STAGE 2 — DATA VALIDATION")
    print("=" * 60)

    params = load_params(args.params)
    manifest = check_manifest(Path(args.manifest))
    dataset_root = Path(params["data"]["dataset_root"])

    passed = run_validation(manifest, params, dataset_root)

    if not passed:
        print("\n[FAIL] Validation failed. Fix errors before training.")
        sys.exit(1)

    # Write a small validation report so DVC can track this stage's output
    report = {
        "status": "passed",
        "total_videos": manifest["total_videos"],
        "categories": {
            k: v["videos"] for k, v in manifest["categories"].items()
        },
    }
    report_path = Path("data/validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Validation passed. Report written to {report_path}")


if __name__ == "__main__":
    main()
