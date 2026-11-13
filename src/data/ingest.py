"""
Stage 1 — Data Ingestion
Verifies the raw dataset is present and writes a manifest so DVC
can detect changes. Run this before validate.py.

Usage:
    python src/data/ingest.py --dataset_root data/raw
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def compute_dir_hash(directory: Path) -> str:
    """Compute a lightweight hash over all file names + sizes in a directory tree."""
    entries = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            full = Path(root) / f
            entries.append(f"{full.relative_to(directory)}:{full.stat().st_size}")
    combined = "\n".join(entries)
    return hashlib.md5(combined.encode()).hexdigest()


def build_manifest(dataset_root: Path) -> dict:
    """Walk dataset_root and collect per-category stats."""
    categories = ["Handgun", "Machine_Gun", "No_Gun"]
    manifest = {"categories": {}, "total_videos": 0}

    for cat in categories:
        cat_path = dataset_root / cat
        if not cat_path.exists():
            print(f"  [WARN] Category folder not found: {cat_path}")
            manifest["categories"][cat] = {"videos": 0, "folders": []}
            continue

        video_folders = sorted(
            [f.name for f in cat_path.iterdir() if f.is_dir()]
        )
        video_count = len(video_folders)
        manifest["categories"][cat] = {
            "videos": video_count,
            "folders": video_folders[:5],   # store first 5 as sample
        }
        manifest["total_videos"] += video_count
        print(f"  {cat}: {video_count} video folders found")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="CrimeWatch — Data Ingestion")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/raw",
        help="Path to raw dataset root (must contain Handgun/, Machine_Gun/, No_Gun/)",
    )
    parser.add_argument(
        "--manifest_out",
        type=str,
        default="data/manifest.json",
        help="Path to write the dataset manifest",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    manifest_out = Path(args.manifest_out)

    print("=" * 60)
    print("STAGE 1 — DATA INGESTION")
    print("=" * 60)

    # --- Check dataset root exists ---
    if not dataset_root.exists():
        print(f"\n[ERROR] Dataset root not found: {dataset_root}")
        print(
            "  Please download the dataset and place it at the path above,\n"
            "  or update params.yaml:data:dataset_root to point to it."
        )
        sys.exit(1)

    print(f"\nDataset root : {dataset_root.resolve()}")
    print("\nScanning categories...")

    manifest = build_manifest(dataset_root)

    if manifest["total_videos"] == 0:
        print("\n[ERROR] No video folders found. Check dataset structure.")
        sys.exit(1)

    # --- Compute directory hash for DVC change detection ---
    manifest["dir_hash"] = compute_dir_hash(dataset_root)
    manifest["dataset_root"] = str(dataset_root.resolve())

    # --- Write manifest ---
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written  : {manifest_out}")
    print(f"Total videos found: {manifest['total_videos']}")
    print(f"Directory hash    : {manifest['dir_hash']}")
    print("\n[OK] Ingestion complete.")


if __name__ == "__main__":
    main()
