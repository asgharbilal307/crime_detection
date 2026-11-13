"""
Stage 3 — Dataset Preparation
Extracts frames from raw videos and converts annotations to YOLO format.
Writes the processed dataset to data/processed/ and a dataset YAML.

Usage:
    python src/data/prepare.py
    python src/data/prepare.py --params params.yaml
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def extract_and_convert(
    dataset_root: Path,
    output_root: Path,
    frame_skip: int = 5,
    frame_skip_no_gun: int = 50,
    train_split: float = 0.8,
) -> dict:
    """Extract frames from videos and convert bounding boxes to YOLO format."""
    for split in ["train", "val"]:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

    categories = ["Handgun", "Machine_Gun", "No_Gun"]
    class_mapping = {"Handgun": 0, "Machine_Gun": 1}

    all_videos = []
    for category in categories:
        category_path = dataset_root / category
        if category_path.exists():
            video_folders = [f for f in category_path.iterdir() if f.is_dir()]
            all_videos.extend([(category, vf) for vf in video_folders])

    random.seed(42)   # reproducible split
    random.shuffle(all_videos)
    split_idx = int(len(all_videos) * train_split)
    train_videos = all_videos[:split_idx]
    val_videos = all_videos[split_idx:]

    print(f"\n  Training videos : {len(train_videos)}")
    print(f"  Validation videos: {len(val_videos)}\n")

    stats = {
        "total_images": 0,
        "total_annotations": 0,
        "total_backgrounds": 0,
        "train_images": 0,
        "val_images": 0,
    }

    for split_name, video_list in [("train", train_videos), ("val", val_videos)]:
        print(f"Processing {split_name.upper()} split...")
        for category, video_folder in tqdm(video_list, desc=split_name):
            video_path = video_folder / "video.mp4"
            if not video_path.exists():
                continue

            label_json = video_folder / "label.json"
            label_data = {}
            if label_json.exists():
                with open(label_json) as f:
                    label_data = json.load(f)

            annotations_by_frame = {}
            if "annotations" in label_data:
                for ann in label_data["annotations"]:
                    img_id = ann.get("image_id", 0)
                    annotations_by_frame.setdefault(img_id, []).append(ann)

            cap = cv2.VideoCapture(str(video_path))
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_skip = frame_skip_no_gun if category == "No_Gun" else frame_skip
                if frame_idx % current_skip == 0:
                    json_image_id = frame_idx + 1
                    has_boxes = False
                    yolo_lines = []

                    if json_image_id in annotations_by_frame:
                        height, width = frame.shape[:2]
                        for ann in annotations_by_frame[json_image_id]:
                            bbox = ann.get("bbox", [0, 0, 0, 0])
                            if len(bbox) == 4:
                                x, y, w, h = bbox
                                if w > 0 and h > 0:
                                    x_center = max(0, min(1, (x + w / 2) / width))
                                    y_center = max(0, min(1, (y + h / 2) / height))
                                    norm_w = max(0, min(1, w / width))
                                    norm_h = max(0, min(1, h / height))
                                    class_id = class_mapping.get(category, 0)
                                    yolo_lines.append(
                                        f"{class_id} {x_center:.6f} {y_center:.6f}"
                                        f" {norm_w:.6f} {norm_h:.6f}\n"
                                    )
                                    has_boxes = True

                    should_save = False
                    if has_boxes:
                        should_save = True
                        stats["total_annotations"] += len(yolo_lines)
                    elif category == "No_Gun":
                        should_save = True
                        stats["total_backgrounds"] += 1
                    else:
                        if random.random() < 0.05:
                            should_save = True
                            stats["total_backgrounds"] += 1

                    if should_save:
                        frame_name = (
                            f"{category}_{video_folder.name}_f{frame_idx:05d}"
                        )
                        img_path = (
                            output_root / split_name / "images" / f"{frame_name}.jpg"
                        )
                        label_path = (
                            output_root / split_name / "labels" / f"{frame_name}.txt"
                        )
                        cv2.imwrite(str(img_path), frame)
                        with open(label_path, "w") as f:
                            for line in yolo_lines:
                                f.write(line)
                        stats["total_images"] += 1
                        stats[f"{split_name}_images"] += 1

                frame_idx += 1
            cap.release()

    return stats


def create_dataset_yaml(output_root: Path) -> Path:
    config = {
        "path": str(output_root.absolute()),
        "train": "train/images",
        "val": "val/images",
        "nc": 2,
        "names": {0: "Handgun", 1: "Machine_Gun"},
    }
    yaml_path = output_root / "gun_detection.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="CrimeWatch — Dataset Preparation")
    parser.add_argument("--params", type=str, default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    dp = params["data"]

    dataset_root = Path(dp["dataset_root"])
    output_root = Path(dp["output_root"])

    print("=" * 60)
    print("STAGE 3 — DATASET PREPARATION")
    print("=" * 60)
    print(f"\n  Source : {dataset_root}")
    print(f"  Output : {output_root}")

    stats = extract_and_convert(
        dataset_root=dataset_root,
        output_root=output_root,
        frame_skip=dp["frame_skip"],
        frame_skip_no_gun=dp["frame_skip_no_gun"],
        train_split=dp["train_split"],
    )

    yaml_path = create_dataset_yaml(output_root)

    # Write preparation stats for DVC tracking
    stats_path = output_root / "prepare_stats.json"
    import json
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Total images saved  : {stats['total_images']}")
    print(f"    Train             : {stats['train_images']}")
    print(f"    Val               : {stats['val_images']}")
    print(f"  Total annotations   : {stats['total_annotations']}")
    print(f"  Background frames   : {stats['total_backgrounds']}")
    print(f"  Dataset YAML        : {yaml_path}")
    print(f"  Stats written to    : {stats_path}")


if __name__ == "__main__":
    main()
