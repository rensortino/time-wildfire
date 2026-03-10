#!/usr/bin/env python3
"""
CLI script for SAM3 bounding box tracking on an image sequence.

Inputs
------
--images-dir   Directory of images whose filenames sort alphabetically
               in sequence order.
--labels       Ground-truth annotations.  Accepts two formats:

               1. A single YOLO-format txt file with one line per frame:
                      class_id x_center y_center width height
                  (normalised 0-1 coordinates).

               2. A directory of per-frame YOLO txt files (standard YOLO
                  layout).  Each file is matched to its image by stem
                  name (e.g. frame_001.txt ↔ frame_001.jpg).

               In both cases the **first frame's** bounding box is used
               as the tracking prompt.  All other entries are optional
               ground-truth annotations drawn for comparison.

--output-dir   Directory where annotated images are saved.

Outputs
-------
For every frame the script writes an image with:
  * a **blue** box   – ground-truth bounding box (if available)
  * a **yellow** box – SAM3-predicted bounding box

Usage
-----
    # Labels directory (YOLO layout: one .txt per frame)
    python run_tracking.py \\
        --images-dir data/val_dataset/wildfire/seq_name/images \\
        --labels     data/val_dataset/wildfire/seq_name/labels \\
        --output-dir output/seq_name/

    # Single labels file (one line per frame)
    python run_tracking.py \\
        --images-dir data/sequence_001/ \\
        --labels     data/sequence_001/labels.txt \\
        --output-dir output/sequence_001/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from sam3_tracker import SAM3Tracker, draw_bbox, draw_legend
from utils import xywh2xyxy_normalized


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_image_paths(images_dir: str) -> list[Path]:
    """Return image paths sorted alphabetically."""
    paths = sorted(
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        sys.exit(f"No images found in {images_dir}")
    return paths


def parse_single_label_file(label_path: Path) -> Optional[dict]:
    """Parse one YOLO-format label file and return the first object entry."""
    if not label_path.exists():
        return None
    text = label_path.read_text().strip()
    if not text:
        return None
    line = text.splitlines()[0].strip()
    parts = line.split()
    if len(parts) < 5:
        return None
    class_id = int(parts[0])
    bbox = np.array([float(x) for x in parts[1:5]], dtype=np.float32)
    return {"class_id": class_id, "bbox_xywh": bbox}


def load_labels(labels_path: str, image_paths: list[Path]) -> list[Optional[dict]]:
    """
    Load labels from either a single multi-line file or a directory of
    per-frame files.  Returns a list aligned with *image_paths* (same
    length); entries are ``None`` when no annotation is available.
    """
    p = Path(labels_path)

    if p.is_dir():
        entries: list[Optional[dict]] = []
        for img_path in image_paths:
            label_file = p / (img_path.stem + ".txt")
            entries.append(parse_single_label_file(label_file))
        return entries

    # Single file: one line per frame
    entries = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                entries.append(None)
                continue
            parts = line.split()
            if len(parts) < 5:
                entries.append(None)
                continue
            class_id = int(parts[0])
            bbox = np.array([float(x) for x in parts[1:5]], dtype=np.float32)
            entries.append({"class_id": class_id, "bbox_xywh": bbox})

    # Pad to match image count
    while len(entries) < len(image_paths):
        entries.append(None)
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3 bounding box tracking on an image sequence."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing the image sequence (sorted alphabetically).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help=(
            "YOLO-format labels: either a single .txt file (one line per "
            "frame) or a directory of per-frame .txt files.  The first "
            "frame's bbox is used as the SAM3 tracking prompt."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write annotated output images.",
    )
    args = parser.parse_args()

    image_paths = collect_image_paths(args.images_dir)
    labels = load_labels(args.labels, image_paths)
    os.makedirs(args.output_dir, exist_ok=True)

    n_labels = sum(1 for e in labels if e is not None)
    print(f"Found {len(image_paths)} images in {args.images_dir}")
    print(f"Found {n_labels} labels from {args.labels}")

    # --- Load images --------------------------------------------------------
    images: list[Image.Image] = []
    for p in image_paths:
        images.append(Image.open(p).convert("RGB"))

    # --- Parse first-frame bbox (tracking prompt) ---------------------------
    if not labels or labels[0] is None:
        sys.exit("The first frame must have a valid bounding box label.")

    first_label = labels[0]
    w, h = images[0].size
    first_bbox_xyxy = xywh2xyxy_normalized(first_label["bbox_xywh"], w, h).tolist()
    print(
        f"First-frame bbox (xyxy px): "
        f"[{first_bbox_xyxy[0]:.1f}, {first_bbox_xyxy[1]:.1f}, "
        f"{first_bbox_xyxy[2]:.1f}, {first_bbox_xyxy[3]:.1f}]"
    )

    # --- Run SAM3 tracking --------------------------------------------------
    print("Loading SAM3 model...")
    tracker = SAM3Tracker()
    print("Running tracking...")
    predicted_bboxes = tracker.track(images, first_bbox_xyxy)
    print(f"Tracking complete – got predictions for {len(predicted_bboxes)} / {len(images)} frames.")

    # --- Build ground-truth lookup (convert normalised xywh -> pixel xyxy) --
    gt_bboxes: dict[int, list[float]] = {}
    for idx, entry in enumerate(labels):
        if idx >= len(images):
            break
        if entry is None:
            continue
        iw, ih = images[idx].size
        gt_bboxes[idx] = xywh2xyxy_normalized(entry["bbox_xywh"], iw, ih).tolist()

    # --- Annotate and save --------------------------------------------------
    legend_entries = [("GT", "blue"), ("Pred", "yellow")]

    for idx, (img, img_path) in enumerate(zip(images, image_paths)):
        annotated = img.copy()

        if idx in gt_bboxes:
            annotated = draw_bbox(annotated, gt_bboxes[idx], color="blue", width=3)

        if idx in predicted_bboxes:
            annotated = draw_bbox(annotated, predicted_bboxes[idx], color="yellow", width=3)

        annotated = draw_legend(annotated, legend_entries)

        out_path = Path(args.output_dir) / img_path.name
        annotated.save(out_path)

    print(f"Annotated images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
