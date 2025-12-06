#!/usr/bin/env python3
"""
Script to extract CSV from tree structure containing images and labels.
CSV fields: image_path, label, bbox, cls_name, sequence
"""

import csv
from pathlib import Path
from typing import List, Tuple


def parse_label_file(label_path: Path) -> List[Tuple[int, str]]:
    """
    Parse a label file and return list of (label_index, bbox_string) tuples.
    
    Each line in the label file contains: label_index x_center y_center width height
    """
    bboxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    label_index = int(parts[0])
                    bbox_coords = ' '.join(parts[1:5])  # x_center y_center width height
                    bboxes.append((label_index, bbox_coords))
    except Exception as e:
        print(f"Warning: Could not read label file {label_path}: {e}")
    return bboxes


def extract_dataset_to_csv(val_dataset_path: str, output_csv: str):
    """
    Extract dataset information to CSV file.
    
    Args:
        val_dataset_path: Path to the val_dataset directory
        output_csv: Path to output CSV file
    """
    val_dataset_path = Path(val_dataset_path)
    
    rows = []
    empty_files = 0

    # Walk through the directory structure
    for class_dir in val_dataset_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        cls_name = class_dir.name
        
        # Find all subdirectories (some classes have nested dirs, some don't)
        subdirs = []
        for item in class_dir.iterdir():
            # Skip hidden files like .DS_Store
            if item.name.startswith('.'):
                continue
            if item.is_dir():
                subdirs.append(item)
        
        # If no subdirs found, check if images/labels are directly in class_dir
        if not subdirs:
            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"
            if images_dir.exists() and labels_dir.exists():
                subdirs = [class_dir]
        
        for subdir in subdirs:
            images_dir = subdir / "images"
            labels_dir = subdir / "labels"
            
            # Handle case where images/labels might be directly in subdir
            if not images_dir.exists():
                images_dir = subdir
            if not labels_dir.exists():
                labels_dir = subdir
            
            if not images_dir.exists():
                continue
            
            # Extract sequence name (folder name under class directory)
            # If subdir is the class_dir itself, use empty string
            sequence_name = subdir.name if subdir != class_dir else ''
            
            # Find all image files
            for image_file in images_dir.glob("*.jpg"):
                # Get relative path from dataset folder
                relative_image_path = str(image_file.relative_to(val_dataset_path))
                
                # Find corresponding label file
                label_file = labels_dir / f"{image_file.stem}.txt"
                
                if not label_file.exists():
                    # If no label file, create a row with empty label/bbox
                    rows.append({
                        'image_path': relative_image_path,
                        'label': '',
                        'bbox': '',
                        'cls_name': cls_name,
                        'sequence': sequence_name
                    })
                else:
                    # Parse label file
                    bboxes = parse_label_file(label_file)
                    
                    if not bboxes:
                        # Empty label file - print warning
                        print(f"Warning: Empty label file: {label_file}, skipping")
                        empty_files += 1
                    else:
                        # Create a row for each bbox
                        for label_index, bbox_coords in bboxes:
                            rows.append({
                                'image_path': relative_image_path,
                                'label': str(label_index),
                                'bbox': bbox_coords,
                                'cls_name': cls_name,
                                'sequence': sequence_name
                            })
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'label', 'bbox', 'cls_name', 'sequence'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Extracted {len(rows)} rows to {output_csv}")
    print(f"Skipped {empty_files} empty label files")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract dataset to CSV')
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='data/val_dataset',
        help='Path to val_dataset directory (default: data/val_dataset)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='dataset.csv',
        help='Output CSV file path (default: dataset.csv)'
    )
    
    args = parser.parse_args()

    output = Path(args.dataset_path) / args.output
    
    extract_dataset_to_csv(args.dataset_path, str(output))

