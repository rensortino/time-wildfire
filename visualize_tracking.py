#!/usr/bin/env python3
"""
Visualize tracking predictions and ground truth for a sequence of frames.
Shows a grid of frames with predictions (orange) and ground truth (blue) bounding boxes.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple, Optional


def load_results(json_path: str, sequence_name: str) -> Optional[dict]:
    """
    Load predictions and ground truths for a specific sequence from JSON results.
    
    Args:
        json_path: Path to JSON results file
        sequence_name: Name of the sequence to visualize
    
    Returns:
        Dictionary with 'predictions' and 'ground_truths' lists, or None if not found
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Search through detailed_results to find the sequence (contains predictions/ground_truths)
    for seq_data in data.get('detailed_results', []):
        if seq_data.get('sequence') == sequence_name:
            metrics = seq_data.get('metrics', {})
            return {
                'predictions': seq_data.get('predictions', []),
                'ground_truths': seq_data.get('ground_truths', []),
                'metrics': {
                    'ap': metrics.get('ap', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1': metrics.get('f1', 0),
                    'mean_iou': metrics.get('mean_iou', 0),
                }
            }
    
    return None


def load_sequence_images(csv_path: str, sequence_name: str) -> List[dict]:
    """
    Load image paths and ground truth bboxes for a sequence from CSV.
    
    Args:
        csv_path: Path to CSV dataset file
        sequence_name: Name of the sequence
    
    Returns:
        List of dictionaries with 'image_path', 'bbox', and 'class_name' keys
    """
    df = pd.read_csv(csv_path)
    
    # Filter by sequence
    sequence_df = df[df['sequence'] == sequence_name].copy()
    
    if len(sequence_df) == 0:
        return []
    
    # Sort by image path to maintain temporal order
    sequence_df = sequence_df.sort_values('image_path').reset_index(drop=True)

    class_name = sequence_df['cls_name'][0]
    
    images = []
    for _, row in sequence_df.iterrows():
        # Parse bbox from string format "x y w h" (normalized)
        bbox_str = row['bbox']
        if pd.isna(bbox_str) or bbox_str == '':
            bbox = None
        else:
            bbox = np.array([float(x) for x in bbox_str.split()])
        
        images.append({
            'image_path': row['image_path'],
            'bbox': bbox,
            'class_name': class_name,
        })
    
    return images


def xywh_to_xyxy(bbox_xywh: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """
    Convert normalized xywh bbox to pixel coordinates xyxy format.
    
    Args:
        bbox_xywh: Normalized bbox [x_center, y_center, width, height] in [0, 1]
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        Bbox in pixel coordinates [x_min, y_min, x_max, y_max]
    """
    if bbox_xywh is None or len(bbox_xywh) == 0:
        return None
    
    x_center, y_center, width, height = bbox_xywh
    
    # Convert to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Convert to xyxy format
    x_min = x_center_px - width_px / 2
    y_min = y_center_px - height_px / 2
    x_max = x_center_px + width_px / 2
    y_max = y_center_px + height_px / 2
    
    return np.array([x_min, y_min, x_max, y_max])


def parse_bbox_pixel(bbox: List[float]) -> Optional[np.ndarray]:
    """
    Parse bbox from JSON format [x_min, y_min, x_max, y_max] in pixel coordinates.
    
    Args:
        bbox: List [x_min, y_min, x_max, y_max] in pixel coordinates
    
    Returns:
        Bbox in xyxy format [x_min, y_min, x_max, y_max] or None
    """
    if bbox is None or len(bbox) == 0:
        return None
    
    # Bbox is already in xyxy format
    return np.array(bbox)


def draw_bbox_on_image(image: Image.Image, bbox: np.ndarray, color: str, label: str = None, width: int = 3):
    """
    Draw a bounding box on an image.
    
    Args:
        image: PIL Image
        bbox: Bounding box in xyxy format [x_min, y_min, x_max, y_max]
        color: Color name or RGB tuple
        label: Optional label text
        width: Line width
    """
    if bbox is None:
        return image
    
    draw = ImageDraw.Draw(image)
    
    x_min, y_min, x_max, y_max = bbox
    
    # Draw rectangle
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)
    
    # Draw label if provided
    if label:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw text background
        text_bbox = draw.textbbox((x_min, y_min - 20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        
        # Draw text
        draw.text((x_min, y_min - 20), label, fill='white', font=font)
    
    return image


def visualize_sequence(
    json_path: str,
    csv_path: str,
    sequence_name: str,
    output_path: Optional[str] = None,
    data_root: Optional[str] = None,
    max_frames: Optional[int] = None
):
    """
    Visualize predictions and ground truth for a sequence.
    
    Args:
        json_path: Path to JSON results file
        csv_path: Path to CSV dataset file
        sequence_name: Name of the sequence to visualize
        output_path: Optional path to save the visualization
        data_root: Root directory for images (defaults to CSV parent directory)
        max_frames: Maximum number of frames to display
    """
    # Load results from JSON
    results = load_results(json_path, sequence_name)
    if results is None:
        print(f"Sequence '{sequence_name}' not found in JSON results")
        return
    
    predictions = results['predictions']
    ground_truths = results['ground_truths']
    metrics = results['metrics']
    
    # Load images from CSV
    images = load_sequence_images(csv_path, sequence_name)
    if len(images) == 0:
        print(f"No images found for sequence '{sequence_name}' in CSV")
        return
    
    # Determine data root
    if data_root is None:
        data_root = str(Path(csv_path).parent)
    
    # Limit number of frames if specified
    if max_frames is not None:
        images = images[:max_frames]
        predictions = predictions[:max_frames]
        ground_truths = ground_truths[:max_frames]
    
    # Ensure we have the same number of frames
    num_frames = min(len(images), len(predictions), len(ground_truths))
    images = images[:num_frames]
    predictions = predictions[:num_frames]
    ground_truths = ground_truths[:num_frames]

    class_name = images[0]['class_name']
    
    # Load and process each frame
    processed_frames = []
    for i, img_info in enumerate(images):
        img_path = os.path.join(data_root, img_info['image_path'])
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_width, img_height = img.size
            
            # Parse prediction bbox (from JSON, pixel coordinates [x_min, y_min, x_max, y_max])
            pred_bbox = None
            if i < len(predictions) and predictions[i] and len(predictions[i]) == 4:
                pred_bbox = parse_bbox_pixel(predictions[i])
            
            # Parse ground truth bbox (from CSV, normalized [x_center, y_center, w, h])
            gt_bbox = None
            if img_info['bbox'] is not None and len(img_info['bbox']) == 4:
                gt_bbox = xywh_to_xyxy(img_info['bbox'], img_width, img_height)
            
            # Draw ground truth (blue) first so prediction (orange) is on top
            if gt_bbox is not None:
                img = draw_bbox_on_image(img, gt_bbox, color='blue', label='GT', width=2)
            
            if pred_bbox is not None:
                img = draw_bbox_on_image(img, pred_bbox, color='orange', label='Pred', width=2)
            
            processed_frames.append(img)
            
        except Exception as e:
            print(f"Error processing frame {i} ({img_path}): {e}")
            continue
    
    if len(processed_frames) == 0:
        print("No frames processed successfully")
        return
    
    # Create grid visualization
    num_frames = len(processed_frames)
    cols = min(4, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    if num_frames == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, img in enumerate(processed_frames):
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Frame {i+1}', fontsize=12)
    
    # Hide unused subplots
    for i in range(num_frames, len(axes)):
        axes[i].axis('off')
    
    # Add title with metrics
    title = f"Sequence: {sequence_name}\n"
    title += f"Class: {class_name}\n"
    title += f"AP: {metrics['ap']:.3f} | Precision: {metrics['precision']:.3f} | "
    title += f"Recall: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f} | IoU: {metrics['mean_iou']:.3f}"
    fig.suptitle(title, fontsize=14, y=0.995)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def list_sequences(json_path: str):
    """List all available sequences in the JSON results."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    sequences = []
    # Try detailed_results first (has predictions), fallback to per_sequence_metrics
    seq_data_list = data.get('detailed_results', [])
    if not seq_data_list:
        seq_data_list = data.get('per_sequence_metrics', [])
    
    for seq_data in seq_data_list:
        metrics = seq_data.get('metrics', seq_data)  # Handle nested or flat structure
        sequences.append({
            'name': seq_data.get('sequence'),
            'num_frames': metrics.get('num_frames', 0),
            'ap': metrics.get('ap', 0),
        })
    
    print(f"\nFound {len(sequences)} sequences:\n")
    for seq in sequences[:20]:  # Show first 20
        print(f"  {seq['name']} ({seq['num_frames']} frames, AP: {seq['ap']:.3f})")
    
    if len(sequences) > 20:
        print(f"\n  ... and {len(sequences) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize tracking predictions and ground truth for a sequence'
    )
    parser.add_argument(
        '--json',
        type=str,
        default='results/sam3_tracking_results.json',
        help='Path to JSON results file'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='data/val_dataset/dataset.csv',
        help='Path to CSV dataset file'
    )
    parser.add_argument(
        '--sequence',
        type=str,
        help='Name of the sequence to visualize'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save the visualization (if not provided, will display)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help='Root directory for images (defaults to CSV parent directory)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to display'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available sequences and exit'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_sequences(args.json)
        return
    
    args.output = args.output or f"results/visualizations/{args.sequence}.png"
    if not os.path.exists(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.sequence is None:
        df = pd.read_csv(args.csv)
        sequences = df['sequence'].unique()
        sequences_by_classname = df.groupby('cls_name')['sequence'].unique().to_dict()
        for classname, sequences in sequences_by_classname.items():
            seq = sequences[0]  
        # for seq in sequences:
            visualize_sequence(
                json_path=args.json,
                csv_path=args.csv,
                sequence_name=seq,
                output_path=f"results/visualizations/{classname}_{seq}.png",
            )

    visualize_sequence(
        json_path=args.json,
        csv_path=args.csv,
        sequence_name=args.sequence,
        output_path=args.output,
        data_root=args.data_root,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()
