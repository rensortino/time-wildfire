#!/usr/bin/env python3
"""
Test SAM3 object tracking capabilities on the validation dataset.
For each sequence, uses the first frame's bounding box as grounding
and tracks the object in subsequent frames.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from typing import List, Dict

try:
    from transformers import Sam3TrackerVideoProcessor, Sam3TrackerVideoModel
    import torch
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    print("Warning: SAM3 transformers not available. Please install: pip install transformers")


def xywh2xyxy_normalized(bbox_xywh: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """
    Convert normalized xywh bbox to pixel coordinates xyxy format.
    
    Args:
        bbox_xywh: Normalized bbox [x_center, y_center, width, height] in [0, 1]
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        Bbox in pixel coordinates [x_min, y_min, x_max, y_max]
    """
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


def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: Bounding box [x_min, y_min, x_max, y_max]
        bbox2: Bounding box [x_min, y_min, x_max, y_max]
    
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection area
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union area
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def calculate_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Calculate Average Precision (AP) using the 11-point interpolation method.
    
    Args:
        precision: Array of precision values
        recall: Array of recall values
    
    Returns:
        Average Precision value
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    return ap


def calculate_ap_metrics(predictions: List[np.ndarray], ground_truths: List[np.ndarray], 
                         iou_threshold: float = 0.1) -> Dict[str, float]:
    """
    Calculate AP metrics for tracking results.
    
    Args:
        predictions: List of predicted bounding boxes
        ground_truths: List of ground truth bounding boxes
        iou_threshold: IoU threshold for considering a detection as correct
    
    Returns:
        Dictionary with AP metrics
    """
    if len(predictions) == 0 or len(ground_truths) == 0:
        return {
            'ap': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    # Calculate IoU for each prediction
    ious = []
    fn = 0
    for pred, gt in zip(predictions, ground_truths):
        if pred.size > 0 and gt.size > 0:
            iou = calculate_iou(pred, gt)
            ious.append(iou)
        elif pred.size == 0 and gt.size > 0:
            ious.append(0.0)
            fn += 1
        else:
            ious.append(0.0)
    
    ious = np.array(ious)
    
    # Calculate precision and recall
    tp = np.sum(ious >= iou_threshold)
    fp = len(predictions) - tp - fn
    # fn = len(ground_truths) - tp
    # fn = abs(len(ground_truths) - len(predictions))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate AP using different IoU thresholds
    ap_values = []
    for threshold in np.arange(0.5, 1.0, 0.05):
        tp_at_threshold = np.sum(ious >= threshold)
        fp_at_threshold = len(predictions) - tp_at_threshold
        
        prec_at_threshold = tp_at_threshold / (tp_at_threshold + fp_at_threshold) if (tp_at_threshold + fp_at_threshold) > 0 else 0.0
        
        ap_values.append(prec_at_threshold)
    
    ap = np.mean(ap_values) if ap_values else 0.0
    
    return {
        'ap': float(ap),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mean_iou': float(np.mean(ious)) if len(ious) > 0 else 0.0,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def mask_to_bbox(mask) -> np.ndarray:
    """
    Convert a mask to a bounding box.
    
    Args:
        mask: Mask can be a numpy array, dict with mask data, or RLE encoded
    
    Returns:
        Bounding box [x_min, y_min, x_max, y_max] or empty array
    """
    if mask is None:
        return np.array([])
    
    # If mask is a dict, try to extract the actual mask data
    if isinstance(mask, dict):
        if "mask" in mask:
            mask = mask["mask"]
        elif "segmentation" in mask:
            mask = mask["segmentation"]
        elif "rle" in mask:
            # Handle RLE encoding if needed (would require pycocotools)
            return np.array([])
    
    # Convert mask to numpy array if it's not already
    if not isinstance(mask, np.ndarray):
        try:
            # Handle torch tensors
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            elif hasattr(mask, 'numpy'):
                mask = mask.numpy()
            else:
                mask = np.array(mask)
        except Exception:
            return np.array([])
    
    # Handle different mask formats
    if mask.ndim == 2:
        # Binary mask
        coords = np.where(mask > 0)
    elif mask.ndim == 3:
        # Multi-channel mask, use first channel or combine
        if mask.shape[2] == 1:
            coords = np.where(mask[:, :, 0] > 0)
        else:
            # Combine all channels
            mask_combined = np.any(mask > 0, axis=2)
            coords = np.where(mask_combined)
    else:
        return np.array([])
    
    if len(coords[0]) > 0:
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        return np.array([x_min, y_min, x_max, y_max])
    else:
        return np.array([])


def bbox_to_points(bbox: np.ndarray) -> List[List[int]]:
    """
    Convert bounding box to center point for SAM3 tracking.
    SAM3 uses points as prompts, so we'll use the center of the bbox.
    
    Args:
        bbox: Bounding box [x_min, y_min, x_max, y_max]
    
    Returns:
        List of point coordinates [[x, y]]
    """
    x_min, y_min, x_max, y_max = bbox
    center_x = int((x_min + x_max) / 2)
    center_y = int((y_min + y_max) / 2)
    return [[center_x, center_y]]


def track_sequence_sam3(image_paths: List[str], first_frame_bbox: np.ndarray, 
                        root_dir: str, processor, model, device: str) -> List[np.ndarray]:
    """
    Track an object in a sequence of images using SAM3 from Hugging Face.
    
    Args:
        image_paths: List of relative image paths
        first_frame_bbox: Bounding box in pixel coordinates [x_min, y_min, x_max, y_max]
        root_dir: Root directory for images
        processor: SAM3TrackerVideoProcessor instance
        model: SAM3TrackerVideoModel instance
        device: Device to run inference on ('cuda' or 'cpu')
    
    Returns:
        List of predicted bounding boxes for each frame
    """
    # Load all images as PIL Images
    full_image_paths = [os.path.join(root_dir, img_path) for img_path in image_paths]
    video_frames = []
    original_sizes = []
    
    for img_path in full_image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            original_sizes.append(img.size)  # (width, height)
            video_frames.append(img)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
    
    if len(video_frames) == 0:
        return [np.array([]) for _ in range(len(image_paths))]
    
    # Initialize video session
    try:
        inference_session = processor.init_video_session(
            video=video_frames,
            inference_device=device,
            dtype=torch.float32,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize SAM3 video session: {e}")
    
    # Convert bbox to box prompt format
    # Format: input_boxes needs 3 levels: image level, box level, box coordinates
    # Box coordinates: x_min, y_min, x_max, y_max
    x_min, y_min, x_max, y_max = first_frame_bbox
    # 3 levels: image level, box level, box coordinates
    box_coords = [float(x_min), float(y_min), float(x_max), float(y_max)]
    input_boxes = [[box_coords]]
    obj_ids = [1]  # Single object to track
    
    # Add bounding box input to inference session for first frame (condition only, no prediction)
    try:
        processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=0,
            obj_ids=obj_ids,
            input_boxes=input_boxes,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to add inputs to SAM3 session: {e}")
    
    # Store results for all frames
    predicted_bboxes = []
    
    # Store the first frame bbox directly (it's the condition, not a prediction)
    predicted_bboxes.append(first_frame_bbox.copy())
    
    # Propagate tracking through remaining frames (starting from frame 1, skipping frame 0)
    try:
        for sam3_tracker_video_output in model.propagate_in_video_iterator(inference_session, start_frame_idx=0):
            frame_idx = sam3_tracker_video_output.frame_idx
            
            # Skip frame 0 since we only added the bounding box condition, no prediction
            if frame_idx == 0:
                continue
            
            video_res_masks = processor.post_process_masks(
                [sam3_tracker_video_output.pred_masks],
                original_sizes=[[inference_session.video_height, inference_session.video_width]],
                binarize=True
            )[0]
            
            # Convert mask to bbox
            if len(video_res_masks) > 0:
                mask = video_res_masks[0]
                bbox = mask_to_bbox(mask.squeeze())
            else:
                bbox = np.array([])
            
            # Ensure we have the right number of bboxes
            while len(predicted_bboxes) <= frame_idx:
                predicted_bboxes.append(np.array([]))
            
            predicted_bboxes[frame_idx] = bbox
    except Exception as e:
        raise RuntimeError(f"Failed to propagate tracking with SAM3: {e}")
    
    # Ensure we have a bbox for each frame
    while len(predicted_bboxes) < len(image_paths):
        predicted_bboxes.append(np.array([]))
    
    return predicted_bboxes


def main():
    """Main function to test SAM3 tracking."""
    if not SAM3_AVAILABLE:
        print("Error: SAM3 transformers is not available. Please install it first:")
        print("  pip install transformers")
        return
    
    # Configuration
    csv_path = "data/val_dataset/dataset.csv"
    root_dir = "data/val_dataset"
    
    print(f"Loading dataset from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Process bbox column: convert string to numpy array
    def process_bbox(bbox_str):
        if pd.isna(bbox_str) or bbox_str == '':
            return np.array([])
        bbox_array = np.array(bbox_str.split(), dtype=np.float32)
        return bbox_array  # Keep as normalized xywh
    
    df['bbox'] = df['bbox'].apply(process_bbox)
    
    # Group by sequence
    grouped = df.groupby('sequence', sort=False)
    
    print(f"Found {len(grouped)} sequences")
    
    # Initialize SAM3 from Hugging Face
    print("Initializing SAM3 from Hugging Face...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'mps' if torch.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        processor = Sam3TrackerVideoProcessor.from_pretrained('facebook/sam3')
        model = Sam3TrackerVideoModel.from_pretrained('facebook/sam3')
        model = model.to(device)
        model.eval()
        print("SAM3 model loaded successfully")
    except Exception as e:
        print(f"Error loading SAM3 model: {e}")
        print("Make sure you have transformers installed: pip install transformers")
        return
    
    # Process each sequence
    all_metrics = []
    sequence_results = []
    
    for sequence_name, group_df in tqdm(grouped, desc="Processing sequences"):
        # Sort by image path to maintain temporal order
        sequence_df = group_df.sort_values('image_path').reset_index(drop=True)
        
        if len(sequence_df) < 2:
            print(f"Skipping sequence {sequence_name}: less than 2 frames")
            continue
        
        # Get first frame info
        first_row = sequence_df.iloc[0]
        first_image_path = first_row['image_path']
        first_bbox_normalized = first_row['bbox']
        
        if first_bbox_normalized.size == 0:
            print(f"Skipping sequence {sequence_name}: no bbox in first frame")
            continue
        
        # Load first image to get dimensions
        first_image_full_path = os.path.join(root_dir, first_image_path)
        try:
            first_image = Image.open(first_image_full_path)
            img_width, img_height = first_image.size
        except Exception as e:
            print(f"Error loading first image {first_image_full_path}: {e}")
            continue
        
        # Convert first frame bbox to pixel coordinates
        first_bbox_px = xywh2xyxy_normalized(first_bbox_normalized, img_width, img_height)
        
        # Get all image paths in sequence
        image_paths = sequence_df['image_path'].tolist()
        
        # Get ground truth bboxes for all frames
        ground_truth_bboxes = []
        for _, row in sequence_df.iterrows():
            bbox_normalized = row['bbox']
            if bbox_normalized.size > 0:
                # Load image to get dimensions (assuming same size)
                img_path = os.path.join(root_dir, row['image_path'])
                try:
                    img = Image.open(img_path)
                    w, h = img.size
                    bbox_px = xywh2xyxy_normalized(bbox_normalized, w, h)
                    ground_truth_bboxes.append(bbox_px)
                except Exception:
                    ground_truth_bboxes.append(np.array([]))
            else:
                ground_truth_bboxes.append(np.array([]))
        
        # Track using SAM3
        try:
            predicted_bboxes = track_sequence_sam3(
                image_paths, 
                first_bbox_px, 
                root_dir, 
                processor,
                model,
                device
            )
        except Exception as e:
            print(f"Error tracking sequence {sequence_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Ensure predicted and ground truth bboxes have the same length
        num_frames = len(image_paths)
        if len(predicted_bboxes) != num_frames:
            print(f"Warning: predicted_bboxes length ({len(predicted_bboxes)}) != num_frames ({num_frames})")
            # Pad or truncate to match
            if len(predicted_bboxes) < num_frames:
                predicted_bboxes.extend([np.array([])] * (num_frames - len(predicted_bboxes)))
            else:
                predicted_bboxes = predicted_bboxes[:num_frames]
        
        if len(ground_truth_bboxes) != num_frames:
            print(f"Warning: ground_truth_bboxes length ({len(ground_truth_bboxes)}) != num_frames ({num_frames})")
            # Pad or truncate to match
            if len(ground_truth_bboxes) < num_frames:
                ground_truth_bboxes.extend([np.array([])] * (num_frames - len(ground_truth_bboxes)))
            else:
                ground_truth_bboxes = ground_truth_bboxes[:num_frames]
        
        # Calculate metrics
        metrics = calculate_ap_metrics(predicted_bboxes, ground_truth_bboxes)
        metrics['sequence'] = sequence_name
        metrics['num_frames'] = len(image_paths)
        
        all_metrics.append(metrics)
        sequence_results.append({
            'sequence': sequence_name,
            'metrics': metrics,
            'predictions': [bbox.tolist() if bbox.size > 0 else [] for bbox in predicted_bboxes],
            'ground_truths': [bbox.tolist() if bbox.size > 0 else [] for bbox in ground_truth_bboxes]
        })
    
    # Calculate overall metrics
    if len(all_metrics) > 0:
        overall_metrics = {
            'mean_ap': np.mean([m['ap'] for m in all_metrics]),
            'mean_precision': np.mean([m['precision'] for m in all_metrics]),
            'mean_recall': np.mean([m['recall'] for m in all_metrics]),
            'mean_f1': np.mean([m['f1'] for m in all_metrics]),
            'mean_iou': np.mean([m['mean_iou'] for m in all_metrics]),
            'total_sequences': len(all_metrics),
            'total_tp': sum([m['tp'] for m in all_metrics]),
            'total_fp': sum([m['fp'] for m in all_metrics]),
            'total_fn': sum([m['fn'] for m in all_metrics])
        }
        
        print("\n" + "="*60)
        print("SAM3 TRACKING RESULTS")
        print("="*60)
        print(f"Total sequences processed: {overall_metrics['total_sequences']}")
        print(f"Mean AP: {overall_metrics['mean_ap']:.4f}")
        print(f"Mean Precision: {overall_metrics['mean_precision']:.4f}")
        print(f"Mean Recall: {overall_metrics['mean_recall']:.4f}")
        print(f"Mean F1: {overall_metrics['mean_f1']:.4f}")
        print(f"Mean IoU: {overall_metrics['mean_iou']:.4f}")
        print(f"\nTotal TP: {overall_metrics['total_tp']}")
        print(f"Total FP: {overall_metrics['total_fp']}")
        print(f"Total FN: {overall_metrics['total_fn']}")
        
        # Save results to JSON
        output_file = "results/sam3_tracking_iou_0.1_results.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        results = {
            'overall_metrics': overall_metrics,
            'per_sequence_metrics': all_metrics,
            'detailed_results': sequence_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    else:
        print("No sequences were successfully processed.")


if __name__ == "__main__":
    main()
